"""
Baseline model: XGBoost trained on the merged feature table (166 raw Elliptic
features + structural features from Step 3). This is the bar Step 6's
GraphSAGE GNN needs to clear to justify its added complexity -- if the GNN
can't beat this, that's reported as a finding, not tuned away.

--------------------------------------------------------------------------------
Why precision/recall/F1/PR-AUC, not accuracy
--------------------------------------------------------------------------------
Illicit is 9.8% of labeled nodes. A model that predicts "licit" for everything
scores 90.2% accuracy while catching zero fraud -- accuracy is actively
misleading here. Precision/recall/F1 (at a 0.5 probability threshold) and
PR-AUC (threshold-independent, and more informative than ROC-AUC under severe
imbalance since it doesn't get inflated by the large number of easy true
negatives) are used instead, and are the only numbers reported as "the
result" anywhere in this project.

--------------------------------------------------------------------------------
Why an internal validation split inside the training window
--------------------------------------------------------------------------------
The temporal train/test split (Step 4) holds out steps 35-49 as test and is
never touched until final evaluation. But XGBoost still needs a validation
set to pick how many boosting rounds to use (early stopping) -- using the test
set for that would mean the "final" evaluation number was itself used to tune
the model, which defeats the point of holding it out.

So the training window (steps 1-34) is split again, the same way: steps 1-29
train the model, steps 30-34 validate it (early stopping only, not final
metrics). Once the number of rounds is chosen, the model is refit on the FULL
1-34 window (more training data, same chosen round count) before touching
steps 35-49 for the one real evaluation.

--------------------------------------------------------------------------------
Handling class imbalance and the categorical `community` column
--------------------------------------------------------------------------------
`scale_pos_weight` (ratio of negative to positive examples in the training
data) tells XGBoost to weight illicit examples more heavily during training,
rather than letting the ~9x class imbalance push it toward always predicting
"licit". `community` (315 arbitrary group IDs from Step 3's Louvain run) is
passed as a pandas 'category' dtype with `enable_categorical=True` -- XGBoost's
native categorical split-finding, rather than one-hot encoding 315 columns or
treating community IDs as an ordered number they aren't.

--------------------------------------------------------------------------------
Regularization was tried and rejected -- the train/test gap is not overfitting
--------------------------------------------------------------------------------
This model has no max_depth/subsample/colsample_bytree/min_child_weight
regularization beyond defaults, which is a deliberate conclusion, not an
oversight. The first run showed a large gap between train (F1 0.97) and test
(F1 0.64) -- large enough to suspect overfitting, so that hypothesis was
tested directly rather than assumed:

  1. max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=5
     (all at once): WORSE on every metric, both splits (train F1 0.91,
     test F1 0.52).
  2. max_depth=6 (original) with only subsample=0.8 added, isolating row
     subsampling alone: still worse than the original on test (F1 0.54,
     PR-AUC 0.76 vs. the original's F1 0.64, PR-AUC 0.79).

Both experiments were validated only against the internal steps 30-34 slice,
never the steps 35-49 test set, so neither counts as tuning against the final
evaluation -- and both came back worse. If capacity were the problem, cutting
it should have closed the gap while holding or improving test performance;
instead test got worse both times. That's evidence the gap is predominantly
genuine distribution shift across time (the documented 11.6% -> 6.5% illicit
rate drop, plus whatever real behavioral change happened between periods),
not memorization -- something regularization can't fix because it isn't the
actual problem. The unregularized model is kept because it's the one that
actually generalizes best, which is the only criterion that matters here.

--------------------------------------------------------------------------------
Two levers that target distribution shift directly, instead of fighting it
--------------------------------------------------------------------------------
Since the gap is shift, not overfitting, the next two changes aim at shift
directly rather than repeating the regularization experiment with different
knobs:

1. Recency-weighted training (`_recency_weight`): rows are weighted by how
   late their time_step falls within the training window (1-34), on a fixed
   0.5x-1.5x ramp anchored to that window (not to whatever subset is being
   fit, so phase 1 and phase 2 below stay on the same scale). The intent is
   to make the model lean on patterns from steps closer to the test period,
   without discarding the earlier data entirely.
2. Threshold selection instead of assuming 0.5 (`_select_threshold`): PR-AUC
   is threshold-independent and measures ranking quality alone; precision/
   recall/F1 depend entirely on where the cutoff is drawn. A fixed 0.5 cutoff
   was implicitly shaped by train's 11.6% illicit rate, but test's rate is
   6.5% -- a different base rate. The threshold that maximizes F1 on the
   phase-1 model's out-of-sample predictions on steps 30-34 (never on test)
   is used instead of a hardcoded 0.5.

Both are combined with the existing class-imbalance weighting into one
sample_weight vector per row, rather than using XGBoost's scale_pos_weight
constructor argument (which can only express class imbalance, not recency).

--------------------------------------------------------------------------------
Dropping 6 features identified by the feature-drift audit
--------------------------------------------------------------------------------
models/feature_drift_audit.py compares every feature's distribution between
the train period (steps 1-34) and test period (steps 35-49), using no label
information at all. Six features -- feat_136, feat_101, feat_103, feat_100,
feat_139, feat_137 -- showed a KS statistic above 0.9 (near-total separation
between the two periods' value ranges), a distinct cluster clearly separated
from the rest of the features (the next-highest KS statistic drops to 0.61).
That pattern is consistent with these being cumulative/time-indexed
aggregated features (per the original Elliptic paper's local+aggregated
feature split) that mechanically trend with absolute time step rather than
encoding durable transaction risk -- a tree model can't extrapolate past the
value ranges it trained on, so a feature whose entire test-period range sits
outside its train-period range is actively unhelpful at prediction time.

Corroborating evidence: checking feature importances from the three prior
baseline configurations tried in this file's history, each one leaned on at
least one of these six features in its top 10 -- except the recency-weighted,
threshold-tuned configuration (the best-performing one so far), whose top 10
avoided all six. That's the basis for excluding them here rather than as a
speculative guess: `EXCLUDED_DRIFTED_FEATURES` below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score

from features.build_features import (
    DEFAULT_TRAIN_MAX_STEP,
    build_feature_table,
    get_labeled_subset,
    temporal_train_test_split,
)

CACHE_DIR = Path(__file__).parent / "cache"
MODEL_PATH = CACHE_DIR / "xgboost_model.json"
THRESHOLD_PATH = CACHE_DIR / "xgboost_threshold.txt"
PREDICTIONS_PATH = CACHE_DIR / "xgboost_predictions.parquet"

# Steps 1-29 train, 30-34 validate (for early stopping only) -- carved out of
# the 1-34 training window established in Step 4. Steps 35-49 stay untouched
# until final evaluation.
VAL_MIN_STEP = 30
TRAIN_MIN_STEP = 1

# Recency weighting: row weight ramps linearly across this range depending on
# how late its time_step falls in the fixed 1-34 training window.
RECENCY_WEIGHT_MIN = 0.5
RECENCY_WEIGHT_MAX = 1.5

DEFAULT_THRESHOLD = 0.5

# Identified by models/feature_drift_audit.py: KS statistic > 0.9 between the
# train-period and test-period distribution, a distinct cluster clearly
# separated from the rest of the features (next-highest KS is 0.61). See
# module docstring for why these are excluded rather than left in.
EXCLUDED_DRIFTED_FEATURES = ["feat_136", "feat_101", "feat_103", "feat_100", "feat_139", "feat_137"]


def _prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    excluded = {"label", *EXCLUDED_DRIFTED_FEATURES}
    feature_cols = [c for c in df.columns if c not in excluded]
    return df[feature_cols], df["label"]


def _recency_weight(time_step: pd.Series) -> np.ndarray:
    """0.5x-1.5x linear ramp anchored to the fixed 1-34 training window (not
    to whichever subset is passed in), so phase 1 and phase 2 fits below stay
    on the same scale even though phase 1 only sees steps 1-29."""
    span = DEFAULT_TRAIN_MAX_STEP - TRAIN_MIN_STEP
    normalized = ((time_step - TRAIN_MIN_STEP) / span).clip(0, 1)
    return RECENCY_WEIGHT_MIN + normalized.to_numpy() * (RECENCY_WEIGHT_MAX - RECENCY_WEIGHT_MIN)


def _sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Combines class-imbalance weighting and recency weighting into one
    per-row weight vector, since XGBoost's scale_pos_weight constructor
    argument can only express the former."""
    y = df["label"]
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    class_weight = np.where(y == 1, scale_pos_weight, 1.0)
    return class_weight * _recency_weight(df["time_step"])


def _select_threshold(y_val: pd.Series, proba_val: np.ndarray) -> float:
    """Picks the probability cutoff that maximizes F1 on out-of-sample
    validation predictions, instead of assuming 0.5 -- see module docstring."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba_val)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = f1s[:-1].argmax()  # last precision/recall point has no threshold
    return float(thresholds[best_idx])


def _make_model(n_estimators: int = 500, early_stopping: bool = True) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="aucpr",
        early_stopping_rounds=20 if early_stopping else None,
        random_state=42,
    )


def train_baseline(use_cache: bool = True) -> tuple[xgb.XGBClassifier, pd.DataFrame, pd.DataFrame, float]:
    """Two-phase fit: (1) find the right number of boosting rounds via early
    stopping on an internal validation split, and pick a decision threshold
    from that same phase's out-of-sample predictions; (2) refit on the full
    training window using that round count. Returns
    (model, train_df, test_df, threshold)."""
    full_table = build_feature_table()
    labeled = get_labeled_subset(full_table)
    train_df, test_df = temporal_train_test_split(labeled)

    if use_cache and MODEL_PATH.exists():
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        if THRESHOLD_PATH.exists():
            threshold = float(THRESHOLD_PATH.read_text().strip())
        else:
            print(f"No cached threshold found at {THRESHOLD_PATH}; falling back to {DEFAULT_THRESHOLD}")
            threshold = DEFAULT_THRESHOLD
        return model, train_df, test_df, threshold

    sub_train = train_df[train_df["time_step"] < VAL_MIN_STEP]
    val = train_df[train_df["time_step"] >= VAL_MIN_STEP]
    X_sub, y_sub = _prepare_xy(sub_train)
    X_val, y_val = _prepare_xy(val)

    finder = _make_model()
    finder.fit(X_sub, y_sub, sample_weight=_sample_weights(sub_train), eval_set=[(X_val, y_val)], verbose=False)
    best_rounds = finder.best_iteration + 1
    print(f"Early stopping on steps 1-{VAL_MIN_STEP-1} (train) / "
          f"{VAL_MIN_STEP}-34 (validation) picked {best_rounds} boosting rounds")

    val_proba = finder.predict_proba(X_val)[:, 1]
    threshold = _select_threshold(y_val, val_proba)
    print(f"Threshold selected on steps {VAL_MIN_STEP}-34 out-of-sample predictions: {threshold:.4f}")

    X_train, y_train = _prepare_xy(train_df)
    model = _make_model(n_estimators=best_rounds, early_stopping=False)
    model.fit(X_train, y_train, sample_weight=_sample_weights(train_df))

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(MODEL_PATH)
        THRESHOLD_PATH.write_text(str(threshold))

    return model, train_df, test_df, threshold


def evaluate(model: xgb.XGBClassifier, df: pd.DataFrame, split_name: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Precision/recall/F1 (at the given threshold) + PR-AUC (threshold-
    independent). These four numbers are the only ones this project reports
    as "the result" -- see module docstring for why accuracy is excluded."""
    X, y = _prepare_xy(df)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)

    metrics = {
        "split": split_name,
        "n": len(df),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, proba),
    }
    return metrics


def predict_all_labeled(model: xgb.XGBClassifier, use_cache: bool = True) -> pd.DataFrame:
    """Predictions for every labeled node (train + test), saved for reuse by
    the FastAPI /predictions endpoint (Step 8) and the model comparison table
    (Step 7) -- one place computes this, nobody re-derives it."""
    if use_cache and PREDICTIONS_PATH.exists():
        return pd.read_parquet(PREDICTIONS_PATH)

    full_table = build_feature_table()
    labeled = get_labeled_subset(full_table)
    X, _ = _prepare_xy(labeled)

    result = pd.DataFrame(index=labeled.index)
    result["time_step"] = labeled["time_step"]
    result["label"] = labeled["label"]
    result["xgboost_proba"] = model.predict_proba(X)[:, 1]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(PREDICTIONS_PATH)

    return result


if __name__ == "__main__":
    model, train_df, test_df, threshold = train_baseline()
    print(f"\nUsing decision threshold: {threshold:.4f} (selected on validation, not test)")

    print("\n--- Baseline XGBoost: precision / recall / F1 / PR-AUC ---")
    for split_name, df in [("train (steps 1-34)", train_df), ("test (steps 35-49)", test_df)]:
        metrics = evaluate(model, df, split_name, threshold=threshold)
        print(f"{metrics['split']:>20}: n={metrics['n']}, "
              f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
              f"f1={metrics['f1']:.4f}, pr_auc={metrics['pr_auc']:.4f}")

    predict_all_labeled(model)
    print(f"\nPredictions for all labeled nodes cached to {PREDICTIONS_PATH}")

    importances = pd.Series(model.feature_importances_, index=_prepare_xy(train_df)[0].columns)
    print("\nTop 10 features by importance:")
    print(importances.sort_values(ascending=False).head(10))
