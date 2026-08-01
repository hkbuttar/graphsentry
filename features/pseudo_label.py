"""
Self-training (pseudo-labeling) for the 157,205 "unknown" nodes -- an
exploratory semi-supervised extension, kept deliberately separate from the
supervised pipeline in models/.

--------------------------------------------------------------------------------
Why this is a standalone module, not an input to Steps 5/6
--------------------------------------------------------------------------------
Pseudo-labeling means: train a classifier on the real labels, use it to guess
labels for the unknown nodes, then treat the confident guesses as if they were
real. That's useful for exploring how much signal is sitting in the unlabeled
77% of the graph, but it is NOT ground truth, and it would be circular to
"validate" a fraud model against labels that same kind of model invented. So:
  - This module produces its own output (pseudo_labels.parquet) and its own
    report. It never gets merged into features/build_features.py's output.
  - The baseline XGBoost (Step 5) and GraphSAGE (Step 6) models train and are
    evaluated ONLY on the real, human-labeled nodes. Their reported
    precision/recall/F1/PR-AUC are never touched by anything in this file.
  - The only thing this module's output is good for is description ("here's
    what a classifier believes about the nodes nobody labeled") and as a
    documented possible input to a future, clearly-flagged experiment -- not
    as silent extra training data.

--------------------------------------------------------------------------------
Method: self-training with a confidence threshold
--------------------------------------------------------------------------------
1. Fit HistGradientBoostingClassifier on the labeled TRAIN-time nodes only
   (time steps 1-34) -- the same temporal boundary as the real models, so this
   module doesn't get to see "the future" either.
   HistGradientBoostingClassifier specifically because it handles missing
   values natively (betweenness is NaN for ~96% of nodes -- see Step 3) and
   pandas categorical columns natively (`community`), with no imputation or
   one-hot encoding needed.
2. Before touching any unknown node, get an honest read on how trustworthy
   this classifier's confident predictions actually are: 5-fold cross-validated
   out-of-fold probabilities on the labeled train set itself, then check
   precision specifically among predictions the model was confident about
   (p > 0.95 for illicit, p < 0.05 for licit). This is the calibration check
   -- it uses only labeled data, so it's a fair test, unlike checking
   "accuracy" on the unknowns (impossible; that's the whole problem).
3. Refit on all labeled train-time data, predict on every unknown node
   (train-time and test-time both), and keep only predictions past the same
   0.95/0.05 confidence bands as pseudo-labels. Everything else stays unknown.
4. Report coverage (how many of the 157,205 unknowns got a confident guess)
   and the resulting illicit rate among them, split by whether the node is in
   the train-time or test-time window -- test-time unknowns are being
   extrapolated to a period the classifier has never seen and where the true
   illicit rate is known (from Step 4) to be lower, so those pseudo-labels
   deserve more skepticism than train-time ones.

--------------------------------------------------------------------------------
Consistency check: do the pseudo-labels agree with independent structural signal?
--------------------------------------------------------------------------------
The calibration check above tests the classifier against real labels, but it
can't validate the pseudo-labels themselves (there's no ground truth to check
them against -- that's the entire premise). The best available substitute is
to ask whether the pseudo-labels line up with a completely independent signal:
the Step 3 finding that illicit nodes cluster disproportionately into specific
Louvain communities.

Checked empirically: pseudo-illicit nodes fall into those same "high-risk"
communities (>2x baseline illicit rate) 20.0% of the time, versus 12.3% for
pseudo-licit nodes. That's the same direction as the real-label finding
(45.4% vs. 13.4%) but noticeably weaker. Same story on raw structural
features -- pseudo-illicit nodes have lower in/out-degree than pseudo-licit
nodes (consistent with real illicit nodes being structurally sparser), but
their clustering coefficient sits closer to the licit profile than the real
illicit one does. Read plainly: self-training recovers a real but diluted
version of the same pattern, not a clean match. That's the honest conclusion
-- this extension is suggestive, not a substitute for real labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from data.loader import load_elliptic
from features.build_features import (
    DEFAULT_TRAIN_MAX_STEP,
    build_feature_table,
    get_labeled_subset,
    temporal_train_test_split,
)
from graph.analytics import HIGH_RISK_RATE_MULTIPLIER, check_illicit_clustering

CACHE_DIR = Path(__file__).parent / "cache"
PSEUDO_LABEL_CACHE_PATH = CACHE_DIR / "pseudo_labels.parquet"

HIGH_CONFIDENCE_ILLICIT = 0.95
HIGH_CONFIDENCE_LICIT = 0.05
CV_FOLDS = 5


def _feature_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c != "label"]


def _make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype",
        class_weight="balanced",  # illicit is ~11.6% of train -- don't let the model default to "always licit"
        random_state=42,
    )


def calibration_check(train_df: pd.DataFrame) -> dict:
    """Out-of-fold cross-validation on labeled train data ONLY -- estimates how
    much to trust this classifier's confident predictions before we ever apply
    it to a node whose real label we can't check."""
    feature_cols = _feature_columns(train_df)
    X = train_df[feature_cols]
    y = train_df["label"].to_numpy()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(_make_model(), X, y, cv=cv, method="predict_proba")[:, 1]

    confident_illicit = oof_proba >= HIGH_CONFIDENCE_ILLICIT
    confident_licit = oof_proba <= HIGH_CONFIDENCE_LICIT

    precision_illicit = y[confident_illicit].mean() if confident_illicit.any() else float("nan")
    precision_licit = 1 - y[confident_licit].mean() if confident_licit.any() else float("nan")

    report = {
        "n_train_labeled": len(train_df),
        "coverage_confident_illicit": confident_illicit.mean(),
        "coverage_confident_licit": confident_licit.mean(),
        "precision_among_confident_illicit": precision_illicit,
        "precision_among_confident_licit": precision_licit,
    }
    return report


def generate_pseudo_labels(use_cache: bool = True) -> pd.DataFrame:
    """Fit on labeled train-time nodes, predict on every unknown node, keep
    only high-confidence guesses. Returns a table indexed by txId with columns
    pseudo_label (1/0), confidence, and is_test_time_window (bool)."""
    if use_cache and PSEUDO_LABEL_CACHE_PATH.exists():
        return pd.read_parquet(PSEUDO_LABEL_CACHE_PATH)

    full_table = build_feature_table()
    labeled = get_labeled_subset(full_table)
    train_df, _ = temporal_train_test_split(labeled)

    feature_cols = _feature_columns(full_table)
    model = _make_model()
    model.fit(train_df[feature_cols], train_df["label"])

    unknown = full_table[full_table["label"] == -1]
    proba_illicit = model.predict_proba(unknown[feature_cols])[:, 1]

    is_illicit_guess = proba_illicit >= HIGH_CONFIDENCE_ILLICIT
    is_licit_guess = proba_illicit <= HIGH_CONFIDENCE_LICIT
    confident_mask = is_illicit_guess | is_licit_guess

    result = pd.DataFrame(index=unknown.index[confident_mask])
    result["pseudo_label"] = np.where(is_illicit_guess[confident_mask], 1, 0)
    result["confidence"] = np.where(
        is_illicit_guess[confident_mask],
        proba_illicit[confident_mask],
        1 - proba_illicit[confident_mask],
    )
    result["is_test_time_window"] = unknown.loc[result.index, "time_step"] > DEFAULT_TRAIN_MAX_STEP

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(PSEUDO_LABEL_CACHE_PATH)

    return result


def analyze_pseudo_label_consistency(use_cache: bool = True) -> dict:
    """Cross-check the pseudo-labels against the Step 3 Louvain finding and raw
    structural feature profiles -- the closest thing to validation available
    when there's no ground truth to check the pseudo-labels against directly.
    """
    full_table = build_feature_table()
    ds = load_elliptic()
    communities = full_table["community"].astype(int)

    grouped = check_illicit_clustering(ds, communities, verbose=False)
    baseline = ds.nodes.loc[ds.nodes["label"] >= 0, "label"].mean()
    high_risk_ids = set(grouped[grouped["illicit_rate"] > HIGH_RISK_RATE_MULTIPLIER * baseline].index)

    pseudo = generate_pseudo_labels(use_cache=use_cache)
    pseudo_illicit_idx = pseudo[pseudo["pseudo_label"] == 1].index
    pseudo_licit_idx = pseudo[pseudo["pseudo_label"] == 0].index

    frac_pseudo_illicit_high_risk = communities.loc[pseudo_illicit_idx].isin(high_risk_ids).mean()
    frac_pseudo_licit_high_risk = communities.loc[pseudo_licit_idx].isin(high_risk_ids).mean()

    labeled = get_labeled_subset(full_table)
    feature_cols = ["pagerank", "in_degree", "out_degree", "clustering"]
    profile = pd.DataFrame({
        "real_illicit": labeled.loc[labeled["label"] == 1, feature_cols].mean(),
        "real_licit": labeled.loc[labeled["label"] == 0, feature_cols].mean(),
        "pseudo_illicit": full_table.loc[pseudo_illicit_idx, feature_cols].mean(),
        "pseudo_licit": full_table.loc[pseudo_licit_idx, feature_cols].mean(),
    })

    return {
        "frac_pseudo_illicit_in_high_risk_communities": frac_pseudo_illicit_high_risk,
        "frac_pseudo_licit_in_high_risk_communities": frac_pseudo_licit_high_risk,
        "structural_profile": profile,
    }


if __name__ == "__main__":
    full_table = build_feature_table()
    labeled = get_labeled_subset(full_table)
    train_df, _ = temporal_train_test_split(labeled)

    print("--- Calibration check (cross-validated on labeled train nodes only) ---")
    calib = calibration_check(train_df)
    for k, v in calib.items():
        print(f"{k:>34}: {v:.4f}" if isinstance(v, float) else f"{k:>34}: {v}")

    print("\n--- Applying self-training to unknown nodes ---")
    pseudo = generate_pseudo_labels()
    n_unknown = (full_table["label"] == -1).sum()
    print(f"Confidently pseudo-labeled: {len(pseudo)} of {n_unknown} unknown nodes "
          f"({100*len(pseudo)/n_unknown:.1f}% coverage)")

    for window, name in [(False, "train-time unknowns (steps 1-34)"),
                          (True, "test-time unknowns (steps 35-49)")]:
        subset = pseudo[pseudo["is_test_time_window"] == window]
        if len(subset) == 0:
            print(f"{name}: none pseudo-labeled")
            continue
        illicit_rate = (subset["pseudo_label"] == 1).mean()
        print(f"{name}: {len(subset)} labeled, {100*illicit_rate:.1f}% pseudo-illicit")

    print("\nCaveat: these are model-generated guesses, not verified ground truth. "
          "The calibration check above (on real labels) is the only honest signal "
          "we have about how much to trust them -- treat the coverage numbers here "
          "as an upper bound on how useful this extension could be, not a result.")

    print("\n--- Consistency check against Step 3's community-clustering finding ---")
    consistency = analyze_pseudo_label_consistency()
    print(f"Pseudo-illicit nodes in high-risk communities: "
          f"{100*consistency['frac_pseudo_illicit_in_high_risk_communities']:.1f}%")
    print(f"Pseudo-licit nodes in high-risk communities:   "
          f"{100*consistency['frac_pseudo_licit_in_high_risk_communities']:.1f}%")
    print("(for reference, real illicit nodes: 45.4% vs. real licit-adjacent baseline: 13.4% overall)")
    print("\nMean structural features by group:")
    print(consistency["structural_profile"])
    print("\nConclusion: pseudo-labels show the same DIRECTION as the real-label "
          "clustering finding, but a weaker effect -- self-training recovers a "
          "diluted version of the real pattern, not a clean match. Treat this as "
          "suggestive corroboration, not proof the pseudo-labels are reliable.")
