"""
Threshold sensitivity sweep for the pseudo-labeling self-training classifier
(features/pseudo_label.py). Kept in a separate file since it's a follow-up
analysis on top of that module, not part of the core self-training pipeline.

--------------------------------------------------------------------------------
Why this exists
--------------------------------------------------------------------------------
pseudo_label.py's calibration_check() reports precision at exactly one
confidence cutoff (>=0.95 for illicit, <=0.05 for licit). That single number
hides a real tradeoff: a stricter threshold gives higher-precision pseudo-labels
but covers fewer of the unknown nodes; a looser threshold covers more nodes but
trusts the classifier further out on predictions it's less sure about.

This module reuses the exact same 5-fold out-of-fold predictions pseudo_label.py
computes (cross_val_predict on labeled train-time nodes only -- no leakage, no
predictions from a model that saw the node it's scoring) and sweeps several
thresholds over them, so the 0.95/0.05 default used elsewhere can be justified
by an actual precision/coverage curve instead of being an arbitrary round
number picked in advance.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from features.build_features import (
    build_feature_table,
    get_labeled_subset,
    temporal_train_test_split,
)
from features.pseudo_label import CV_FOLDS, _feature_columns, _make_model


def out_of_fold_probabilities(train_df: pd.DataFrame) -> tuple:
    """Re-derives the same out-of-fold P(illicit) array pseudo_label.py's
    calibration_check() uses -- kept as its own call here (rather than
    importing a shared private helper) so this file has no hidden coupling to
    pseudo_label.py's internals beyond the model definition and feature list."""
    feature_cols = _feature_columns(train_df)
    X = train_df[feature_cols]
    y = train_df["label"].to_numpy()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(_make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    return oof_proba, y


def threshold_sensitivity_sweep(
    train_df: pd.DataFrame, thresholds: tuple = (0.90, 0.95, 0.99)
) -> pd.DataFrame:
    """For each threshold t: what fraction of train nodes get a confident
    illicit/licit call (coverage), and of those, how often is the call
    actually correct (precision) -- checked against real labels."""
    oof_proba, y = out_of_fold_probabilities(train_df)

    rows = []
    for t in thresholds:
        confident_illicit = oof_proba >= t
        confident_licit = oof_proba <= (1 - t)
        rows.append({
            "threshold": t,
            "coverage_illicit": confident_illicit.mean(),
            "precision_illicit": y[confident_illicit].mean() if confident_illicit.any() else float("nan"),
            "coverage_licit": confident_licit.mean(),
            "precision_licit": 1 - y[confident_licit].mean() if confident_licit.any() else float("nan"),
        })
    return pd.DataFrame(rows).set_index("threshold")


if __name__ == "__main__":
    full_table = build_feature_table()
    labeled = get_labeled_subset(full_table)
    train_df, _ = temporal_train_test_split(labeled)

    print("Threshold sensitivity sweep (5-fold out-of-fold predictions, labeled train nodes only)")
    sweep = threshold_sensitivity_sweep(train_df)
    print(sweep.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nHow to read this: 'coverage' is the fraction of train nodes confident "
          "enough to get a pseudo-label at that threshold; 'precision' is how often "
          "that confident call was actually right, checked against real labels. "
          "Raising the threshold should trade coverage down for precision up -- "
          "use this table to judge whether pseudo_label.py's 0.95/0.05 default is "
          "a reasonable point on that curve or worth moving.")
