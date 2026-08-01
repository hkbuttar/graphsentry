"""
Feature-level distribution-shift audit: does any single feature look
different between the train period (steps 1-34) and the test period
(steps 35-49), independent of any label?

--------------------------------------------------------------------------------
Why this is a different question than the four prior baseline experiments
--------------------------------------------------------------------------------
Two rounds of regularization and one round of recency-weighting + threshold
tuning (see models/baseline_xgboost.py) all adjusted the MODEL, and none
closed the train/test generalization gap by much -- consistent evidence that
the gap is real distribution shift across time, not something model tuning
can fix. This asks a different question: do the INPUTS themselves look
different across time? If a handful of features have severely different
distributions in the test period, a model relying on them would generalize
poorly no matter how it's tuned -- a targeted, explainable finding, rather
than another blunt hyperparameter search.

This is checkable without touching labels at all: comparing a feature's
marginal distribution across time periods is the same kind of covariate-shift
check a production monitoring system would run on live, unlabeled data. No
label information crosses the train/test boundary here, so this isn't test
leakage in the sense that matters for the model's reported metrics.

--------------------------------------------------------------------------------
Method
--------------------------------------------------------------------------------
Two-sample Kolmogorov-Smirnov test per numeric feature (`community` and
`time_step` excluded -- the former is categorical/arbitrary, the latter is
trivially different by construction since it defines the split), comparing
ALL nodes (labeled and unknown -- labels are irrelevant to this check) in the
train window against all nodes in the test window. The KS statistic is a
distance between 0 (identical distributions) and 1 (completely disjoint);
ranked descending to surface the most-drifted features first. With ~200k
nodes, even trivially small differences often reach statistical significance,
so the KS statistic's magnitude matters more than whether p < 0.05.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp

from features.build_features import DEFAULT_TRAIN_MAX_STEP, build_feature_table


def audit_feature_drift(full_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Returns a table indexed by feature name, sorted by KS statistic
    descending (most-drifted first), with columns ks_statistic, p_value,
    n_train, n_test."""
    if full_table is None:
        full_table = build_feature_table()

    train = full_table[full_table["time_step"] <= DEFAULT_TRAIN_MAX_STEP]
    test = full_table[full_table["time_step"] > DEFAULT_TRAIN_MAX_STEP]

    numeric_cols = [
        c for c in full_table.columns
        if c not in ("label", "community", "time_step")
        and pd.api.types.is_numeric_dtype(full_table[c])
    ]

    rows = []
    for col in numeric_cols:
        train_vals = train[col].dropna()
        test_vals = test[col].dropna()
        if len(train_vals) < 2 or len(test_vals) < 2:
            continue
        stat, pvalue = ks_2samp(train_vals, test_vals)
        rows.append({
            "feature": col,
            "ks_statistic": stat,
            "p_value": pvalue,
            "n_train": len(train_vals),
            "n_test": len(test_vals),
        })

    return pd.DataFrame(rows).set_index("feature").sort_values("ks_statistic", ascending=False)


if __name__ == "__main__":
    drift = audit_feature_drift()
    print(f"Feature drift audit: {len(drift)} numeric features compared between "
          f"train (steps 1-{DEFAULT_TRAIN_MAX_STEP}) and test (steps {DEFAULT_TRAIN_MAX_STEP + 1}-49), "
          f"using ALL nodes (labeled and unknown -- no label information used).")

    print("\nTop 15 most-drifted features (by KS statistic):")
    print(drift.head(15).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\nBottom 5 least-drifted features:")
    print(drift.tail(5).to_string(float_format=lambda x: f"{x:.4f}"))

    n_significant = (drift["p_value"] < 0.01).sum()
    print(f"\n{n_significant} of {len(drift)} features show statistically significant "
          f"drift (p < 0.01) -- with this much data, that threshold alone isn't very "
          f"informative (see module docstring); the KS statistic ranking above is what "
          f"actually distinguishes meaningfully-drifted features from noise.")
