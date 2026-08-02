"""
Tests for models/baseline_xgboost.py. The exclusion logic and recency-weight
formula are pure functions of a DataFrame -- tested here with a small
synthetic table, no cached artifacts required.
"""

from __future__ import annotations

import pandas as pd

from models.baseline_xgboost import (
    EXCLUDED_DRIFTED_FEATURES,
    RECENCY_WEIGHT_MAX,
    RECENCY_WEIGHT_MIN,
    _prepare_xy,
    _recency_weight,
    _sample_weights,
)


def _toy_table() -> pd.DataFrame:
    return pd.DataFrame({
        "time_step": [1, 17, 34],
        "feat_1": [0.1, 0.2, 0.3],
        "feat_100": [1.0, 2.0, 3.0],  # one of the excluded drifted features
        "community": pd.Categorical([0, 1, 2]),
        "label": [0, 1, 0],
    })


def test_prepare_xy_drops_label_and_drifted_features_keeps_community():
    table = _toy_table()
    X, y = _prepare_xy(table)

    assert "label" not in X.columns
    assert "feat_100" not in X.columns  # one of EXCLUDED_DRIFTED_FEATURES
    assert "community" in X.columns  # kept -- see module docstring for why
    assert "feat_1" in X.columns
    assert list(y) == [0, 1, 0]


def test_all_six_drifted_features_are_excluded():
    table = _toy_table()
    for feat in EXCLUDED_DRIFTED_FEATURES:
        table[feat] = 0.0
    X, _ = _prepare_xy(table)
    for feat in EXCLUDED_DRIFTED_FEATURES:
        assert feat not in X.columns


def test_recency_weight_is_anchored_to_the_fixed_1_34_window():
    time_step = pd.Series([1, 34])
    weights = _recency_weight(time_step)

    assert weights[0] == RECENCY_WEIGHT_MIN
    assert weights[1] == RECENCY_WEIGHT_MAX


def test_recency_weight_uses_the_full_window_even_on_a_partial_subset():
    """Phase 1 training only sees steps 1-29, but recency weight must stay
    anchored to the full 1-34 window so phase 1 and phase 2 fits are on the
    same scale -- see module docstring. A row at step 29 should NOT get the
    max weight, since step 34 (not in this subset) is the true anchor."""
    partial_subset = pd.Series([1, 29])  # mimics phase 1's steps 1-29 only
    weights = _recency_weight(partial_subset)

    assert weights[0] == RECENCY_WEIGHT_MIN
    assert weights[1] < RECENCY_WEIGHT_MAX


def test_sample_weights_combine_class_and_recency_weighting():
    table = _toy_table()
    weights = _sample_weights(table)

    # row 0: label=0 (licit, weight 1.0) at time_step=1 (recency=min)
    assert weights[0] == RECENCY_WEIGHT_MIN
    # row 1: label=1 (illicit) -- gets extra weight beyond pure recency
    assert weights[1] > RECENCY_WEIGHT_MIN
