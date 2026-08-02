"""
Tests for features/build_features.py, run against the cached merged feature
table. Pins down the temporal split boundary and the unknown-label handling
that Steps 5-7 all build on.
"""

from __future__ import annotations

import pytest

from features.build_features import (
    DEFAULT_TRAIN_MAX_STEP,
    FULL_TABLE_CACHE_PATH,
    build_feature_table,
    get_labeled_subset,
    temporal_train_test_split,
)
from tests.conftest import skip_if_missing


@pytest.fixture(scope="module")
def full_table():
    skip_if_missing(FULL_TABLE_CACHE_PATH, "python -m features.build_features")
    return build_feature_table()


def test_full_table_shape(full_table):
    # 166 raw + 5 structural (pagerank, in_degree, out_degree, clustering,
    # betweenness) + community + label = 173
    assert full_table.shape == (203_769, 173)


def test_get_labeled_subset_drops_only_unknowns(full_table):
    labeled = get_labeled_subset(full_table)
    assert (labeled["label"] >= 0).all()
    assert len(labeled) == 46_564  # 4,545 illicit + 42,019 licit


def test_temporal_split_boundary_is_exclusive_and_exhaustive(full_table):
    labeled = get_labeled_subset(full_table)
    train, test = temporal_train_test_split(labeled)

    assert train["time_step"].max() == DEFAULT_TRAIN_MAX_STEP
    assert test["time_step"].min() == DEFAULT_TRAIN_MAX_STEP + 1
    assert len(train) + len(test) == len(labeled)


def test_temporal_split_reproduces_documented_class_balance_shift(full_table):
    """The illicit rate genuinely differs between train and test (11.6% vs.
    6.5%, per the README) -- this is the empirical finding that justifies a
    temporal split over a random one. Pinned here so it's never silently
    "fixed" by a future change without that being a deliberate decision."""
    labeled = get_labeled_subset(full_table)
    train, test = temporal_train_test_split(labeled)

    train_illicit_rate = (train["label"] == 1).mean()
    test_illicit_rate = (test["label"] == 1).mean()

    assert train_illicit_rate == pytest.approx(0.1158, abs=0.001)
    assert test_illicit_rate == pytest.approx(0.0650, abs=0.001)
