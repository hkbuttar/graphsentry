"""
Tests for data/loader.py, run against the real downloaded Elliptic dataset.
These pin down the exact documented counts from the README's Data section --
if a re-download or a pandas upgrade ever silently changed how these are
parsed, this is what would catch it.
"""

from __future__ import annotations

import pytest

from data.loader import FEATURES_CSV, load_elliptic, sanity_check
from tests.conftest import skip_if_missing


@pytest.fixture(scope="module")
def dataset():
    skip_if_missing(FEATURES_CSV, "download the dataset -- see README Setup")
    return load_elliptic()


def test_dataset_shape_matches_readme(dataset):
    report = sanity_check(dataset)
    assert report["n_nodes"] == 203_769
    assert report["n_edges"] == 234_355
    assert report["n_time_steps"] == 49
    assert report["time_step_range"] == (1, 49)


def test_class_distribution_matches_readme(dataset):
    report = sanity_check(dataset)
    assert report["n_illicit"] == 4_545
    assert report["n_licit"] == 42_019
    assert report["n_unknown"] == 157_205


def test_features_have_166_columns_including_time_step(dataset):
    # time_step + feat_1..feat_165 = 166, per the Elliptic paper's convention
    # of counting the time step as the first feature
    feature_cols = [c for c in dataset.nodes.columns if c not in ("label",)]
    assert len(feature_cols) == 166
    assert "time_step" in feature_cols


def test_label_encoding(dataset):
    assert set(dataset.nodes["label"].unique()) == {-1, 0, 1}
