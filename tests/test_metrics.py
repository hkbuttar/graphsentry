"""
Pure unit tests for models/metrics.py -- no cached data required, since these
functions operate on plain arrays. This is the shared evaluation logic every
model in the project reports against, so it's worth pinning down precisely.
"""

from __future__ import annotations

import numpy as np

from models.metrics import compute_metrics, select_threshold


def test_compute_metrics_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = compute_metrics(y, proba, "test", threshold=0.5)

    assert metrics["n"] == 6
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_compute_metrics_all_wrong():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.9, 0.9, 0.1, 0.1])  # every prediction backwards at threshold 0.5
    metrics = compute_metrics(y, proba, "test", threshold=0.5)

    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_compute_metrics_threshold_changes_precision_recall_not_pr_auc():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.2, 0.4, 0.6, 0.7, 0.9])

    loose = compute_metrics(y, proba, "test", threshold=0.3)
    strict = compute_metrics(y, proba, "test", threshold=0.65)

    # PR-AUC only depends on the ranking of proba, not the threshold
    assert loose["pr_auc"] == strict["pr_auc"]
    # but precision/recall change because more/fewer predictions cross the line
    assert loose["recall"] >= strict["recall"]


def test_select_threshold_picks_the_f1_maximizing_cutoff():
    # a case with an obvious best cutoff: values cluster tightly around 0.2 (licit)
    # and 0.8 (illicit), with one ambiguous point at 0.5 that could go either way
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold = select_threshold(y, proba)

    # any cutoff strictly between 0.3 and 0.7 achieves perfect separation --
    # confirm the selected threshold actually achieves that, rather than
    # pinning down one exact float
    metrics = compute_metrics(y, proba, "test", threshold=threshold)
    assert metrics["f1"] == 1.0
