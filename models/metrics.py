"""
Shared evaluation logic for every supervised model in this project (XGBoost
baseline, GraphSAGE GNN): precision/recall/F1 at a chosen decision threshold,
PR-AUC independent of any threshold, and the out-of-sample threshold-selection
routine used by both. Factored out here so neither model file re-derives the
same four-line computation -- see models/baseline_xgboost.py's docstring for
why accuracy is excluded and why the threshold is selected rather than
assumed to be 0.5.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score

DEFAULT_THRESHOLD = 0.5


def select_threshold(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    """Picks the probability cutoff that maximizes F1 on out-of-sample
    validation predictions, instead of assuming 0.5."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba_val)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = f1s[:-1].argmax()  # last precision/recall point has no threshold
    return float(thresholds[best_idx])


def compute_metrics(y: np.ndarray, proba: np.ndarray, split_name: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Precision/recall/F1 (at the given threshold) + PR-AUC (threshold-
    independent). These four numbers are the only ones this project reports
    as "the result" for any model -- accuracy is never used (see
    models/baseline_xgboost.py's docstring for why)."""
    pred = (proba >= threshold).astype(int)
    return {
        "split": split_name,
        "n": len(y),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, proba),
    }


def print_metrics(metrics: dict) -> None:
    print(f"{metrics['split']:>20}: n={metrics['n']}, "
          f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
          f"f1={metrics['f1']:.4f}, pr_auc={metrics['pr_auc']:.4f}")
