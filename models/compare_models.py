"""
Final comparison: XGBoost baseline vs. GraphSAGE GNN, on the exact same
steps 35-49 test set, using the exact same metrics (models/metrics.py).

Deliberately a thin script, not a re-run of either model: both
models/baseline_xgboost.py and models/gnn_graphsage.py already cache their
per-node probabilities and selected decision threshold. This module loads
those cached outputs and computes the comparison from them -- one source of
truth for each model's predictions, not two independent re-derivations that
could quietly drift apart from what was actually reported per-model.

--------------------------------------------------------------------------------
The honest conclusion
--------------------------------------------------------------------------------
The GNN does not beat the baseline. Not on precision, not on recall, not on
F1, not on PR-AUC. This was checked with real effort behind it, not assumed:
the GNN's weak first result was diagnosed (early stopping wasn't the cause),
validation-swept (four configurations, small gains), and cross-examined
against two competing hypotheses for why it underperforms (a neighborhood-
shape-drift theory that didn't hold up, and a betweenness-coverage theory
that did, in the opposite direction expected). None of it closed the gap.

That outcome is stated plainly here rather than tuned toward a different
answer, because it's also a common, defensible finding in the graph-fraud
literature: once strong structural features (pagerank, community, degree,
clustering) are already available to a tree-based model directly, a GNN's
main theoretical edge -- learning structure implicitly -- has less room to
add value, and here it introduces a new failure mode instead: heavier
reliance on graph structure that itself doesn't transfer across time as
cleanly as node-level features do. See both models' own docstrings
(models/baseline_xgboost.py, models/gnn_graphsage.py) for the full
investigation behind each number here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from features.build_features import DEFAULT_TRAIN_MAX_STEP
from models.metrics import compute_metrics

CACHE_DIR = Path(__file__).parent / "cache"


def load_predictions() -> pd.DataFrame:
    """Joins both models' cached predictions on transaction ID. Both were
    computed from the same labeled node set (see features/build_features.py),
    so this join should not drop any rows -- checked with an assertion below
    rather than silently tolerated."""
    xgb = pd.read_parquet(CACHE_DIR / "xgboost_predictions.parquet")
    gnn = pd.read_parquet(CACHE_DIR / "gnn_predictions.parquet")

    joined = xgb.join(gnn[["gnn_proba"]], how="inner")
    assert len(joined) == len(xgb) == len(gnn), (
        "expected both models' predictions to cover the exact same labeled nodes"
    )
    return joined


def load_thresholds() -> tuple[float, float]:
    xgb_threshold = float((CACHE_DIR / "xgboost_threshold.txt").read_text().strip())
    gnn_threshold = float((CACHE_DIR / "gnn_threshold.txt").read_text().strip())
    return xgb_threshold, gnn_threshold


def compare(split_name: str, df: pd.DataFrame, xgb_threshold: float, gnn_threshold: float) -> pd.DataFrame:
    xgb_metrics = compute_metrics(df["label"].to_numpy(), df["xgboost_proba"].to_numpy(), "XGBoost", threshold=xgb_threshold)
    gnn_metrics = compute_metrics(df["label"].to_numpy(), df["gnn_proba"].to_numpy(), "GraphSAGE", threshold=gnn_threshold)

    table = pd.DataFrame([xgb_metrics, gnn_metrics]).set_index("split")[["n", "precision", "recall", "f1", "pr_auc"]]
    table.index.name = f"model ({split_name})"
    return table


if __name__ == "__main__":
    predictions = load_predictions()
    xgb_threshold, gnn_threshold = load_thresholds()

    train = predictions[predictions["time_step"] <= DEFAULT_TRAIN_MAX_STEP]
    test = predictions[predictions["time_step"] > DEFAULT_TRAIN_MAX_STEP]

    print("--- Train (steps 1-34) ---")
    print(compare("train", train, xgb_threshold, gnn_threshold).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n--- Test (steps 35-49) -- the number that matters ---")
    test_table = compare("test", test, xgb_threshold, gnn_threshold)
    print(test_table.to_string(float_format=lambda x: f"{x:.4f}"))

    xgb_f1, gnn_f1 = test_table.loc["XGBoost", "f1"], test_table.loc["GraphSAGE", "f1"]
    xgb_pr, gnn_pr = test_table.loc["XGBoost", "pr_auc"], test_table.loc["GraphSAGE", "pr_auc"]
    print(f"\nGNN vs. baseline on test: F1 {gnn_f1:.4f} vs {xgb_f1:.4f} "
          f"({'beats' if gnn_f1 > xgb_f1 else 'does not beat'} baseline), "
          f"PR-AUC {gnn_pr:.4f} vs {xgb_pr:.4f} "
          f"({'beats' if gnn_pr > xgb_pr else 'does not beat'} baseline).")
