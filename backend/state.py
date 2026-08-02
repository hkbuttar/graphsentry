"""
Loads every cached artifact this backend serves, exactly once, at process
startup -- not per-request. This is the one place the backend touches the
graph/feature/model modules; every router reads from the objects built here.

--------------------------------------------------------------------------------
Why load-once-at-startup, not lazy/per-request
--------------------------------------------------------------------------------
The full graph (graph/cache/graph.pkl) is a 381MB pickle that takes ~1.2s to
deserialize -- fine once, unacceptable per HTTP request. Every other artifact
here (feature table, both models' predictions) is a cached parquet file, not
a re-run of graph analytics or model training -- this backend is a thin
reader of what Steps 3-7 already computed, matching this project's stated
principle of not duplicating logic between the modeling code and anything
that serves it.

If any of these files are missing, that means the corresponding pipeline
step (graph.analytics, features.build_features, models.baseline_xgboost,
models.gnn_graphsage) hasn't been run yet -- this module fails loudly with a
clear message pointing at which command to run, rather than serving empty or
fabricated data.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from features.build_features import DEFAULT_TRAIN_MAX_STEP, build_feature_table
from models.compare_models import load_predictions, load_thresholds

GRAPH_CACHE_PATH = Path(__file__).parent.parent / "graph" / "cache" / "graph.pkl"


@dataclass
class AppState:
    graph: nx.DiGraph
    full_table: pd.DataFrame  # every node, 173 columns, from features.build_features
    predictions: pd.DataFrame  # labeled nodes only: time_step, label, xgboost_proba, gnn_proba
    xgb_threshold: float
    gnn_threshold: float


def _require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `{hint}` first (see README Setup)."
        )


def load_state() -> AppState:
    _require(GRAPH_CACHE_PATH, "python -m graph.graph_builder")

    with open(GRAPH_CACHE_PATH, "rb") as f:
        graph = pickle.load(f)

    full_table = build_feature_table()  # reads features/cache/full_features.parquet
    predictions = load_predictions()  # reads both models' cached prediction parquets
    xgb_threshold, gnn_threshold = load_thresholds()

    return AppState(
        graph=graph,
        full_table=full_table,
        predictions=predictions,
        xgb_threshold=xgb_threshold,
        gnn_threshold=gnn_threshold,
    )


__all__ = ["AppState", "load_state", "DEFAULT_TRAIN_MAX_STEP"]
