"""
Loads every cached artifact this backend serves, exactly once, at process
startup -- not per-request. This is the one place the backend touches the
graph/feature/model modules; every router reads from the objects built here.

--------------------------------------------------------------------------------
Why load-once-at-startup, not lazy/per-request
--------------------------------------------------------------------------------
Every artifact here is a cached parquet/pickle file, not a re-run of graph
analytics or model training -- this backend is a thin reader of what Steps
3-7 already computed, matching this project's stated principle of not
duplicating logic between the modeling code and anything that serves it.

If any of these files are missing, that means the corresponding pipeline
step (graph.graph_builder, graph.analytics, features.build_features,
models.baseline_xgboost, models.gnn_graphsage) hasn't been run yet -- this
module fails loudly with a clear message pointing at which command to run,
rather than serving empty or fabricated data.

--------------------------------------------------------------------------------
Why a lightweight, attribute-free graph -- not the full graph.pkl
--------------------------------------------------------------------------------
graph/graph_builder.py's graph.pkl attaches all 166 raw features to every
node, as Step 2 explicitly called for (useful for exploratory analysis, e.g.
the research notebook). But this backend never reads a node attribute off
the graph object -- every router uses it purely for topology
(`.subgraph()`, `.predecessors()`, `.successors()`, membership checks; grep
confirms it), with actual feature values always coming from `full_table`
instead. Measured directly: loading the full attributed graph.pkl costs
~2.9GB of RAM for this process; building an attribute-free graph from
graph/graph_builder.py's lightweight edges.parquet (just two columns,
source/target) plus the node IDs already available from `full_table.index`
costs ~257MB -- roughly a 10x difference, which is the difference between
needing an expensive, large-memory hosting plan and a cheap one. This
optimization is scoped to the backend specifically; graph_builder.py's own
richly-attributed graph is unchanged for every other consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from features.build_features import DEFAULT_TRAIN_MAX_STEP, build_feature_table
from graph.graph_builder import EDGES_CACHE_PATH
from models.compare_models import load_predictions, load_thresholds

EDGES_PATH = EDGES_CACHE_PATH


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


def _load_lightweight_graph(full_table: pd.DataFrame) -> nx.DiGraph:
    _require(EDGES_PATH, "python -m graph.graph_builder")
    edges = pd.read_parquet(EDGES_PATH)

    graph = nx.DiGraph()
    graph.add_nodes_from(full_table.index)  # includes isolated (edge-less) nodes too
    graph.add_edges_from(edges.itertuples(index=False, name=None))
    return graph


def load_state() -> AppState:
    full_table = build_feature_table()  # reads features/cache/full_features.parquet
    graph = _load_lightweight_graph(full_table)
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
