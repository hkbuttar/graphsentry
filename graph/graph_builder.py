"""
Builds the Elliptic transaction graph as a networkx.DiGraph.

--------------------------------------------------------------------------------
The 49-time-step snapshot structure (and why it matters)
--------------------------------------------------------------------------------
The Elliptic dataset does not represent one continuously evolving graph. It is
49 discrete snapshots (time steps 1..49), each roughly two weeks of Bitcoin
activity, laid out sequentially. Every transaction (node) belongs to exactly
one time step, via its `time_step` attribute.

Empirically verified against this exact copy of the data (see the `if __name__`
block below): every one of the 234,355 edges connects two nodes with the SAME
time_step. There are zero edges that cross time steps. So the full graph is
really a disjoint union, edge-wise, of 49 separate transaction subgraphs -- a
transaction can only appear connected to other transactions from its own
snapshot window.

Why this drives the train/test split (Step 4):
  - A RANDOM split (e.g. sklearn train_test_split shuffling all labeled nodes)
    would put nodes from time step 30 in training and other nodes from time
    step 30 in test. Since illicit activity in Elliptic is known to cluster in
    bursts tied to specific real-world events (e.g. dark web market shutdowns),
    a random split leaks that event's signature into both train and test,
    inflating scores in a way that would not hold up on genuinely future data.
  - A TEMPORAL split -- train on early time steps, test on later ones -- is
    the only split that mimics how this model would actually be deployed: you
    only ever have past transactions to train on and must generalize to
    transactions you haven't seen yet. This is the walk-forward discipline
    used elsewhere in this project (alpha-signal-lab) applied to graph data.
  - It's also why the GNN (Step 6) has to be inductive (GraphSAGE, not a
    transductive method like plain GCN trained on a fixed node set): test-time
    nodes from later time steps are never seen during training at all.

--------------------------------------------------------------------------------
Node attributes
--------------------------------------------------------------------------------
Per the original Elliptic paper, there are 166 features total per node, where
the first feature IS the time step. This module keeps that first feature as
its own explicit attribute (`time_step`) for readability/filtering, plus 165
attributes `feat_1`..`feat_165` for the rest -- 166 numbers per node either way,
just with the time step pulled out and named. It also carries the `label`
attribute (1 = illicit, 0 = licit, -1 = unknown; see data/loader.py).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

from data.loader import EllipticDataset, load_elliptic

CACHE_DIR = Path(__file__).parent / "cache"
GRAPH_CACHE_PATH = CACHE_DIR / "graph.pkl"
EDGES_CACHE_PATH = CACHE_DIR / "edges.parquet"


def build_graph(ds: EllipticDataset, use_cache: bool = True) -> nx.DiGraph:
    """Build a directed graph: one node per transaction, one edge per payment flow.

    Direction matters here (hence DiGraph, not Graph): an edge txId1 -> txId2
    means an output of txId1 was spent as an input to txId2, so txId2's
    existence causally depends on txId1's. Collapsing this to undirected would
    throw away that "which transaction came first" information, which is
    exactly the kind of structural signal (fan-in/fan-out, in-degree vs.
    out-degree) that's useful for spotting laundering patterns like layering.
    """
    if use_cache and GRAPH_CACHE_PATH.exists():
        with open(GRAPH_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    graph = nx.DiGraph()

    # Add every node up front (not just ones that appear in an edge) so that
    # isolated transactions -- which do exist in this dataset -- are still
    # present with their features and label, e.g. for the feature table in
    # Step 4 even if they contribute nothing to centrality/community metrics.
    graph.add_nodes_from(ds.nodes.index)

    # Vectorized edge insertion: itertuples is far faster here than iterrows
    # for 234k rows since it avoids constructing a Series per row.
    graph.add_edges_from(ds.edges.itertuples(index=False, name=None))

    # One dict-of-dicts lookup covers time_step, feat_1..feat_165, and label
    # in a single set_node_attributes call instead of 166 separate passes.
    node_attrs = ds.nodes.to_dict(orient="index")
    nx.set_node_attributes(graph, node_attrs)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_CACHE_PATH, "wb") as f:
            pickle.dump(graph, f)

    return graph


def save_lightweight_edges(ds: EllipticDataset, use_cache: bool = True) -> pd.DataFrame:
    """Caches just the edge list (txId1, txId2) as a small parquet file --
    no per-node attributes, no networkx object. Exists for backend/state.py:
    the backend only ever needs graph TOPOLOGY (subgraph extraction,
    predecessors/successors), never node attributes (those come from the
    feature table instead). Measured directly: loading the full attributed
    graph.pkl costs ~2.9GB of RAM for the backend's purposes, versus ~257MB
    for an attribute-free graph built from this file plus the feature table
    it already loads for other endpoints -- see backend/state.py.
    """
    if use_cache and EDGES_CACHE_PATH.exists():
        return pd.read_parquet(EDGES_CACHE_PATH)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        ds.edges.to_parquet(EDGES_CACHE_PATH)
    return ds.edges


def time_step_summary(ds: EllipticDataset) -> pd.DataFrame:
    """Per-time-step node counts by class, to make the 49-snapshot structure concrete."""
    counts = ds.nodes.groupby("time_step")["label"].value_counts().unstack(fill_value=0)
    counts = counts.rename(columns={1: "illicit", 0: "licit", -1: "unknown"})
    for col in ("illicit", "licit", "unknown"):
        if col not in counts.columns:
            counts[col] = 0
    counts["total"] = counts[["illicit", "licit", "unknown"]].sum(axis=1)
    return counts[["illicit", "licit", "unknown", "total"]]


def verify_edges_stay_within_time_step(ds: EllipticDataset) -> pd.Series:
    """Recompute the time-step delta for every edge; used to justify the claim above."""
    ts = ds.nodes["time_step"]
    merged = ds.edges.join(ts.rename("ts1"), on="txId1").join(ts.rename("ts2"), on="txId2")
    delta = (merged["ts2"] - merged["ts1"]).value_counts().sort_index()
    return delta


if __name__ == "__main__":
    dataset = load_elliptic()

    delta_counts = verify_edges_stay_within_time_step(dataset)
    print("Edge time_step2 - time_step1 distribution:")
    print(delta_counts)
    assert list(delta_counts.index) == [0], (
        "Expected every edge to stay within a single time step; found cross-step edges."
    )

    g = build_graph(dataset)
    print(f"\nGraph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, directed={g.is_directed()}")

    save_lightweight_edges(dataset)
    print(f"Lightweight edge cache written to {EDGES_CACHE_PATH}")

    sample_node = next(iter(g.nodes))
    print(f"\nSample node {sample_node} attributes (first 5 shown):")
    attrs = g.nodes[sample_node]
    for k in list(attrs)[:5]:
        print(f"  {k}: {attrs[k]}")
    print(f"  ... ({len(attrs)} attributes total)")

    print("\nPer-time-step summary (first 5 and last 5 of 49):")
    summary = time_step_summary(dataset)
    print(pd.concat([summary.head(5), summary.tail(5)]))
