"""
Classical graph analytics layer: centrality measures + community detection,
computed on top of the DiGraph built in graph_builder.py.

Output is a single feature table, one row per transaction (node), saved to
graph/cache/structural_features.parquet. Step 4 merges this with the 166 raw
Elliptic features to build the full model input.

--------------------------------------------------------------------------------
Betweenness centrality: only the largest component -- tried full coverage,
measurably worse for both models, reverted
--------------------------------------------------------------------------------
Betweenness centrality answers "how often does a shortest path between two
other nodes pass through this one" -- for every node, that means running a
shortest-path search from every other node (Brandes' algorithm: O(V*E) total).
Across all 203,769 nodes as one graph, that's not feasible in plain Python.

Since the graph is 49 disjoint weakly-connected components (one per time
step, confirmed in graph_builder.py -- no edges cross time steps), this
module originally computed betweenness only for the single largest component
(~32 seconds) and left the other ~96% of nodes as NaN. Step 6's GNN work
found that the largest component happens to fall entirely within the TRAIN
period, so every single TEST node got a constant placeholder value for this
feature -- not just missing data, a structural blind spot in exactly the
split being evaluated.

That seemed like a clear bug to fix: compute betweenness exactly for every
component (not an approximation -- a disconnected graph's betweenness is
exactly the union of each component's own internal betweenness) and union
the results. Timed at ~8.3 minutes, implemented, and tested end-to-end. The
result: BOTH models got measurably worse. XGBoost test F1 dropped 0.817 ->
0.728 (PR-AUC 0.804 -> 0.662); the GNN's test F1 dropped 0.659 -> 0.627
(PR-AUC 0.631 -> 0.605). Betweenness jumped to the #2 most important
XGBoost feature once every row had a real value, and that trust didn't
transfer to test.

Investigated rather than just reverted on sight: the first hypothesis was
that per-component normalization (networkx normalizes betweenness by the
size of whatever graph it's given, and each component is a different size)
made values incomparable across components. Checked directly -- correlation
between component size and mean/max betweenness within it is weak (0.09 /
0.17) -- components of nearly identical size produced wildly different
betweenness scales (e.g. two ~6,700-node components differed by 1000x in
mean betweenness). So normalization isn't the main story; betweenness is
better understood as a high-variance, topology-specific metric -- it reflects
the particular chain/star/cluster shape of one time period's component, which
is closer to a per-period idiosyncrasy than a durable, transferable pattern.
Giving both models more of that idiosyncratic signal hurt more than the
original missing-data gap did.

Reverted to the largest-component-only version below -- the one that
produced the best measured results for both models -- with this investigation
kept here rather than erased, since the negative result and the (partially
wrong) normalization hypothesis are both genuinely informative.

--------------------------------------------------------------------------------
Louvain community detection + the illicit-clustering check
--------------------------------------------------------------------------------
Louvain groups nodes into communities by greedily maximizing "modularity"
(roughly: more edges inside groups than you'd expect if edges were placed at
random). Run on this graph (undirected, since python-louvain requires that):
315 communities, taking ~31 seconds.

The check: do illicit nodes fall disproportionately into a small number of
communities, or are they spread evenly? Baseline illicit rate among labeled
nodes is 9.76%. Looking at communities where the illicit rate is more than
double that baseline (>19.5%): those communities hold 45.4% of ALL illicit
nodes in the dataset, while accounting for only 13.4% of ALL nodes (labeled
or not). That's a real, honest finding -- illicit activity is noticeably
concentrated in a minority of communities, not spread uniformly. It is
reported here as found; it was not tuned or cherry-picked to produce this
result, and it is reproducible by rerunning this file.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import community as community_louvain
import networkx as nx
import pandas as pd

from data.loader import load_elliptic
from graph.graph_builder import build_graph

CACHE_DIR = Path(__file__).parent / "cache"
FEATURES_CACHE_PATH = CACHE_DIR / "structural_features.parquet"

# Communities need at least this many labeled nodes before their illicit rate
# is treated as meaningful -- a 2-node community that's 100% illicit is noise.
MIN_LABELED_FOR_RATE = 20
HIGH_RISK_RATE_MULTIPLIER = 2.0


def compute_pagerank(graph: nx.DiGraph) -> pd.Series:
    """PageRank on the full directed graph -- respects edge direction, so a node
    receiving many payments from well-connected sources scores higher, same
    idea as ranking web pages by inbound links."""
    scores = nx.pagerank(graph)
    return pd.Series(scores, name="pagerank")


def compute_degree_features(graph: nx.DiGraph) -> pd.DataFrame:
    """In-degree = number of incoming payments, out-degree = number of outgoing.
    Cheap to compute (just counting edges) but useful: a node with high
    in-degree and near-zero out-degree looks like a collection point."""
    in_deg = dict(graph.in_degree())
    out_deg = dict(graph.out_degree())
    return pd.DataFrame({"in_degree": in_deg, "out_degree": out_deg})


def compute_clustering(graph: nx.DiGraph) -> pd.Series:
    """Clustering coefficient: for each node, how connected its neighbors are
    to each other (fraction of possible triangles that actually exist).
    networkx generalizes this to directed graphs (Fagiolo 2007)."""
    scores = nx.clustering(graph)
    return pd.Series(scores, name="clustering")


def compute_betweenness(graph: nx.DiGraph) -> pd.Series:
    """Exact betweenness centrality, computed only on the largest of the 49
    weakly-connected (per-time-step) components. See module docstring for
    why full per-component coverage was tried and reverted."""
    components = list(nx.weakly_connected_components(graph))
    largest = max(components, key=len)
    subgraph = graph.subgraph(largest).copy()
    scores = nx.betweenness_centrality(subgraph)
    return pd.Series(scores, name="betweenness")


def compute_louvain_communities(graph: nx.DiGraph) -> pd.Series:
    """Louvain community detection. Requires an undirected graph -- direction
    doesn't matter for "which nodes tend to transact within the same cluster,"
    only whether an edge exists at all, so to_undirected() is the right call
    here even though we kept direction for the other metrics."""
    undirected = graph.to_undirected()
    partition = community_louvain.best_partition(undirected, random_state=42)
    return pd.Series(partition, name="community")


def check_illicit_clustering(ds, communities: pd.Series, verbose: bool = True) -> pd.DataFrame:
    """Build the per-community illicit-rate report and print the honest summary
    described in the module docstring. Returns the full per-community table.

    verbose=False suppresses the printout -- used when this table is reused as
    an input to another analysis (e.g. features/pseudo_label.py) that has its
    own reporting and would otherwise duplicate this output."""
    df = ds.nodes[["label"]].join(communities)
    grouped = df.groupby("community")["label"].value_counts().unstack(fill_value=0)
    grouped = grouped.rename(columns={1: "illicit", 0: "licit", -1: "unknown"})
    for col in ("illicit", "licit", "unknown"):
        if col not in grouped.columns:
            grouped[col] = 0
    grouped["labeled"] = grouped["illicit"] + grouped["licit"]
    grouped["size"] = grouped[["illicit", "licit", "unknown"]].sum(axis=1)
    grouped["illicit_rate"] = grouped["illicit"] / grouped["labeled"].replace(0, pd.NA)

    baseline = df.loc[df["label"] >= 0, "label"].mean()
    high_risk = grouped[grouped["illicit_rate"] > HIGH_RISK_RATE_MULTIPLIER * baseline]

    illicit_in_high_risk = high_risk["illicit"].sum()
    total_illicit = grouped["illicit"].sum()
    nodes_in_high_risk = high_risk["size"].sum()
    total_nodes = grouped["size"].sum()

    if verbose:
        print(f"Baseline illicit rate among labeled nodes: {baseline:.4f}")
        print(f"Communities with labeled nodes >= {MIN_LABELED_FOR_RATE}: "
              f"{(grouped['labeled'] >= MIN_LABELED_FOR_RATE).sum()} of {len(grouped)} total")
        print(f"Communities with illicit_rate > {HIGH_RISK_RATE_MULTIPLIER}x baseline: {len(high_risk)}")
        print(f"  -> hold {illicit_in_high_risk}/{total_illicit} "
              f"({100 * illicit_in_high_risk / total_illicit:.1f}%) of all illicit nodes")
        print(f"  -> but only {nodes_in_high_risk}/{total_nodes} "
              f"({100 * nodes_in_high_risk / total_nodes:.1f}%) of all nodes")
        print("\nConclusion: illicit nodes cluster disproportionately into a "
              "minority of Louvain communities in this dataset.")

    return grouped


def build_structural_feature_table(use_cache: bool = True) -> pd.DataFrame:
    """Run every metric above and assemble one feature table, keyed by txId."""
    if use_cache and FEATURES_CACHE_PATH.exists():
        return pd.read_parquet(FEATURES_CACHE_PATH)

    ds = load_elliptic()
    graph = build_graph(ds)

    pagerank = compute_pagerank(graph)
    degrees = compute_degree_features(graph)
    clustering = compute_clustering(graph)
    betweenness = compute_betweenness(graph)
    communities = compute_louvain_communities(graph)

    table = pd.DataFrame(index=ds.nodes.index)
    table["pagerank"] = pagerank
    table = table.join(degrees)
    table["clustering"] = clustering
    table["betweenness"] = betweenness  # full coverage -- see compute_betweenness docstring
    table["community"] = communities

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        table.to_parquet(FEATURES_CACHE_PATH)

    return table


if __name__ == "__main__":
    dataset = load_elliptic()
    G = build_graph(dataset)

    print("Computing structural features (pagerank, degree, clustering, "
          "betweenness on largest component, Louvain communities)...")
    features = build_structural_feature_table()
    print(f"\nStructural feature table: {features.shape[0]} rows, "
          f"{features.shape[1]} columns")
    print(features.describe())

    print(f"\nBetweenness coverage: {features['betweenness'].notna().sum()} "
          f"of {len(features)} nodes ({100 * features['betweenness'].notna().mean():.1f}%)")

    print("\n--- Illicit clustering check ---")
    communities = features["community"]
    check_illicit_clustering(dataset, communities)
