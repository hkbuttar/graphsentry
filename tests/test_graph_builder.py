"""
Tests for graph/graph_builder.py, run against the real cached graph.

The "every edge stays within its own time step" invariant is the single
most load-bearing fact in this project -- it's the justification for the
temporal train/test split, the reason GraphSAGE has to be inductive, and the
reason Louvain communities never span time steps. If a future change to the
raw data or the graph-building logic ever broke that invariant silently,
everything built on top of it (Steps 4-7) would be reasoning from a false
premise. This test exists to catch that immediately, not eventually.
"""

from __future__ import annotations

import networkx as nx
import pytest

from data.loader import FEATURES_CSV, load_elliptic
from graph.graph_builder import GRAPH_CACHE_PATH, build_graph
from tests.conftest import skip_if_missing


@pytest.fixture(scope="module")
def graph():
    skip_if_missing(FEATURES_CSV, "download the dataset -- see README Setup")
    skip_if_missing(GRAPH_CACHE_PATH, "python -m graph.graph_builder")
    dataset = load_elliptic()
    return build_graph(dataset)


def test_graph_shape_matches_documented_dataset_size(graph):
    assert graph.number_of_nodes() == 203_769
    assert graph.number_of_edges() == 234_355
    assert graph.is_directed()


def test_every_node_carries_expected_attributes(graph):
    sample_node = next(iter(graph.nodes))
    attrs = graph.nodes[sample_node]
    assert "time_step" in attrs
    assert "label" in attrs
    assert attrs["label"] in (-1, 0, 1)


def test_no_edge_crosses_a_time_step_boundary(graph):
    """The core structural finding this whole project's temporal split relies
    on. Checked directly on every single edge, not sampled."""
    mismatches = [
        (u, v)
        for u, v in graph.edges()
        if graph.nodes[u]["time_step"] != graph.nodes[v]["time_step"]
    ]
    assert mismatches == [], f"found {len(mismatches)} edges crossing a time-step boundary"


def test_graph_has_exactly_49_weakly_connected_components(graph):
    """Confirms the "49 disjoint components, one per time step" structure
    that Steps 3, 6, and 7 all depend on -- not just that edges don't cross
    steps, but that each step forms exactly one connected piece."""
    components = list(nx.weakly_connected_components(graph))
    assert len(components) == 49
