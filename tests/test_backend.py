"""
Tests for the FastAPI backend, using TestClient (which runs the real
lifespan -- state.py's load_state() actually executes, reading the same
cached artifacts the manual curl testing during Step 8 used). Skips
gracefully if those artifacts don't exist yet.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.state import GRAPH_CACHE_PATH
from features.build_features import FULL_TABLE_CACHE_PATH
from models.baseline_xgboost import PREDICTIONS_PATH as XGB_PREDICTIONS_PATH
from models.gnn_graphsage import PREDICTIONS_PATH as GNN_PREDICTIONS_PATH
from tests.conftest import skip_if_missing


@pytest.fixture(scope="module")
def client():
    skip_if_missing(GRAPH_CACHE_PATH, "python -m graph.graph_builder")
    skip_if_missing(FULL_TABLE_CACHE_PATH, "python -m features.build_features")
    skip_if_missing(XGB_PREDICTIONS_PATH, "python -m models.baseline_xgboost")
    skip_if_missing(GNN_PREDICTIONS_PATH, "python -m models.gnn_graphsage")

    from backend.main import app
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_network_default_time_step(client):
    response = client.get("/network")
    assert response.status_code == 200
    body = response.json()

    assert body["time_step"] == 32
    assert len(body["nodes"]) > 0
    assert isinstance(body["truncated"], bool)
    # every edge endpoint must be one of the returned nodes -- the induced
    # subgraph guarantee the network router's docstring describes
    node_ids = {n["id"] for n in body["nodes"]}
    for edge in body["edges"]:
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids


def test_network_respects_max_nodes(client):
    response = client.get("/network?time_step=32&max_nodes=20")
    body = response.json()
    assert len(body["nodes"]) <= 20
    assert body["truncated"] is True  # time step 32 has 4,525 nodes, well over 20


def test_predictions_limit_and_ordering(client):
    response = client.get("/predictions?model=xgboost&limit=5")
    body = response.json()
    assert body["n"] == 5
    probas = [row["xgboost_proba"] for row in body["predictions"]]
    assert probas == sorted(probas, reverse=True)


def test_metrics_has_both_models_both_splits(client):
    response = client.get("/metrics")
    body = response.json()
    combos = {(m["model"], m["split"]) for m in body["metrics"]}
    assert combos == {("XGBoost", "train"), ("XGBoost", "test"), ("GraphSAGE", "train"), ("GraphSAGE", "test")}


def test_node_not_found_returns_404(client):
    response = client.get("/node/1")
    assert response.status_code == 404


def test_node_found_returns_features_and_neighbors(client):
    # grab a real node id from the network endpoint first
    network = client.get("/network?max_nodes=5").json()
    node_id = network["nodes"][0]["id"]

    response = client.get(f"/node/{node_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == node_id
    assert "feat_1" in body["features"]
    assert isinstance(body["in_neighbors"], list)
    assert isinstance(body["out_neighbors"], list)
