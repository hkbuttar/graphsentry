"""
GET /network -- one time step's transaction subgraph as nodes/edges JSON,
for the frontend's force-directed graph view.

--------------------------------------------------------------------------------
Why one time step at a time, not a global sample
--------------------------------------------------------------------------------
The full graph is 203,769 nodes -- unrenderable in a browser, and a random
global sample would mostly return edge-less junk pulled from unrelated time
periods, since (established back in graph_builder.py) no edge ever crosses a
time step boundary. Every time step's component is already a coherent,
self-contained subgraph, which is what actually renders as a meaningful
picture. `time_step` is a query param (default 32 -- 342 illicit nodes out
of 4,525, a visually meaningful default with real illicit clusters visible,
not just an arbitrary first-available one).

--------------------------------------------------------------------------------
Subsampling within a time step
--------------------------------------------------------------------------------
Even a single time step's component (up to ~7,880 nodes) is too large to
render smoothly. If the component exceeds `max_nodes`, ALL illicit-labeled
nodes in it are kept (they're rare and the whole point of the dashboard),
and the remaining budget is filled with a random sample of the rest -- rather
than a uniform random sample of everyone, which would likely wash out the
illicit nodes entirely given they're a small minority even within a single
time step. The induced subgraph on just the sampled node set is returned, so
edges only appear between two nodes that are both present in the response.
"""

from __future__ import annotations

import random

from fastapi import APIRouter, Request

from backend.schemas import NetworkEdge, NetworkNode, NetworkResponse

router = APIRouter()

DEFAULT_TIME_STEP = 32
DEFAULT_MAX_NODES = 300
RANDOM_SEED = 42


@router.get("/network", response_model=NetworkResponse)
def get_network(request: Request, time_step: int = DEFAULT_TIME_STEP, max_nodes: int = DEFAULT_MAX_NODES) -> NetworkResponse:
    state = request.app.state.graphsentry
    full_table = state.full_table
    predictions = state.predictions

    component_ids = full_table.index[full_table["time_step"] == time_step]
    truncated = len(component_ids) > max_nodes

    if truncated:
        illicit_ids = list(full_table.loc[component_ids].index[full_table.loc[component_ids, "label"] == 1])
        remaining_budget = max(max_nodes - len(illicit_ids), 0)
        other_ids = [i for i in component_ids if i not in set(illicit_ids)]
        rng = random.Random(RANDOM_SEED)
        sampled_other = rng.sample(other_ids, min(remaining_budget, len(other_ids)))
        node_ids = set(illicit_ids) | set(sampled_other)
    else:
        node_ids = set(component_ids)

    subgraph = state.graph.subgraph(node_ids)

    nodes = []
    for node_id in node_ids:
        row = full_table.loc[node_id]
        pred_row = predictions.loc[node_id] if node_id in predictions.index else None
        nodes.append(NetworkNode(
            id=str(node_id),
            time_step=int(row["time_step"]),
            label=int(row["label"]),
            xgboost_proba=float(pred_row["xgboost_proba"]) if pred_row is not None else None,
            gnn_proba=float(pred_row["gnn_proba"]) if pred_row is not None else None,
            pagerank=float(row["pagerank"]),
            community=int(row["community"]),
            in_degree=int(row["in_degree"]),
            out_degree=int(row["out_degree"]),
        ))

    edges = [NetworkEdge(source=str(u), target=str(v)) for u, v in subgraph.edges()]

    return NetworkResponse(time_step=time_step, nodes=nodes, edges=edges, truncated=truncated)
