"""
GET /node/{id} -- single-node lookup: raw + structural features, both
models' predictions (if labeled), and direct in/out-neighbors.

Neighbors come straight from the loaded networkx graph (state.graph),
respecting direction the same way every other part of this project does:
predecessors = nodes that paid this one, successors = nodes this one paid.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.schemas import NodeDetail

router = APIRouter()


@router.get("/node/{node_id}", response_model=NodeDetail)
def get_node(node_id: int, request: Request) -> NodeDetail:
    state = request.app.state.graphsentry
    full_table = state.full_table

    if node_id not in full_table.index:
        raise HTTPException(status_code=404, detail=f"node {node_id} not found")

    row = full_table.loc[node_id]
    feature_cols = [c for c in full_table.columns if c not in ("time_step", "label")]
    features = {col: float(row[col]) for col in feature_cols}

    pred_row = state.predictions.loc[node_id] if node_id in state.predictions.index else None

    in_neighbors = list(state.graph.predecessors(node_id)) if node_id in state.graph else []
    out_neighbors = list(state.graph.successors(node_id)) if node_id in state.graph else []

    return NodeDetail(
        id=str(node_id),
        time_step=int(row["time_step"]),
        label=int(row["label"]),
        features=features,
        xgboost_proba=float(pred_row["xgboost_proba"]) if pred_row is not None else None,
        gnn_proba=float(pred_row["gnn_proba"]) if pred_row is not None else None,
        in_neighbors=[str(n) for n in in_neighbors],
        out_neighbors=[str(n) for n in out_neighbors],
    )
