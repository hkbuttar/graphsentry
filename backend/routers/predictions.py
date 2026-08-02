"""
GET /predictions -- per-node model outputs (baseline and GNN), for the
dashboard's node table and the class-probability slider.

Reads directly from `state.predictions`, which is the exact same DataFrame
models/compare_models.py uses for the Step 7 comparison -- one source of
truth, not a re-derivation. Optional query params let the frontend narrow
the payload server-side instead of shipping all 46,564 labeled nodes on
every request.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from backend.schemas import PredictionRow, PredictionsResponse

router = APIRouter()


@router.get("/predictions", response_model=PredictionsResponse)
def get_predictions(
    request: Request,
    model: str = Query("xgboost", pattern="^(xgboost|gnn)$"),
    min_proba: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(5000, ge=1, le=46564),
) -> PredictionsResponse:
    state = request.app.state.graphsentry
    proba_col = f"{model}_proba"

    df = state.predictions[state.predictions[proba_col] >= min_proba]
    df = df.sort_values(proba_col, ascending=False).head(limit)

    rows = [
        PredictionRow(
            id=str(idx),
            time_step=int(row["time_step"]),
            label=int(row["label"]),
            xgboost_proba=float(row["xgboost_proba"]),
            gnn_proba=float(row["gnn_proba"]),
        )
        for idx, row in df.iterrows()
    ]
    return PredictionsResponse(n=len(rows), predictions=rows)
