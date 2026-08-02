"""
GET /metrics -- the Step 7 model comparison table (precision/recall/F1/PR-AUC
for both models, train and test), for the dashboard's comparison view.

Reuses models/compare_models.py's compare() function directly rather than
recomputing precision/recall/F1/PR-AUC here -- the same numbers reported in
the README come from this exact function, so the dashboard can't drift from
what's documented as the project's result.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from backend.schemas import MetricsResponse, ModelMetrics
from backend.state import DEFAULT_TRAIN_MAX_STEP
from models.compare_models import compare

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
def get_metrics(request: Request) -> MetricsResponse:
    state = request.app.state.graphsentry
    predictions = state.predictions

    train = predictions[predictions["time_step"] <= DEFAULT_TRAIN_MAX_STEP]
    test = predictions[predictions["time_step"] > DEFAULT_TRAIN_MAX_STEP]

    metrics = []
    for split_name, df in [("train", train), ("test", test)]:
        table = compare(split_name, df, state.xgb_threshold, state.gnn_threshold)
        for model_name, row in table.iterrows():
            metrics.append(ModelMetrics(
                model=model_name,
                split=split_name,
                n=int(row["n"]),
                precision=float(row["precision"]),
                recall=float(row["recall"]),
                f1=float(row["f1"]),
                pr_auc=float(row["pr_auc"]),
            ))

    return MetricsResponse(metrics=metrics)
