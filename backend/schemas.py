"""Pydantic response models for every endpoint -- kept separate from the
route handlers so the response shape is documented in one place and FastAPI
can generate accurate OpenAPI docs (visible at /docs) from it."""

from __future__ import annotations

from pydantic import BaseModel


class NetworkNode(BaseModel):
    id: str
    time_step: int
    label: int  # 1 = illicit, 0 = licit, -1 = unknown
    xgboost_proba: float | None = None
    gnn_proba: float | None = None
    pagerank: float
    community: int
    in_degree: int
    out_degree: int


class NetworkEdge(BaseModel):
    source: str
    target: str


class NetworkResponse(BaseModel):
    time_step: int
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]
    truncated: bool  # True if this time step's component was larger than the render cap


class PredictionRow(BaseModel):
    id: str
    time_step: int
    label: int
    xgboost_proba: float
    gnn_proba: float


class PredictionsResponse(BaseModel):
    n: int
    predictions: list[PredictionRow]


class ModelMetrics(BaseModel):
    model: str
    split: str
    n: int
    precision: float
    recall: float
    f1: float
    pr_auc: float


class MetricsResponse(BaseModel):
    metrics: list[ModelMetrics]


class NodeDetail(BaseModel):
    id: str
    time_step: int
    label: int
    features: dict[str, float]
    xgboost_proba: float | None = None
    gnn_proba: float | None = None
    in_neighbors: list[str]
    out_neighbors: list[str]
