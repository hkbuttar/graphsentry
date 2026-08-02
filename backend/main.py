"""
FastAPI app. Thin by design: every endpoint reads from state loaded once at
startup (backend/state.py) and every computation it needs already exists in
graph/, features/, or models/ -- this file wires routes together, it doesn't
reimplement anything.

--------------------------------------------------------------------------------
CORS: ALLOWED_ORIGINS env var, not a hardcoded list
--------------------------------------------------------------------------------
Locally this defaults to common SvelteKit/Vite dev ports. In production
(Step 10), Render sets ALLOWED_ORIGINS to the actual Vercel frontend URL --
same pattern used elsewhere in this author's projects (alpha-signal-lab),
so the backend never needs a code change to point at a different frontend
deployment, just an env var.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import metrics, network, node, predictions
from backend.state import load_state

DEFAULT_DEV_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graphsentry = load_state()
    yield


app = FastAPI(title="GraphSentry API", lifespan=lifespan)

allowed_origins = os.environ.get("ALLOWED_ORIGINS", DEFAULT_DEV_ORIGINS).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(network.router)
app.include_router(predictions.router)
app.include_router(metrics.router)
app.include_router(node.router)


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "graphsentry-api"}
