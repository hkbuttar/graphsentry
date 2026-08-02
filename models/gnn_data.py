"""
Builds the torch_geometric.data.Data object for the GraphSAGE GNN: node
features, directed edge_index, labels, and train/val/test masks. This is the
graph-native counterpart to features/build_features.py's flat table -- same
underlying data, reshaped for a model that consumes graph structure directly
rather than one row per node with no notion of neighbors.

--------------------------------------------------------------------------------
Node features: same exclusions as the baseline, one different exclusion added
--------------------------------------------------------------------------------
Input features are the merged table's raw + structural columns, minus:
  - The 6 severely time-drifted features identified in
    models/feature_drift_audit.py (feat_100/101/103/136/137/139) -- the same
    exclusion applied to the XGBoost baseline, for the same reason: their
    value ranges barely overlap between train and test periods.
  - `community`, excluded here for a DIFFERENT reason than in the baseline.
    The baseline kept `community` despite community IDs being 100% disjoint
    between train and test (206 train IDs, 109 test IDs, zero overlap),
    because XGBoost's categorical split-finding has a defined, LEARNED
    fallback for unseen categories (a default split direction chosen during
    training). A GNN embedding table has no equivalent: every test-period
    community ID is unseen, so every single test node would map to the same
    constant "unknown" embedding vector, which is not learned behavior, just
    a fixed placeholder repeated 100% of the time. There's no reason to
    expect that helps, and the reasoning that made the baseline's inclusion
    defensible doesn't transfer. Community-*like* signal isn't lost, though:
    message passing over the actual graph edges is exactly the mechanism
    that lets a GNN discover locally-dense neighborhoods on its own, without
    needing a non-transferable ID handed to it directly.

`betweenness` is NaN for ~96% of nodes (only computed for the largest
time-step component -- see graph/analytics.py). Full per-component coverage
was tried (closing the gap where every test node otherwise gets the same
constant filled value) and measurably made both this model and the baseline
worse on test -- see graph/analytics.py's docstring for the investigation
and why it was reverted. Missing values here are filled with the
train-period mean before standardization, which maps them to approximately
0 after scaling -- a neutral value rather than a missing-data signal a
neural net has no principled way to interpret on its own.

--------------------------------------------------------------------------------
Normalization uses train-period statistics only
--------------------------------------------------------------------------------
Neural nets are sensitive to input scale in a way tree models aren't (trees
split on thresholds regardless of scale; a neural net's weights and
activations are shaped by it directly). Mean/std are computed from ALL nodes
in the train period (steps 1-34, labeled and unlabeled alike -- no label
information is used for this, so it isn't leakage), then applied to every
node, train and test alike. This mirrors real deployment: you'd fit a scaler
on all past data you have, not just the labeled fraction of it.

--------------------------------------------------------------------------------
Masks mirror the baseline's nested validation split exactly
--------------------------------------------------------------------------------
train_mask: labeled, steps 1-29. val_mask: labeled, steps 30-34 (early
stopping / threshold selection only). trainval_mask: labeled, steps 1-34
(the final-fit window, after the right epoch count is chosen on val).
test_mask: labeled, steps 35-49, touched only once for final evaluation.
Same boundaries as models/baseline_xgboost.py, so the two models' reported
numbers are actually comparable rather than evaluated under different rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from data.loader import load_elliptic
from features.build_features import DEFAULT_TRAIN_MAX_STEP, build_feature_table
from models.baseline_xgboost import EXCLUDED_DRIFTED_FEATURES, VAL_MIN_STEP

EXCLUDED_FEATURES = [*EXCLUDED_DRIFTED_FEATURES, "community"]


def _feature_columns(full_table: pd.DataFrame) -> list[str]:
    return [c for c in full_table.columns if c not in ("label", *EXCLUDED_FEATURES)]


def build_pyg_data() -> tuple[Data, pd.Index]:
    """Returns (data, node_ids) where node_ids[i] is the txId for row i of
    every tensor in `data` -- needed to map predictions back to transactions."""
    full_table = build_feature_table()
    ds = load_elliptic()

    node_ids = full_table.index
    pos_of = pd.Series(np.arange(len(node_ids)), index=node_ids)

    feature_cols = _feature_columns(full_table)
    features = full_table[feature_cols].copy()
    features["betweenness"] = features["betweenness"].fillna(features.loc[
        full_table["time_step"] <= DEFAULT_TRAIN_MAX_STEP, "betweenness"
    ].mean())

    train_period = full_table["time_step"] <= DEFAULT_TRAIN_MAX_STEP
    mean = features.loc[train_period].mean()
    std = features.loc[train_period].std().replace(0, 1.0)
    features = (features - mean) / std

    x = torch.tensor(features.to_numpy(), dtype=torch.float)

    src = ds.edges["txId1"].map(pos_of).to_numpy()
    dst = ds.edges["txId2"].map(pos_of).to_numpy()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    y = torch.tensor(full_table["label"].to_numpy(), dtype=torch.float)

    time_step = full_table["time_step"].to_numpy()
    labeled = full_table["label"].to_numpy() >= 0
    train_mask = torch.tensor(labeled & (time_step < VAL_MIN_STEP))
    val_mask = torch.tensor(labeled & (time_step >= VAL_MIN_STEP) & (time_step <= DEFAULT_TRAIN_MAX_STEP))
    trainval_mask = torch.tensor(labeled & (time_step <= DEFAULT_TRAIN_MAX_STEP))
    test_mask = torch.tensor(labeled & (time_step > DEFAULT_TRAIN_MAX_STEP))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.trainval_mask = trainval_mask
    data.test_mask = test_mask

    return data, node_ids


if __name__ == "__main__":
    data, node_ids = build_pyg_data()
    print(data)
    print(f"\nFeature columns ({data.x.shape[1]}): {_feature_columns(build_feature_table())}")
    print(f"\ntrain_mask: {data.train_mask.sum().item()} nodes")
    print(f"val_mask:   {data.val_mask.sum().item()} nodes")
    print(f"trainval_mask: {data.trainval_mask.sum().item()} nodes")
    print(f"test_mask:  {data.test_mask.sum().item()} nodes")
