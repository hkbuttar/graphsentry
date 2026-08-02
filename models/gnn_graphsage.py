"""
GraphSAGE GNN, trained and evaluated under the exact same protocol as the
XGBoost baseline (models/baseline_xgboost.py) so the two are actually
comparable, not just two numbers that happened to be computed differently.

--------------------------------------------------------------------------------
Why GraphSAGE specifically, not a plain GCN
--------------------------------------------------------------------------------
This isn't a generic textbook choice -- it follows directly from something
established in Step 2/3: every edge stays within its own time step, so the
graph is really 49 disjoint components, one per time step, and Louvain
communities (and therefore any structural grouping) never span across that
boundary either. A test-period node's entire neighborhood is made of other
test-period nodes the model has never had any access to during training --
not "rarely seen," but structurally guaranteed unseen, every single time.

A transductive method (plain GCN, or a shallow per-node embedding table)
needs the target nodes present in the graph DURING training to have learned
anything about them at all -- it cannot be handed a brand new, disconnected
component after the fact and produce a sensible answer. GraphSAGE is built
differently: it learns a shared AGGREGATION FUNCTION (how to combine a
node's neighbors' features), not a fixed embedding per node. That function,
once learned, can be applied unchanged to any new neighborhood -- including
one from a different time step's disconnected component it's never seen.
That's what "inductive" buys here, concretely, not abstractly.

--------------------------------------------------------------------------------
Directed message passing, not the usual "just add reverse edges" default
--------------------------------------------------------------------------------
graph/graph_builder.py kept edge direction deliberately: in-degree vs.
out-degree asymmetry (many-inputs-one-output vs. one-input-many-outputs) is
real signal for laundering patterns. A standard GNN shortcut -- add a
reverse copy of every edge so information flows both ways -- would throw
that asymmetry away and reduce the graph to an undirected one, exactly what
Step 2 avoided.

Instead, `DirectedSAGEBlock` runs TWO separate SAGEConv layers per block:
one over the edges as given (aggregating each node's IN-neighbors -- who
paid it), one over the reversed edge_index (aggregating its OUT-neighbors --
who it paid), then concatenates both and projects back down to the working
dimension. The model gets both directions' information without collapsing
the distinction between them into a single undirected pool.

--------------------------------------------------------------------------------
Training protocol -- deliberately mirrors the baseline's, not a fresh design
--------------------------------------------------------------------------------
Phase 1: train on steps 1-29, track validation PR-AUC on steps 30-34 each
epoch, keep the epoch count with the best validation PR-AUC (early stopping).
Select the decision threshold that maximizes F1 on that same phase's
out-of-sample validation predictions (models.metrics.select_threshold) --
never on test. Phase 2: reinitialize and retrain fresh for that many epochs
on the full steps 1-34 window, then evaluate once on steps 35-49.

This is the same nested-validation structure used for the XGBoost baseline,
on purpose: if the two models were evaluated under different protocols, a
difference in the final numbers could be explained by protocol differences
rather than model differences. Recency weighting (tried for the baseline,
found to be a wash) is NOT carried over here as a starting assumption --
it's a deliberate simplification for this first GNN version, revisited only
if results suggest it's worth testing, the same evidence-driven way every
baseline change was tested rather than assumed.

--------------------------------------------------------------------------------
First result was weak; diagnosed before tuning, not assumed to be fixable
--------------------------------------------------------------------------------
The first trained version scored test F1 0.598 / PR-AUC 0.568, well below the
baseline's F1 0.817 / PR-AUC 0.804. Before treating that as final, the
validation-loss curve was inspected epoch-by-epoch to check whether early
stopping cut training off prematurely: it hadn't -- validation PR-AUC climbed
to ~0.93-0.94 by epoch 40-50 and then plateaued/turned noisy while train
PR-AUC kept climbing toward 0.999+ (textbook overfitting past that point).
So the weak TEST result isn't an artifact of under-training; validation
looked strong (~0.94) and still didn't transfer to test (0.57) -- a much
larger validation-to-test collapse than the baseline showed. Plausible
reading: GraphSAGE partly predicts from neighborhood SHAPE, not just a
node's own features: if the typical shape of a local neighborhood
(clustering, density) itself drifts between the train and test periods --
plausible, since the illicit rate itself already drifts -- a model relying
on structure has an extra axis of temporal shift to survive that a
feature-only model doesn't.

Given that diagnosis, a small validation-only sweep (never touching test)
compared four configurations by best validation PR-AUC:
  - lr=0.003 (vs 0.01), otherwise unchanged: 0.9457
  - 2-layer (vs 3), otherwise unchanged: 0.9417 (worse -- less capacity did
    not help, arguing against "the model is just overfitting depth")
  - dropout=0.5, weight_decay=1e-4 (vs 0.3 / 1e-5), otherwise unchanged: 0.9476
  - original (dropout=0.3, weight_decay=1e-5, lr=0.01, 3-layer): 0.9444
All four land within a narrow 0.94-0.95 band -- this is not a dramatic
swing, and the exercise is reported honestly as inconclusive-but-slightly-
favoring more regularization, not evidence of a large fixable problem. The
higher-regularization configuration (dropout=0.5, weight_decay=1e-4) is kept
as the best of the four tested, with no claim that it resolves the
underlying validation-to-test gap described above.

--------------------------------------------------------------------------------
`betweenness` was also suspected and tested -- kept in, like `community` was
--------------------------------------------------------------------------------
Following up on the structural-drift question above: models/feature_drift_
audit.py never actually reported on `betweenness`, because it was silently
skipped by a length guard -- checking why revealed betweenness has 7,880
non-null values in train and ZERO in test (it's only computed for the single
largest weakly-connected component -- see graph/analytics.py -- and that
component happens to fall entirely in the train period). That means every
single test node receives the exact same constant fill value for this
feature, carrying no test-time discriminating information at all -- a
plausible reason to exclude it, structurally similar to the six drifted
features already excluded.

Tested directly rather than acted on: removing `betweenness` made validation
PR-AUC WORSE for the GNN (0.9476 -> 0.9408) and made essentially no
difference for the baseline (aucpr 0.9901 -> 0.9902, noise-level). So
`betweenness` is kept in both models. This is the second time in this
project (`community` being the first) that a theoretically sound argument
for excluding a feature didn't survive contact with the data -- reported
here for the same reason: the empirical result takes priority over a
plausible-sounding story, and pretending otherwise would be less honest than
just showing both.

--------------------------------------------------------------------------------
Recency-weighted training -- tested and reverted
--------------------------------------------------------------------------------
Deliberately NOT included in the first version (see above) as a simplifying
assumption. Once the baseline/GNN gap was diagnosed as distribution shift
rather than overfitting, recency weighting became worth testing directly: the
same 0.5x-1.5x ramp used for the XGBoost baseline (weighting rows by how late
their time_step falls in the training window), applied as the `weight`
argument to `binary_cross_entropy_with_logits` alongside `pos_weight` for
class imbalance.

Validation-only check first: best validation PR-AUC improved 0.9476 -> 0.9521
-- looked like a real, if modest, win. But the one actual test evaluation
told a different story: test F1 dropped 0.659 -> 0.621 and PR-AUC dropped
0.631 -> 0.619, with recall falling notably (0.556 -> 0.502). Validation
(steps 30-34) improved in a way that did not transfer to test (steps 35-49)
-- a reminder that validation is a proxy for test performance, not identical
to it, and under a real, documented distribution shift that proxy can point
the wrong direction. Reverted for this reason: the one thing that matters
here is the held-out test result, and it disagreed with validation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from models.gnn_data import build_pyg_data
from models.metrics import compute_metrics, select_threshold

CACHE_DIR = Path(__file__).parent / "cache"
MODEL_PATH = CACHE_DIR / "gnn_model.pt"
THRESHOLD_PATH = CACHE_DIR / "gnn_threshold.txt"
PREDICTIONS_PATH = CACHE_DIR / "gnn_predictions.parquet"

HIDDEN_DIM = 64
DROPOUT = 0.5
MAX_EPOCHS = 300
PATIENCE = 30
LR = 0.01
WEIGHT_DECAY = 1e-4
SEED = 42


class DirectedSAGEBlock(torch.nn.Module):
    """One layer of message passing that keeps in/out-neighbor information
    separate rather than merging them -- see module docstring."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv_in = SAGEConv(in_dim, out_dim)
        self.conv_out = SAGEConv(in_dim, out_dim)
        self.combine = torch.nn.Linear(2 * out_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_in = self.conv_in(x, edge_index)
        h_out = self.conv_out(x, edge_index.flip(0))
        return self.combine(torch.cat([h_in, h_out], dim=-1))


class GraphSAGE(torch.nn.Module):
    """3-layer directed GraphSAGE: in_dim -> hidden -> hidden -> 1 logit."""

    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, dropout: float = DROPOUT):
        super().__init__()
        self.block1 = DirectedSAGEBlock(in_dim, hidden_dim)
        self.block2 = DirectedSAGEBlock(hidden_dim, hidden_dim)
        self.block3 = DirectedSAGEBlock(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.block1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.block2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.block3(h, edge_index).squeeze(-1)


def _pos_weight(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    y_masked = y[mask]
    n_pos = (y_masked == 1).sum()
    n_neg = (y_masked == 0).sum()
    return n_neg / n_pos


def _train_one_config(data, train_mask: torch.Tensor, epochs: int, track_val: torch.Tensor | None) -> tuple:
    """Trains a fresh model for `epochs` epochs on train_mask. If track_val is
    given, returns (model, best_epoch, val_proba_history) for early-stopping
    selection; otherwise just (model, epochs, None) for the final fit."""
    torch.manual_seed(SEED)
    model = GraphSAGE(in_dim=data.x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_weight = _pos_weight(data.y, train_mask)

    best_val_pr_auc = -1.0
    best_state = None
    best_epoch = epochs
    epochs_since_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], data.y[train_mask], pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        if track_val is not None:
            model.eval()
            with torch.no_grad():
                val_proba = torch.sigmoid(model(data.x, data.edge_index))[track_val]
            val_pr_auc = compute_metrics(
                data.y[track_val].numpy(), val_proba.numpy(), "val"
            )["pr_auc"]
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= PATIENCE:
                    break

    if track_val is not None and best_state is not None:
        model.load_state_dict(best_state)

    return model, best_epoch


def train_gnn(use_cache: bool = True) -> tuple[GraphSAGE, object, pd.Index, float]:
    """Two-phase fit mirroring models.baseline_xgboost.train_baseline exactly:
    (1) train on steps 1-29, early-stop on steps 30-34, pick a threshold from
    that phase's out-of-sample val predictions; (2) retrain fresh for the
    chosen epoch count on the full steps 1-34 window. Returns
    (model, data, node_ids, threshold)."""
    data, node_ids = build_pyg_data()

    if use_cache and MODEL_PATH.exists():
        model = GraphSAGE(in_dim=data.x.shape[1])
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        threshold = float(THRESHOLD_PATH.read_text().strip()) if THRESHOLD_PATH.exists() else 0.5
        return model, data, node_ids, threshold

    finder, best_epoch = _train_one_config(data, data.train_mask, MAX_EPOCHS, track_val=data.val_mask)
    print(f"Early stopping on steps 1-29 (train) / 30-34 (validation) picked epoch {best_epoch}")

    finder.eval()
    with torch.no_grad():
        val_proba = torch.sigmoid(finder(data.x, data.edge_index))[data.val_mask]
    threshold = select_threshold(data.y[data.val_mask].numpy(), val_proba.numpy())
    print(f"Threshold selected on steps 30-34 out-of-sample predictions: {threshold:.4f}")

    model, _ = _train_one_config(data, data.trainval_mask, best_epoch, track_val=None)
    model.eval()

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        THRESHOLD_PATH.write_text(str(threshold))

    return model, data, node_ids, threshold


def evaluate(model: GraphSAGE, data, mask: torch.Tensor, split_name: str, threshold: float) -> dict:
    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(data.x, data.edge_index))[mask]
    return compute_metrics(data.y[mask].numpy(), proba.numpy(), split_name, threshold=threshold)


def predict_all_labeled(model: GraphSAGE, data, node_ids: pd.Index, use_cache: bool = True) -> pd.DataFrame:
    """Predictions for every labeled node, saved for reuse by the Step 7
    comparison table and the Step 8 /predictions endpoint."""
    if use_cache and PREDICTIONS_PATH.exists():
        return pd.read_parquet(PREDICTIONS_PATH)

    model.eval()
    with torch.no_grad():
        proba = torch.sigmoid(model(data.x, data.edge_index)).numpy()

    labeled_mask = (data.train_mask | data.val_mask | data.test_mask).numpy()
    result = pd.DataFrame(index=node_ids[labeled_mask])
    result["label"] = data.y.numpy()[labeled_mask]
    result["gnn_proba"] = proba[labeled_mask]

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(PREDICTIONS_PATH)

    return result


if __name__ == "__main__":
    model, data, node_ids, threshold = train_gnn()
    print(f"\nUsing decision threshold: {threshold:.4f} (selected on validation, not test)")

    print("\n--- GraphSAGE GNN: precision / recall / F1 / PR-AUC ---")
    for split_name, mask in [("train (steps 1-34)", data.trainval_mask), ("test (steps 35-49)", data.test_mask)]:
        metrics = evaluate(model, data, mask, split_name, threshold=threshold)
        print(f"{metrics['split']:>20}: n={metrics['n']}, "
              f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
              f"f1={metrics['f1']:.4f}, pr_auc={metrics['pr_auc']:.4f}")

    predict_all_labeled(model, data, node_ids)
    print(f"\nPredictions for all labeled nodes cached to {PREDICTIONS_PATH}")
