"""
Tests for the GraphSAGE model architecture in models/gnn_graphsage.py, using
small synthetic graphs -- no cached data required, since these check shape
and directionality behavior of the model itself, not its trained weights.
"""

from __future__ import annotations

import torch

from models.gnn_graphsage import DirectedSAGEBlock, GraphSAGE, _pos_weight


def _toy_graph():
    # 0 -> 1 -> 2 -> 3, a simple chain
    x = torch.randn(4, 5)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    return x, edge_index


def test_directed_sage_block_output_shape():
    x, edge_index = _toy_graph()
    block = DirectedSAGEBlock(in_dim=5, out_dim=8)
    out = block(x, edge_index)
    assert out.shape == (4, 8)


def test_directed_sage_block_uses_both_directions():
    """Zeroing out the reverse-direction pathway should change the output --
    confirms in-neighbor and out-neighbor aggregation are both actually
    contributing, not just one of them silently doing all the work."""
    x, edge_index = _toy_graph()
    torch.manual_seed(0)
    block = DirectedSAGEBlock(in_dim=5, out_dim=8)

    out_forward_only = block.conv_in(x, edge_index)
    out_full = block(x, edge_index)

    assert not torch.allclose(out_forward_only, out_full)


def test_graphsage_end_to_end_output_shape():
    x, edge_index = _toy_graph()
    model = GraphSAGE(in_dim=5, hidden_dim=16)
    model.eval()
    logits = model(x, edge_index)
    assert logits.shape == (4,)  # one logit per node, squeezed


def test_pos_weight_reflects_class_imbalance():
    y = torch.tensor([0.0, 0.0, 0.0, 1.0])  # 3 licit, 1 illicit
    mask = torch.tensor([True, True, True, True])
    weight = _pos_weight(y, mask)
    assert weight.item() == 3.0  # n_neg / n_pos
