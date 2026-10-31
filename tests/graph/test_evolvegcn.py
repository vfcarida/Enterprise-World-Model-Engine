"""Unit, property, and benchmark tests for EvolveGCN and EvolveGCNLayer."""
# ruff: noqa: E402

from __future__ import annotations

import time
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from ewm_engine.graph.evolvegcn import EvolveGCN, EvolveGCNLayer


@pytest.mark.unit
@pytest.mark.parametrize("recurrent_type", ["gru", "lstm"])
def test_evolvegcn_100_node_forward_and_10_step_sequence(recurrent_type: str) -> None:
    """Forward pass with 100-node graph -> (100, out_channels) and 10-step state evolution."""
    torch.manual_seed(42)
    model = EvolveGCN(
        in_channels=5,
        hidden_channels=16,
        out_channels=8,
        num_layers=2,
        recurrent_type=recurrent_type,
    )

    num_nodes = 100
    src = torch.arange(0, num_nodes, dtype=torch.long)
    dst = (src + 1) % num_nodes
    edge_index = torch.stack([src, dst], dim=0)

    weight_snapshots: list[Any] = []
    for step in range(10):
        x = torch.randn(num_nodes, 5) + float(step) * 0.1
        embeddings = model(x, edge_index)
        assert embeddings.shape == (100, 8)
        assert not torch.isnan(embeddings).any()
        assert model.hidden_states[0] is not None
        assert model.hidden_states[1] is not None
        layer0 = model.layers[0]
        assert isinstance(layer0, EvolveGCNLayer)
        weight_snapshots.append(layer0.gcn.lin.weight.detach().clone())

    # Verify weights actually evolved over the 10 steps
    assert not torch.allclose(weight_snapshots[0], weight_snapshots[-1])

    # Reset hidden states
    model.reset_hidden_states()
    assert all(s is None for s in model.hidden_states)


@pytest.mark.unit
def test_evolvegcn_gradient_flow_and_validation() -> None:
    """Verify gradients flow through both recurrent unit and GCN layers during backward."""
    torch.manual_seed(7)
    model = EvolveGCN(in_channels=5, hidden_channels=12, out_channels=4, num_layers=2)
    x = torch.randn(30, 5, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

    # Step 1 initializes hidden state; Step 2 exercises recurrent hidden-to-hidden weights (weight_hh)
    _ = model(x, edge_index)
    out = model(x, edge_index)
    target = torch.zeros_like(out)
    loss = F.mse_loss(out, target)
    loss.backward()

    assert x.grad is not None
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} received no gradient"
        assert torch.any(param.grad != 0.0), f"Parameter {name} has all-zero gradient"

    # Also test empty edge_index and (E, 2) edge_index handling
    empty_edges = torch.zeros((2, 0), dtype=torch.long)
    out_empty = model(x.detach(), empty_edges)
    assert out_empty.shape == (30, 4)

    transposed_edges = edge_index.t().contiguous()
    out_transposed = model(x.detach(), transposed_edges)
    assert out_transposed.shape == (30, 4)

    with pytest.raises(ValueError, match="Unknown recurrent_type"):
        EvolveGCNLayer(5, 4, recurrent_type="invalid_rnn")

    with pytest.raises(ValueError, match="num_layers must be >= 1"):
        EvolveGCN(5, 8, 4, num_layers=0)


@pytest.mark.benchmark
def test_evolvegcn_1000_node_forward_under_50ms(benchmark: Any) -> None:
    """Performance: Process 1,000-node graph in <50ms per snapshot."""
    torch.manual_seed(42)
    model = EvolveGCN(in_channels=5, hidden_channels=32, out_channels=16, num_layers=2)
    model.eval()

    x = torch.randn(1000, 5)
    src = torch.arange(0, 1000, dtype=torch.long).repeat(3)
    dst = torch.cat(
        [
            (torch.arange(0, 1000, dtype=torch.long) + 1) % 1000,
            (torch.arange(0, 1000, dtype=torch.long) + 7) % 1000,
            (torch.arange(0, 1000, dtype=torch.long) + 19) % 1000,
        ]
    )
    edge_index = torch.stack([src, dst], dim=0)

    def _forward_step() -> Any:
        with torch.no_grad():
            model.reset_hidden_states()
            return model(x, edge_index)

    out = benchmark(_forward_step)
    assert out.shape == (1000, 16)

    t0 = time.perf_counter()
    _forward_step()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"1,000-node EvolveGCN forward took {elapsed_ms:.2f}ms (>= 50ms)"


@pytest.mark.property
@given(
    delta_scale=st.floats(min_value=0.05, max_value=2.0, allow_nan=False, allow_infinity=False),
)
def test_property_graph_homophily_and_temporal_drift_consistency(delta_scale: float) -> None:
    """Property: Connected nodes share closer embeddings than opposite clusters; drift scales with perturbation."""
    torch.manual_seed(123)
    model = EvolveGCN(in_channels=4, hidden_channels=8, out_channels=8, num_layers=1)
    model.eval()

    # Two connected pairs: (0 <-> 1) with positive features, (2 <-> 3) with negative features
    x_base = torch.tensor(
        [
            [1.0, 1.0, 0.5, 0.5],
            [1.0, 0.9, 0.5, 0.4],
            [-1.0, -1.0, -0.5, -0.5],
            [-1.0, -0.9, -0.5, -0.4],
        ],
        dtype=torch.float32,
    )
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

    with torch.no_grad():
        model.reset_hidden_states()
        emb_base = model(x_base, edge_index)

        # Connected nodes (0, 1) must have higher cosine similarity than disconnected (0, 2)
        sim_connected = float(
            F.cosine_similarity(emb_base[0].unsqueeze(0), emb_base[1].unsqueeze(0)).item()
        )
        sim_disconnected = float(
            F.cosine_similarity(emb_base[0].unsqueeze(0), emb_base[2].unsqueeze(0)).item()
        )
        assert sim_connected > sim_disconnected

        # Temporal consistency: larger perturbation -> larger embedding drift
        model.reset_hidden_states()
        emb_small = model(x_base + delta_scale * 0.1, edge_index)

        model.reset_hidden_states()
        emb_large = model(x_base + delta_scale * 2.0, edge_index)

        drift_small = float(torch.norm(emb_small - emb_base).item())
        drift_large = float(torch.norm(emb_large - emb_base).item())
        assert drift_large > drift_small
