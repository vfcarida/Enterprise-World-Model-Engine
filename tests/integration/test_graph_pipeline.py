"""End-to-end integration tests for ContinuousGraphRegistry -> EvolveGCNTrainer pipeline."""

from __future__ import annotations

import pytest
import torch

from ewm_engine.graph.aiops_features import AIOpsNodeFeatures
from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry, Edge, Node
from ewm_engine.graph.financial_features import FinancialNodeFeatures
from ewm_engine.graph.training_pipeline import EvolveGCNTrainer


@pytest.mark.integration
def test_streaming_financial_graph_to_evolvegcn_training_convergence() -> None:
    """Ingest streaming edge updates -> registry -> take snapshots -> train EvolveGCN -> verify convergence."""
    torch.manual_seed(42)
    registry = ContinuousGraphRegistry()

    tickers = ["NVDA", "TSM", "AMD", "ASML", "AAPL", "MSFT"]
    for idx, ticker in enumerate(tickers):
        feats = FinancialNodeFeatures(
            market_cap=500.0 + idx * 200.0,
            vwap=100.0 + idx * 15.0,
            bid_ask_spread_bps=1.2,
            sector_id=8,
            volatility_30d=0.25,
        )
        registry.add_node(Node(node_id=ticker, node_type="equity", features=feats.to_dict()))

    # Stream edge updates and take 8 temporal snapshots with smooth feature evolution
    for step in range(8):
        for idx, ticker in enumerate(tickers):
            updated_feats = FinancialNodeFeatures(
                market_cap=(500.0 + idx * 200.0) * (1.0 + 0.01 * step),
                vwap=(100.0 + idx * 15.0) * (1.0 + 0.005 * step),
                bid_ask_spread_bps=1.2 + 0.05 * step,
                sector_id=8,
                volatility_30d=0.25 + 0.002 * step,
            )
            registry.add_node(
                Node(node_id=ticker, node_type="equity", features=updated_feats.to_dict())
            )

        # Mutate supply-chain and correlation edges over time
        registry.add_edge(Edge("ASML", "TSM", "supply_chain", weight=0.9))
        registry.add_edge(Edge("TSM", "NVDA", "supply_chain", weight=0.85))
        registry.add_edge(Edge("TSM", "AMD", "supply_chain", weight=0.75))
        if step % 2 == 0:
            registry.add_edge(Edge("NVDA", "MSFT", "supply_chain", weight=0.65))
        else:
            registry.remove_edge("NVDA", "MSFT")

        registry.take_snapshot()

    assert len(registry.snapshot_history) == 8

    trainer = EvolveGCNTrainer(
        registry=registry,
        in_channels=len(FinancialNodeFeatures.feature_names()),
        hidden_channels=32,
        out_channels=16,
        num_layers=2,
        learning_rate=0.01,
    )

    epoch_losses = trainer.train_on_snapshot_sequence(num_epochs=25, sequence_length=8)
    assert len(epoch_losses) == 25
    # Verify training loss decreases significantly over epochs
    assert epoch_losses[-1] < epoch_losses[0]

    # Verify trained model produces valid final embeddings on latest snapshot
    trainer.model.eval()
    latest_x, latest_a = registry.snapshot_history[-1][1:]
    x_tensor = torch.from_numpy(latest_x)
    edge_index = torch.from_numpy(latest_a.nonzero()[0]).unsqueeze(0).repeat(2, 1)
    with torch.no_grad():
        final_embeddings = trainer.model(x_tensor, edge_index)
    assert final_embeddings.shape == (len(tickers), 16)


@pytest.mark.integration
def test_aiops_ephemeral_pods_dynamic_node_set_training() -> None:
    """Verify EvolveGCNTrainer handles changing node counts (ephemeral Kubernetes pods) across snapshots."""
    torch.manual_seed(99)
    registry = ContinuousGraphRegistry()

    # Empty registry or <2 snapshots returns empty loss list safely
    trainer = EvolveGCNTrainer(registry=registry, in_channels=5, hidden_channels=16, out_channels=8)
    assert trainer.train_on_snapshot_sequence(num_epochs=3) == []

    base_services = ["gateway-svc", "auth-svc", "payment-svc"]
    for svc in base_services:
        feats = AIOpsNodeFeatures(50.0, 1e8, 0.01, 12.0, 100.0)
        registry.add_node(Node(svc, "service", feats.to_dict()))
    registry.add_edge(Edge("gateway-svc", "auth-svc", "api_call", 1.0))
    registry.take_snapshot()

    # Ephemeral pod scales out in snapshot 2
    registry.add_node(
        Node(
            "payment-pod-ephemeral", "pod", AIOpsNodeFeatures(80.0, 2e8, 0.1, 25.0, 50.0).to_dict()
        )
    )
    registry.add_edge(Edge("payment-svc", "payment-pod-ephemeral", "dependency", 0.9))
    registry.take_snapshot()

    # Ephemeral pod terminates in snapshot 3
    registry.remove_node("payment-pod-ephemeral")
    registry.take_snapshot()

    losses = trainer.train_on_snapshot_sequence(num_epochs=5, sequence_length=3)
    assert len(losses) == 5
    assert all(loss >= 0.0 for loss in losses)
