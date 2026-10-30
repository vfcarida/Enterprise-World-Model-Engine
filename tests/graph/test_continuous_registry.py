"""Unit and benchmark tests for ContinuousGraphRegistry and node feature schemas."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from ewm_engine.graph.aiops_features import AIOpsNodeFeatures
from ewm_engine.graph.continuous_registry import ContinuousGraphRegistry, Edge, Node
from ewm_engine.graph.financial_features import FinancialNodeFeatures


@pytest.mark.unit
def test_node_feature_schemas_financial_and_aiops() -> None:
    """Verify FinancialNodeFeatures and AIOpsNodeFeatures serialization and round-trip."""
    fin = FinancialNodeFeatures(
        market_cap=2850.5,
        vwap=192.35,
        bid_ask_spread_bps=1.25,
        sector_id=8,
        volatility_30d=0.24,
    )
    fin_dict = fin.to_dict()
    assert list(fin_dict.keys()) == FinancialNodeFeatures.feature_names()
    assert fin_dict["sector_id"] == 8.0
    assert FinancialNodeFeatures.from_dict(fin_dict) == fin
    assert fin.to_vector().shape == (5,)
    assert fin.to_vector().dtype == np.float32

    ops = AIOpsNodeFeatures(
        cpu_utilization_pct=72.5,
        memory_bytes=4.2e9,
        error_rate_per_sec=0.05,
        latency_p95_ms=18.4,
        active_connections=240.0,
    )
    ops_dict = ops.to_dict()
    assert list(ops_dict.keys()) == AIOpsNodeFeatures.feature_names()
    assert AIOpsNodeFeatures.from_dict(ops_dict) == ops
    assert ops.to_vector().shape == (5,)
    assert ops.to_vector().dtype == np.float32


@pytest.mark.unit
def test_continuous_registry_crud_and_snapshot_operations() -> None:
    """Verify node/edge CRUD, missing-node error handling, and snapshot history."""
    registry = ContinuousGraphRegistry(max_snapshot_history=3)

    n1 = Node("NVDA", "equity", {"market_cap": 3000.0, "vwap": 130.0})
    n2 = Node("TSM", "equity", {"market_cap": 900.0, "vwap": 175.0})
    n3 = Node("AMD", "equity", {"market_cap": 250.0, "vwap": 155.0})

    registry.add_node(n1)
    registry.add_node(n2)
    registry.add_node(n3)

    # Adding edge with non-existent node raises ValueError
    with pytest.raises(ValueError, match="Cannot add edge"):
        registry.add_edge(Edge("NVDA", "UNKNOWN", "supply_chain", 0.8))

    registry.add_edge(Edge("TSM", "NVDA", "supply_chain", 0.9))
    registry.add_edge(Edge("TSM", "AMD", "supply_chain", 0.7))
    registry.add_edge(Edge("NVDA", "AMD", "sector_correlation", 0.6))

    x_mat, a_mat = registry.take_snapshot()
    assert x_mat.shape == (3, 2)
    assert a_mat.shape == (3, 3)
    # Sorted order: AMD (0), NVDA (1), TSM (2)
    assert a_mat[2, 1] == pytest.approx(0.9)
    assert a_mat[2, 0] == pytest.approx(0.7)
    assert a_mat[1, 0] == pytest.approx(0.6)

    # Remove single edge
    registry.remove_edge("NVDA", "AMD")
    assert ("NVDA", "AMD") not in registry.edges
    assert "AMD" not in registry.adjacency_list["NVDA"]

    # Remove node TSM (removes all outgoing edges from TSM and incoming edges to TSM)
    registry.add_edge(Edge("NVDA", "TSM", "dependency", 0.4))
    registry.remove_node("TSM")
    assert "TSM" not in registry.nodes
    assert ("TSM", "NVDA") not in registry.edges
    assert ("TSM", "AMD") not in registry.edges
    assert ("NVDA", "TSM") not in registry.edges

    # Removing non-existent node is a safe no-op
    registry.remove_node("NON_EXISTENT")

    # Verify max_snapshot_history cap
    for _ in range(5):
        registry.take_snapshot()
    assert len(registry.snapshot_history) == 3


@pytest.mark.unit
def test_registry_10k_nodes_50k_edges_under_1_second() -> None:
    """Add 10,000 nodes + 50,000 edges in <1 second."""
    registry = ContinuousGraphRegistry()

    t0 = time.perf_counter()
    for i in range(10000):
        registry.add_node(Node(f"n_{i}", "equity", {"f1": float(i), "f2": 1.0}))

    for i in range(10000):
        src = f"n_{i}"
        for offset in range(1, 6):
            tgt = f"n_{(i + offset) % 10000}"
            registry.add_edge(Edge(src, tgt, "supply_chain", 0.5))

    elapsed_s = time.perf_counter() - t0
    assert len(registry.nodes) == 10000
    assert len(registry.edges) == 50000
    assert elapsed_s < 1.0, f"Adding 10k nodes + 50k edges took {elapsed_s:.3f}s (>= 1.0s)"


@pytest.mark.benchmark
def test_registry_10k_edge_updates_under_50ms(benchmark: Any) -> None:
    """Perform 10,000 edge additions/deletions in <50ms."""
    registry = ContinuousGraphRegistry()
    for i in range(500):
        registry.add_node(Node(f"svc_{i}", "service", {"cpu": 50.0}))

    edges_to_toggle = [
        Edge(f"svc_{i % 500}", f"svc_{(i * 7 + 1) % 500}", "api_call", 1.0) for i in range(5000)
    ]

    def _run_10k_edge_ops() -> int:
        for e in edges_to_toggle:
            registry.add_edge(e)
        for e in edges_to_toggle:
            registry.remove_edge(e.source_id, e.target_id)
        return len(edges_to_toggle) * 2

    count = benchmark(_run_10k_edge_ops)
    assert count == 10000

    t0 = time.perf_counter()
    _run_10k_edge_ops()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 50.0, f"10k edge updates took {elapsed_ms:.2f}ms (>= 50ms)"


@pytest.mark.benchmark
def test_registry_snapshot_1000_nodes_under_10ms(benchmark: Any) -> None:
    """Snapshot generation for 1,000-node graph in <10ms."""
    registry = ContinuousGraphRegistry()
    for i in range(1000):
        registry.add_node(
            Node(
                f"node_{i:04d}",
                "equity",
                {
                    "market_cap": float(i),
                    "vwap": 100.0,
                    "spread": 1.5,
                    "sector": float(i % 11),
                    "vol": 0.2,
                },
            )
        )
    for i in range(1000):
        registry.add_edge(Edge(f"node_{i:04d}", f"node_{(i + 1) % 1000:04d}", "supply_chain", 0.8))
        registry.add_edge(
            Edge(f"node_{i:04d}", f"node_{(i + 5) % 1000:04d}", "sector_correlation", 0.4)
        )

    x_mat, a_mat = benchmark(registry.take_snapshot)
    assert x_mat.shape == (1000, 5)
    assert a_mat.shape == (1000, 1000)

    samples_ms: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        registry.take_snapshot()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    median_ms = float(np.median(samples_ms))
    assert median_ms < 10.0, f"1,000-node snapshot median took {median_ms:.2f}ms (>= 10ms)"
