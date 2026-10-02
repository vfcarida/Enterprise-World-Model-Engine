"""Unit tests for provenance, metadata, and systemic traces."""

from __future__ import annotations

from ewm_engine.core.state import WorldState
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.metadata import SimulationMetadata
from ewm_engine.provenance.trace import SystemicTrace


def test_simulation_metadata_fingerprint(sample_world_state: WorldState) -> None:
    """Test that simulation metadata produces a deterministic SHA-256 fingerprint."""
    meta1 = SimulationMetadata(
        scenario_id="experiment_01",
        world_hash=sample_world_state.state_hash,
        dynamics_name="CompositeDynamics",
        constraint_versions={"cap_1": "1.0.0"},
        random_seed=42,
        horizon=24,
        samples=100,
    )
    fp1 = meta1.fingerprint
    assert len(fp1) == 64

    meta2 = SimulationMetadata(
        scenario_id="experiment_01",
        world_hash=sample_world_state.state_hash,
        dynamics_name="CompositeDynamics",
        constraint_versions={"cap_1": "1.0.0"},
        random_seed=42,
        horizon=24,
        samples=100,
    )
    assert meta2.fingerprint == fp1


def test_systemic_trace_dag_and_mermaid() -> None:
    """Test building a systemic trace DAG and exporting to Mermaid syntax."""
    trace = SystemicTrace()
    trace.add_node(
        node_id="policy_change",
        step=0,
        category="intervention",
        label="Intervention: Expedited Replenishment",
        evidence_level=EvidenceLevel.INTERVENTIONAL,
    )
    trace.add_node(
        node_id="demand_shock",
        step=1,
        category="event",
        label="Exogenous Shock: Demand Surge",
        evidence_level=EvidenceLevel.STRUCTURAL,
    )
    trace.add_node(
        node_id="stock_depletion",
        step=1,
        category="state_change",
        label="Stock Depletion in WH North",
        evidence_level=EvidenceLevel.STRUCTURAL,
    )
    trace.add_edge("policy_change", "stock_depletion", relation="mitigates")
    trace.add_edge("demand_shock", "stock_depletion", relation="causes")

    assert len(trace.nodes) == 3
    assert len(trace.edges) == 2

    mermaid_str = trace.to_mermaid()
    assert "flowchart TD" in mermaid_str
    assert "Intervention: Expedited Replenishment" in mermaid_str
    assert '-->|"causes"|' in mermaid_str

    # Test NetworkX export
    nx_graph = trace.to_networkx()
    assert nx_graph.number_of_nodes() == 3
    assert nx_graph.number_of_edges() == 2
