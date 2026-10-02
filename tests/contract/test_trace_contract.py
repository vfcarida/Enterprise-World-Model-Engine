"""Contract tests for AC-011: Systemic trace integrity, step binding, relation honesty, and EvidenceLevel."""

from __future__ import annotations

import pytest

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import ActionTransferAvailabilityConstraint
from ewm_engine.core.actions import Intervention
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace, TraceEdge
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


@pytest.mark.contract
def test_trace_edge_requires_existing_nodes() -> None:
    """AC-011: Adding a TraceEdge must reference existing nodes in the SystemicTrace."""
    trace = SystemicTrace()
    trace.add_node(
        node_id="node_a",
        step=0,
        category="event",
        label="Event A",
        evidence_level=EvidenceLevel.STRUCTURAL,
    )

    # Missing target node
    with pytest.raises(ValueError, match="target node 'missing_target' does not exist"):
        trace.add_edge(source="node_a", target="missing_target")

    # Missing source node
    with pytest.raises(ValueError, match="source node 'missing_source' does not exist"):
        trace.add_edge(source="missing_source", target="node_a")


@pytest.mark.contract
def test_trace_rejects_causes_relation() -> None:
    """AC-011: Systemic traces strictly reject 'causes' as a relation to prevent unverified causal claims."""
    trace = SystemicTrace()
    trace.add_node(node_id="n1", step=0, category="event", label="Node 1")
    trace.add_node(node_id="n2", step=1, category="state_change", label="Node 2")

    # Rejected via add_edge
    with pytest.raises(ValueError, match="relation 'causes' is forbidden"):
        trace.add_edge(source="n1", target="n2", relation="causes")

    # Case-insensitive rejection
    with pytest.raises(ValueError, match="relation 'causes' is forbidden"):
        trace.add_edge(source="n1", target="n2", relation="CAUSES")

    # Rejected via direct TraceEdge model validation
    with pytest.raises(ValueError, match="relation 'causes' is forbidden"):
        TraceEdge(source="n1", target="n2", relation="causes")


@pytest.mark.contract
def test_trace_edges_carry_step_relation_and_evidence_level() -> None:
    """AC-011: Edges carry step, descriptive relation, and EvidenceLevel."""
    trace = SystemicTrace()
    trace.add_node(node_id="src", step=2, category="intervention", label="Policy Shift")
    trace.add_node(node_id="dst", step=3, category="action", label="Order Dispatched")

    # Explicit step and structural relation
    edge1 = trace.add_edge(
        source="src",
        target="dst",
        step=3,
        relation="conditions",
        evidence_level=EvidenceLevel.INTERVENTIONAL,
    )
    assert edge1.step == 3
    assert edge1.relation == "conditions"
    assert edge1.evidence_level == EvidenceLevel.INTERVENTIONAL

    # Omitted step defaults to target node's step
    edge2 = trace.add_edge(
        source="src",
        target="dst",
        relation="drives",
        evidence_level=EvidenceLevel.STRUCTURAL,
    )
    assert edge2.step == 3
    assert edge2.relation == "drives"
    assert edge2.evidence_level == EvidenceLevel.STRUCTURAL

    # Default relation is 'influences' and default evidence is ASSUMED
    edge3 = trace.add_edge(source="src", target="dst")
    assert edge3.relation == "influences"
    assert edge3.evidence_level == EvidenceLevel.ASSUMED


@pytest.mark.contract
def test_trace_mixed_evidence_levels_and_mermaid_export() -> None:
    """AC-011: Trace supports mixed EvidenceLevels across nodes and edges without fabricating causality."""
    trace = SystemicTrace()
    trace.add_node(
        node_id="iv_0",
        step=0,
        category="intervention",
        label="Pricing Intervention",
        evidence_level=EvidenceLevel.INTERVENTIONAL,
    )
    trace.add_node(
        node_id="shock_1",
        step=1,
        category="event",
        label="Macro Shock",
        evidence_level=EvidenceLevel.PREDICTIVE,
    )
    trace.add_node(
        node_id="trans_1",
        step=1,
        category="state_change",
        label="Revenue Shift",
        evidence_level=EvidenceLevel.STRUCTURAL,
    )

    trace.add_edge(
        source="iv_0",
        target="trans_1",
        relation="perturbs",
        evidence_level=EvidenceLevel.INTERVENTIONAL,
    )
    trace.add_edge(
        source="shock_1",
        target="trans_1",
        relation="influences",
        evidence_level=EvidenceLevel.ASSUMED,
    )

    mermaid = trace.to_mermaid()
    assert 'iv_0["Pricing Intervention (interventional)"]' in mermaid
    assert 'shock_1["Macro Shock (predictive)"]' in mermaid
    assert 'trans_1["Revenue Shift (structural)"]' in mermaid
    assert 'iv_0 -->|"perturbs"| trans_1' in mermaid
    assert 'shock_1 -->|"influences"| trans_1' in mermaid
    assert "causes" not in mermaid.lower()


@pytest.mark.contract
def test_simulation_engine_trace_contract_conformance(sample_world_state: WorldState) -> None:
    """AC-011: Trace generated by SimulationEngine strictly conforms to AC-011 invariants."""
    world = World(
        state=sample_world_state,
        dynamics=DeterministicTransferDynamics(),
        constraints=ConstraintRegistry([ActionTransferAvailabilityConstraint()]),
    )
    actor = ThresholdReplenishmentActor(
        actor_id="test_actor",
        source_resource="stock_wh1",
        target_resource="stock_wh2",
        reorder_point=100.0,
        order_quantity=20.0,
    )
    world.add_actor(actor)

    intervention = Intervention(id="policy_rep", description="Replenishment intervention")
    scenario = Scenario(
        name="ContractTest", horizon=2, samples=1, seed=42, intervention=intervention
    )

    engine = SimulationEngine()
    result = engine.run(world=world, scenario=scenario)

    trace = result.trajectories[0].systemic_trace
    assert len(trace.nodes) > 0
    assert len(trace.edges) > 0

    for edge in trace.edges:
        # 1. Edge source and target must exist in nodes
        assert edge.source in trace.nodes, f"Edge source '{edge.source}' not in trace nodes"
        assert edge.target in trace.nodes, f"Edge target '{edge.target}' not in trace nodes"

        # 2. Step must be valid integer
        assert isinstance(edge.step, int)
        assert edge.step >= 0

        # 3. Relation must not be 'causes'
        assert edge.relation != "causes"
        assert edge.relation != ""

        # 4. EvidenceLevel must be a valid EvidenceLevel enum
        assert isinstance(edge.evidence_level, EvidenceLevel)
