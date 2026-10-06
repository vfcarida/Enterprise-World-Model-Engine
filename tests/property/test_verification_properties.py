"""Property-based and invariant tests for verification subsystem.

Conforms to Track T3 & P14:
- Property hash stability, determinism, and permutation-invariance in fingerprint folding.
- Strict non-mutation of traces and trajectories during verification.
- Causality topological ordering invariants (independent branches free to interleave).
- Adapter stubs (MoonLightSTRELAdapter and LLMSoftCheckAdapter).
"""

from __future__ import annotations

import pytest

from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.verification import (
    LLMSoftCheckAdapter,
    MoonLightSTRELAdapter,
    OracleEdge,
    OracleGraph,
    OracleNode,
    PropertySpec,
    always,
    compute_property_hash,
    evaluate_oracle_graph,
    evaluate_stl_bounded,
    fold_properties_into_fingerprint,
    predicate,
)


def test_property_spec_hash_determinism_and_permutation_invariance() -> None:
    """Property hashes must be deterministic, and folding into fingerprints must be permutation invariant."""
    spec_a = PropertySpec(
        property_id="prop_safety",
        name="Buffer never negative",
        description="Buffer never negative",
        property_type="stl_bounded",
        definition={"formula": "always(buffer >= 0)"},
        parameters={"k1": 0, "k2": 10},
    )
    spec_b = PropertySpec(
        property_id="prop_liveness",
        name="Eventual recovery",
        description="Eventual recovery",
        property_type="oracle_graph",
        definition={"nodes": ["dispatch", "deliver"]},
    )

    hash_a1 = compute_property_hash(spec_a)
    hash_a2 = compute_property_hash(spec_a)
    assert hash_a1 == hash_a2
    assert spec_a.property_hash == hash_a1

    # Permutation invariance when folding into fingerprints
    base_fp = "a" * 64
    folded_ab = fold_properties_into_fingerprint(base_fp, [spec_a, spec_b])
    folded_ba = fold_properties_into_fingerprint(base_fp, [spec_b, spec_a])
    assert folded_ab == folded_ba
    assert folded_ab != base_fp

    # Modifying any specification detail strictly changes property hash
    spec_a_modified = spec_a.model_copy(update={"definition": {"formula": "always(buffer >= 1)"}})
    assert spec_a_modified.property_hash != spec_a.property_hash


def test_verification_never_mutates_trace() -> None:
    """Verifying an oracle graph must strictly read the systemic trace without modifying it."""
    trace = SystemicTrace()
    trace.add_node(
        node_id="e1",
        step=1,
        category="dispatch",
        label="Dispatch Event",
        evidence_level=EvidenceLevel.STRUCTURAL,
        details={"mission_id": "M1"},
    )
    trace.add_node(
        node_id="e2",
        step=3,
        category="delivered",
        label="Delivered Event",
        evidence_level=EvidenceLevel.INTERVENTIONAL,
        details={"mission_id": "M1"},
    )

    nodes_before = dict(trace.nodes)
    edges_before = list(trace.edges)

    graph = OracleGraph.create(
        nodes=[
            OracleNode(
                node_id="dispatch", category="dispatch", expected_details={"mission_id": "M1"}
            ),
            OracleNode(
                node_id="delivered", category="delivered", expected_details={"mission_id": "M1"}
            ),
        ],
        edges=[
            OracleEdge(source="dispatch", target="delivered", min_step_delay=1, max_step_delay=5),
        ],
    )

    result = evaluate_oracle_graph(graph, trace)
    assert result.satisfied is True

    # Trace state must be completely unmutated
    assert trace.nodes == nodes_before
    assert trace.edges == edges_before


def test_verification_never_mutates_trajectory() -> None:
    """Verifying STL properties must strictly read the trajectory without modifying states or steps."""
    world = World(
        initial_state=WorldState(
            step=0,
            entities=[Entity(id="node_1", type="router")],
            resources=[Resource(id="bandwidth", current=100.0)],
        )
    )
    scenario = Scenario(scenario_id="immutability_check", horizon=4, samples=1, seed=42)
    engine = SimulationEngine()
    result = engine.run(world, scenario)
    traj = result.trajectories[0]

    # Deep snapshot of trajectory state
    init_state_fp = traj.initial_state.fingerprint
    final_state_fp = traj.final_state.fingerprint
    steps_count = len(traj.steps)

    prop = always(predicate("bandwidth", ">=", 50.0), k1=0, k2=3)
    verdict = evaluate_stl_bounded(prop, traj)

    assert verdict.satisfied is True
    assert traj.initial_state.fingerprint == init_state_fp
    assert traj.final_state.fingerprint == final_state_fp
    assert len(traj.steps) == steps_count


def test_oracle_branch_independence() -> None:
    """Independent branches in the Oracle DAG may interleave arbitrarily without causal violation."""
    # DAG has two independent parallel branches:
    # A1 -> A2
    # B1 -> B2
    # Trace interleave: A1 (step 1) -> B1 (step 2) -> A2 (step 3) -> B2 (step 4)
    graph = OracleGraph.create(
        nodes=[
            OracleNode(node_id="A1", category="typeA"),
            OracleNode(node_id="A2", category="typeA"),
            OracleNode(node_id="B1", category="typeB"),
            OracleNode(node_id="B2", category="typeB"),
        ],
        edges=[
            OracleEdge(source="A1", target="A2", min_step_delay=1),
            OracleEdge(source="B1", target="B2", min_step_delay=1),
        ],
    )

    trace = SystemicTrace()
    trace.add_node(node_id="e1", step=1, category="typeA", label="A1")
    trace.add_node(node_id="e2", step=2, category="typeB", label="B1")
    trace.add_node(node_id="e3", step=3, category="typeA", label="A2")
    trace.add_node(node_id="e4", step=4, category="typeB", label="B2")

    result = evaluate_oracle_graph(graph, trace)
    assert result.satisfied is True
    assert len(result.violations) == 0


def test_stubs_adapters() -> None:
    """Test documented adapter stubs: MoonLightSTRELAdapter and LLMSoftCheckAdapter."""
    with pytest.raises(
        SimulationConfigurationError, match="MoonLightSTRELAdapter requires the '\\[strel\\]' extra"
    ):
        MoonLightSTRELAdapter(script_path="path/to/spec.strel")

    # LLM adapter uses sync/async callable without SDK dependency
    def mock_llm_callable(prompt: str) -> str:
        assert "Rubric:" in prompt
        return "The check passes. Answer: YES."

    llm_adapter = LLMSoftCheckAdapter(llm_callable=mock_llm_callable, rubric="Valid packaging")
    is_valid = llm_adapter.verify_node(node_label="package_check", details={"sealed": True})
    assert is_valid is True

    # Test error when callable is missing
    unconfigured_adapter = LLMSoftCheckAdapter()
    with pytest.raises(SimulationConfigurationError, match="requires a host-provided llm_callable"):
        unconfigured_adapter.verify_node("node", {})
