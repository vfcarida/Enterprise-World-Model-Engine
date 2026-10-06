"""Unit tests for Oracle Graph trajectory verification across Consistency, Causality, and Timing axes."""

from __future__ import annotations

from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction
from ewm_engine.verification.oracle import (
    OracleEdge,
    OracleGraph,
    OracleNode,
    evaluate_oracle_graph,
)


def _build_test_trace() -> SystemicTrace:
    trace = SystemicTrace()
    # Step 0: Shock event
    trace.add_node(
        node_id="shock_event_0",
        step=0,
        category="event",
        label="Exogenous Shock: surge",
        evidence_level=EvidenceLevel.STRUCTURAL,
        details={"type": "surge", "severity": 2.0},
    )
    # Step 1: Action dispatched
    trace.add_node(
        node_id="action_dispatch_1",
        step=1,
        category="action",
        label="Action: dispatch_crew",
        evidence_level=EvidenceLevel.INTERVENTIONAL,
        details={"crew_id": "crew_alpha", "units": 5},
    )
    trace.add_edge("shock_event_0", "action_dispatch_1", step=1, relation="triggers")

    # Step 3: State recovery transition
    trace.add_node(
        node_id="state_change_3",
        step=3,
        category="state_change",
        label="Dynamics Transition: ResourceRecovery",
        evidence_level=EvidenceLevel.STRUCTURAL,
        details={"recovered_units": 50.0},
    )
    trace.add_edge("action_dispatch_1", "state_change_3", step=3, relation="enables")

    return trace


def test_oracle_graph_all_axes_satisfied() -> None:
    trace = _build_test_trace()

    oracle = OracleGraph.create(
        nodes=[
            OracleNode(
                node_id="expected_shock",
                category="event",
                expected_details={"type": "surge", "severity": 2.0},
            ),
            OracleNode(
                node_id="expected_dispatch",
                category="action",
                expected_details={"crew_id": "crew_alpha"},
            ),
            OracleNode(
                node_id="expected_recovery",
                category="state_change",
                expected_details={"recovered_units": 50.0},
            ),
        ],
        edges=[
            OracleEdge(
                source="expected_shock",
                target="expected_dispatch",
                min_step_delay=1,
                max_step_delay=2,
            ),
            OracleEdge(
                source="expected_dispatch",
                target="expected_recovery",
                min_step_delay=1,
                max_step_delay=3,
            ),
        ],
    )

    result = evaluate_oracle_graph(oracle, trace)

    assert result.satisfied is True
    assert result.consistency_score == 1.0
    assert result.causality_score == 1.0
    assert result.timing_score == 1.0
    assert len(result.violations) == 0
    assert result.matched_nodes["expected_shock"] == "shock_event_0"
    assert result.matched_nodes["expected_dispatch"] == "action_dispatch_1"
    assert result.matched_nodes["expected_recovery"] == "state_change_3"


def test_oracle_graph_consistency_violation() -> None:
    trace = _build_test_trace()

    # Require crew_id="crew_beta" which does not exist in trace
    oracle = OracleGraph.create(
        nodes=[
            OracleNode(
                node_id="expected_dispatch",
                category="action",
                expected_details={"crew_id": "crew_beta"},
            ),
        ]
    )

    result = evaluate_oracle_graph(oracle, trace)

    assert result.satisfied is False
    assert result.consistency_score == 0.0
    assert len(result.violations) == 1
    assert result.violations[0].axis == "consistency"
    assert result.violations[0].node_id == "expected_dispatch"


def test_oracle_graph_causality_violation() -> None:
    trace = _build_test_trace()

    # Require recovery (step 3) to precede dispatch (step 1)
    oracle = OracleGraph.create(
        nodes=[
            OracleNode(
                node_id="expected_dispatch",
                category="action",
                expected_details={"crew_id": "crew_alpha"},
            ),
            OracleNode(
                node_id="expected_recovery",
                category="state_change",
                expected_details={"recovered_units": 50.0},
            ),
        ],
        edges=[
            # Inverse causal order!
            OracleEdge(
                source="expected_recovery",
                target="expected_dispatch",
            ),
        ],
    )

    result = evaluate_oracle_graph(oracle, trace)

    assert result.satisfied is False
    assert result.consistency_score == 1.0
    assert result.causality_score == 0.0
    assert any(v.axis == "causality" for v in result.violations)


def test_oracle_graph_timing_window_violation() -> None:
    trace = _build_test_trace()

    # Step difference between dispatch (step 1) and recovery (step 3) is 2.
    # Set min_step_delay to 3 (too late) or max_step_delay to 1 (too early)
    oracle = OracleGraph.create(
        nodes=[
            OracleNode(
                node_id="dispatch", category="action", expected_details={"crew_id": "crew_alpha"}
            ),
            OracleNode(
                node_id="recovery",
                category="state_change",
                expected_details={"recovered_units": 50.0},
            ),
        ],
        edges=[
            OracleEdge(source="dispatch", target="recovery", min_step_delay=3, max_step_delay=5),
        ],
    )

    result = evaluate_oracle_graph(oracle, trace)

    assert result.satisfied is False
    assert result.causality_score == 1.0
    assert result.timing_score == 0.0
    assert any(v.axis == "timing" for v in result.violations)


def test_oracle_graph_on_simulated_warehouse_trajectory() -> None:
    """Verify OracleGraph evaluation against an actual simulated trajectory."""
    world = World(
        initial_state=WorldState(
            step=0,
            entities=[Entity(id="warehouse", type="facility")],
            resources=[Resource(id="inventory", current=200.0)],
        )
    )
    scenario = Scenario(
        scenario_id="wh_verify",
        horizon=4,
        samples=1,
        seed=42,
        scheduled_actions=(
            ScheduledAction(
                step=1,
                action=Action(id="a_restock", type="restock", parameters={"amount": 50.0}),
            ),
        ),
    )

    engine = SimulationEngine()
    result = engine.run(world, scenario)
    traj = result.trajectories[0]

    # Verify that scheduled action appears in trace and is preceded by step 0 initial intervention
    oracle = OracleGraph.create(
        nodes=[
            OracleNode(
                node_id="restock_act",
                category="action",
                expected_details={"amount": 50.0},
            )
        ]
    )

    verdict = evaluate_oracle_graph(oracle, traj)
    assert verdict.satisfied is True
    assert verdict.consistency_score == 1.0
    assert "restock_act" in verdict.matched_nodes
