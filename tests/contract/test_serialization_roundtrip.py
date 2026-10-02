"""Contract test (AC-002): All public serializable models round-trip losslessly via canonical JSON."""

from __future__ import annotations

import pytest

from ewm_engine.constraints.results import ConstraintPhase, ConstraintResult, ConstraintSeverity
from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.spec import (
    ComponentSpec,
    EntitySpec,
    ResourceSpec,
    WorldMetadataSpec,
    WorldSpec,
)
from ewm_engine.core.state import WorldState
from ewm_engine.dynamics.base import TransitionResult
from ewm_engine.provenance.evidence import EvidenceLevel
from ewm_engine.provenance.metadata import ComponentVersion, Provenance
from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.serialization.json import dump_model_json, load_model_json
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import StepRecord, Trajectory, TrajectoryStatus


@pytest.mark.contract
def test_action_serialization_roundtrip() -> None:
    """AC-002: Action round-trips losslessly via canonical JSON."""
    action = Action(
        id="act_transfer_1",
        type="transfer_resource",
        actor_id="warehouse_manager",
        parameters={"source": "wh_1", "target": "wh_2", "amount": 50.0},
        timestamp=12.5,
        priority=2,
    )
    json_str = dump_model_json(action)
    reloaded = load_model_json(Action, json_str)

    assert reloaded == action
    assert reloaded.id == "act_transfer_1"
    assert reloaded.parameters["amount"] == 50.0
    assert reloaded.schema_version == "1.0.0"


@pytest.mark.contract
def test_world_state_serialization_roundtrip() -> None:
    """AC-002: WorldState round-trips losslessly and preserves canonical fingerprint."""
    state = WorldState(
        step=3,
        timestamp=45.0,
        entities=[
            Entity(id="e1", type="node", attributes={"capacity": 100}, tags=("alpha", "beta")),
            Entity(id="e2", type="terminal", attributes={"active": True}),
        ],
        relationships=[
            Relationship(source="e1", target="e2", type="pipeline", attributes={"bandwidth": 10.0}),
        ],
        resources=[
            Resource(
                id="r1", current=75.5, min_value=0.0, max_value=100.0, unit="liters", entity_id="e1"
            ),
        ],
        memory={"total_dispatches": 14},
        active_rules={"priority_mode": True},
        context={"ambient_temp_c": 21.5},
    )

    json_str = dump_model_json(state)
    reloaded = load_model_json(WorldState, json_str)

    assert reloaded == state
    assert reloaded.fingerprint == state.fingerprint
    assert reloaded.get_resource("r1").current == 75.5
    assert reloaded.schema_version == "1.0.0"


@pytest.mark.contract
def test_world_spec_serialization_roundtrip() -> None:
    """AC-002: WorldSpec round-trips losslessly and builds identical World."""
    spec = WorldSpec(
        world=WorldMetadataSpec(name="RoundTripWorld", step=1, timestamp=10.0, memory={"k": "v"}),
        entities=[
            EntitySpec(id="depot", type="facility", attributes={"region": "north"}),
        ],
        relationships=[],
        resources=[
            ResourceSpec(
                id="fuel",
                current=200.0,
                min_value=0.0,
                max_value=500.0,
                unit="gallons",
                entity_id="depot",
            ),
        ],
        constraints=[
            ComponentSpec(type="capacity", parameters={"resource_id": "fuel", "severity": "hard"}),
        ],
        dynamics=ComponentSpec(type="transfer", parameters={"action_type": "transfer_fuel"}),
    )

    json_str = dump_model_json(spec)
    reloaded = load_model_json(WorldSpec, json_str)

    assert reloaded == spec
    assert reloaded.world.name == "RoundTripWorld"
    assert len(reloaded.constraints) == 1
    assert reloaded.constraints[0].type == "capacity"

    w1 = spec.build_world()
    w2 = reloaded.build_world()
    assert w1.initial_state.fingerprint == w2.initial_state.fingerprint


@pytest.mark.contract
def test_scenario_serialization_roundtrip() -> None:
    """AC-002: Scenario round-trips losslessly via canonical JSON."""
    scenario = Scenario(
        name="HighDemandStress",
        scenario_id="scenario_stress_01",
        horizon=20,
        samples=100,
        seed=12345,
        intervention=Intervention(
            id="expand_buffer",
            description="Double buffer capacity during stress peak",
            parameters={"multiplier": 2.0},
        ),
        metadata={"author": "RiskEngineeringTeam"},
    )

    json_str = dump_model_json(scenario)
    reloaded = load_model_json(Scenario, json_str)

    assert reloaded == scenario
    assert reloaded.fingerprint == scenario.fingerprint
    assert reloaded.horizon == 20
    assert reloaded.schema_version == "1.0.0"


@pytest.mark.contract
def test_trajectory_serialization_roundtrip() -> None:
    """AC-002: Trajectory round-trips losslessly preserving steps, trace, and status."""
    initial_state = WorldState(
        step=0,
        timestamp=0.0,
        entities=[Entity(id="e1", type="node")],
        resources=[Resource(id="r1", current=50.0, min_value=0.0, max_value=100.0)],
    )

    next_state = initial_state.update_resource("r1", delta=-10.0).advance_time(delta_t=1.0)

    trace = SystemicTrace()
    trace.add_node(
        "node_1",
        step=0,
        category="intervention",
        label="Start shock",
        evidence_level=EvidenceLevel.STRUCTURAL,
    )
    trace.add_node(
        "node_2",
        step=1,
        category="state_change",
        label="Drop inventory",
        evidence_level=EvidenceLevel.ASSUMED,
    )
    trace.add_edge(
        "node_1", "node_2", step=1, relation="influences", evidence_level=EvidenceLevel.ASSUMED
    )

    step_record = StepRecord(
        step=1,
        timestamp=1.0,
        state_hash=initial_state.fingerprint,
        actions_proposed=(Action(id="a1", type="consume", parameters={"amount": 10.0}),),
        actions_accepted=(Action(id="a1", type="consume", parameters={"amount": 10.0}),),
        exogenous_events=(
            ExogenousEvent(id="ev1", type="power_dip", parameters={"duration_s": 5}),
        ),
        transition_result=TransitionResult(
            next_state=next_state,
            applied_changes={"r1": -10.0},
            evidence_level=EvidenceLevel.STRUCTURAL,
        ),
        constraint_violations=(
            ConstraintResult(
                constraint_id="capacity_r1",
                constraint_version="1.0.0",
                phase=ConstraintPhase.POST_TRANSITION,
                severity=ConstraintSeverity.SOFT,
                satisfied=False,
                message="Resource r1 exceeded warning threshold",
            ),
        ),
        step_metrics={"inventory_r1": 40.0},
    )

    trajectory = Trajectory(
        sample_id=0,
        seed=42,
        initial_state=initial_state,
        systemic_trace=trace,
        status=TrajectoryStatus.COMPLETED,
    )
    trajectory.append_step(step_record, resulting_state=next_state)
    trajectory.finalize()

    json_str = dump_model_json(trajectory)
    reloaded = load_model_json(Trajectory, json_str)

    assert reloaded.sample_id == trajectory.sample_id
    assert reloaded.seed == trajectory.seed
    assert reloaded.status == trajectory.status
    assert len(reloaded.steps) == 1
    assert reloaded.steps[0].step_metrics["inventory_r1"] == 40.0
    assert reloaded.final_state.get_resource("r1").current == 40.0
    assert len(reloaded.systemic_trace.nodes) == 2
    assert len(reloaded.systemic_trace.edges) == 1
    assert reloaded.schema_version == "1.0.0"


@pytest.mark.contract
def test_provenance_serialization_roundtrip() -> None:
    """AC-002: Provenance round-trips losslessly via canonical JSON."""
    provenance = Provenance(
        engine_version="0.1.0",
        scenario_fingerprint="abc123canonicalscenario",
        initial_state_fingerprint="def456canonicalstate",
        seed=999,
        horizon=50,
        samples=200,
        components=[
            ComponentVersion(
                component_id="DeterministicTransferDynamics", component_version="1.0.0"
            ),
        ],
        constraint_versions=[
            ComponentVersion(component_id="ResourceCapacityConstraint", component_version="1.0.0"),
        ],
        runtime_metadata={"host": "ci-runner", "python_version": "3.12"},
    )

    json_str = dump_model_json(provenance)
    reloaded = load_model_json(Provenance, json_str)

    assert reloaded == provenance
    assert reloaded.scenario_fingerprint == "abc123canonicalscenario"
    assert reloaded.samples == 200
    assert len(reloaded.components) == 1
    assert len(reloaded.constraint_versions) == 1
    assert reloaded.schema_version == "1.0.0"
