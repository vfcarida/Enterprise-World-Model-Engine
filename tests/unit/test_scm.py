"""Unit tests for StructuralCausalModel (SCM) compilation and Pearl's causal surgery."""

from __future__ import annotations

from typing import Any

from ewm_engine.core.entities import Entity
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.scm import (
    StructuralAssignment,
    StructuralCausalModel,
)
from ewm_engine.core.spec import WorldSpec
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics


class MockDemandShockSource:
    """Event source generating demand shocks for SCM testing."""

    def __init__(self, shock_amount: float = 15.0) -> None:
        self.name = "DemandShockSource"
        self.shock_amount = shock_amount

    def sample(self, state: WorldState, step: int, rng: Any) -> list[ExogenousEvent]:
        return [
            ExogenousEvent(
                id=f"shock_{step}",
                type="demand_shock",
                severity=1.0,
                parameters={"amount": self.shock_amount},
            )
        ]


def _build_test_world() -> World:
    state = WorldState(
        resources=[
            Resource(id="inventory_A", current=100.0, min_value=0.0, max_value=500.0),
            Resource(id="inventory_B", current=50.0, min_value=0.0, max_value=500.0),
            Resource(id="cash", current=1000.0, min_value=0.0, max_value=10000.0),
        ],
        entities=[
            Entity(id="warehouse_1", type="warehouse", attributes={"capacity": 500}),
        ],
    )
    dyn = DeterministicTransferDynamics(
        action_type="transfer_resource",
        name="InventoryTransferDynamics",
    )
    world = World(
        state=state,
        dynamics=dyn,
        event_sources=[MockDemandShockSource()],
    )
    return world


def test_scm_compilation_from_world() -> None:
    world = _build_test_world()
    scm = world.as_scm(horizon=3)

    assert isinstance(scm, StructuralCausalModel)
    assert "resource:inventory_A" in scm.endogenous_variables
    assert "resource:inventory_B" in scm.endogenous_variables
    assert "resource:cash" in scm.endogenous_variables

    # Exogenous variables compiled for event sources and dynamics
    assert any("event:DemandShockSource" in k for k in scm.exogenous_variables)
    assert any("dynamics:InventoryTransferDynamics" in k for k in scm.exogenous_variables)

    # Structural assignments
    asgn_a = scm.assignments["resource:inventory_A"]
    assert isinstance(asgn_a, StructuralAssignment)
    assert asgn_a.target == "resource:inventory_A"
    assert asgn_a.is_intervened is False
    assert len(scm.fingerprint) == 64


def test_scm_compilation_from_world_spec() -> None:
    yaml_content = """
schema_version: "1.0.0"
world:
  name: "SCMSpecWorld"
resources:
  - id: "fuel"
    current: 250.0
    min_value: 0.0
    max_value: 1000.0
  - id: "cargo"
    current: 50.0
    min_value: 0.0
    max_value: 100.0
"""
    spec = WorldSpec.from_yaml(yaml_content)
    scm = spec.as_scm(horizon=1)

    assert isinstance(scm, StructuralCausalModel)
    assert "resource:fuel" in scm.endogenous_variables
    assert "resource:cargo" in scm.endogenous_variables
    assert len(scm.fingerprint) == 64


def test_scm_compilation_from_state_and_graph_view() -> None:
    world = _build_test_world()
    state = world.initial_state

    # Via WorldState.as_scm()
    scm_from_state = state.as_scm()
    assert isinstance(scm_from_state, StructuralCausalModel)
    assert "resource:inventory_A" in scm_from_state.endogenous_variables

    # Via HeterogeneousGraphView.as_scm()
    graph_view = state.as_graph()
    scm_from_graph = graph_view.as_scm()
    assert isinstance(scm_from_graph, StructuralCausalModel)
    assert "resource:inventory_A" in scm_from_graph.endogenous_variables


def test_scm_causal_surgery_mutilation() -> None:
    world = _build_test_world()
    scm = world.as_scm(horizon=1)

    # Before surgery, incoming edges exist
    incoming_before = [edge for edge in scm.causal_graph.edges if edge[1] == "resource:inventory_A"]
    assert len(incoming_before) > 0

    # Perform causal surgery: do(inventory_A = 200.0)
    mutilated = scm.intervene({"inventory_A": 200.0})

    assert mutilated.fingerprint != scm.fingerprint
    asgn_a = mutilated.assignments["resource:inventory_A"]
    assert asgn_a.is_intervened is True
    assert asgn_a.intervened_value == 200.0
    assert asgn_a.parents == ()

    # All incoming edges to inventory_A must be severed in mutilated DAG
    incoming_after = [
        edge for edge in mutilated.causal_graph.edges if edge[1] == "resource:inventory_A"
    ]
    assert len(incoming_after) == 0

    # Non-intervened variable remains unsevered
    asgn_b = mutilated.assignments["resource:inventory_B"]
    assert asgn_b.is_intervened is False


def test_scm_identifiability_check() -> None:
    world = _build_test_world()
    scm = world.as_scm(horizon=1)

    # Identifiability check delegates to causal graph backdoor criterion
    id_res = scm.check_identifiability(
        treatment="resource:inventory_A",
        outcome="resource:inventory_B",
    )
    assert id_res.treatment == "resource:inventory_A"
    assert id_res.outcome == "resource:inventory_B"
