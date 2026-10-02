"""Property test (AC-005): Branches derived from the same snapshot never contaminate each other."""

from __future__ import annotations

import pytest

from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.simulation.branching import branch_scenario, branch_world
from ewm_engine.simulation.scenario import Scenario, ScheduledAction


@pytest.mark.property
def test_scenario_branch_isolation_property() -> None:
    """AC-005: Branching a Scenario never mutates the parent or its fingerprint."""
    initial_state = WorldState(
        entities=[Entity(id="plant_1", type="factory")],
        resources=[Resource(id="widgets", current=100.0, min_value=0.0, max_value=500.0)],
    )

    parent_scenario = Scenario(
        name="BaselinePlan",
        scenario_id="scen_base",
        horizon=10,
        samples=3,
        seed=42,
        initial_state=initial_state,
        metadata={"author": "Team A", "priority": "high"},
    )
    parent_fingerprint_before = parent_scenario.fingerprint

    # Create two divergent branches from parent
    branch_a = parent_scenario.branch(
        scenario_id="scen_branch_a",
        name="AggressiveBranch",
        intervention=Intervention(
            id="expedite",
            description="Accelerate production",
            parameters={"shift_multiplier": 1.5},
        ),
        scheduled_actions=[
            ScheduledAction(
                step=1,
                action=Action(id="act_expedite", type="boost", parameters={"boost": 20.0}),
            )
        ],
        seed=1001,
    )

    branch_b = branch_scenario(
        parent_scenario,
        name="ConservativeBranch",
        scenario_id="scen_branch_b",
        scheduled_actions=[
            ScheduledAction(
                step=3,
                action=Action(id="act_save", type="conserve", parameters={"rate": 0.8}),
            )
        ],
        seed=2002,
    )

    # Assert parent was NOT modified
    assert parent_scenario.fingerprint == parent_fingerprint_before
    assert parent_scenario.scenario_id == "scen_base"
    assert parent_scenario.intervention is None
    assert len(parent_scenario.scheduled_actions) == 0

    # Assert branch A and branch B have distinct identities but preserve initial state
    assert branch_a.scenario_id == "scen_branch_a"
    assert branch_b.scenario_id == "scen_branch_b"
    assert branch_a.initial_state is not None
    assert branch_b.initial_state is not None
    assert parent_scenario.initial_state is not None
    assert branch_a.initial_state.fingerprint == parent_scenario.initial_state.fingerprint
    assert branch_b.initial_state.fingerprint == parent_scenario.initial_state.fingerprint
    assert branch_a.fingerprint != branch_b.fingerprint


@pytest.mark.property
def test_world_branch_isolation_under_simulation() -> None:
    """AC-005: Running simulation on a branched world leaves the parent world completely unmutated."""
    base_state = WorldState(
        resources=[Resource(id="inventory", current=100.0, min_value=0.0, max_value=200.0)],
    )
    parent_world = World(
        state=base_state,
        dynamics=StochasticDemandDynamics(
            resource_id="inventory", mean_demand=10.0, std_demand=2.0
        ),
    )
    parent_initial_fingerprint = parent_world.initial_state.fingerprint

    # Branch the world
    branched_world = branch_world(parent_world)

    scenario = Scenario(horizon=5, samples=1, seed=42)

    # Simulate ONLY on the branch
    res_branch = branched_world.simulate(scenario)
    assert res_branch is not None

    # Verify branch state has progressed
    branch_traj = res_branch.trajectories[0]
    assert branch_traj.final_state.step == 5
    assert branch_traj.final_state.get_resource("inventory").current < 100.0

    # Verify parent world state is COMPLETELY unmodified
    assert parent_world.initial_state.fingerprint == parent_initial_fingerprint
    assert parent_world.initial_state.step == 0
    assert parent_world.initial_state.get_resource("inventory").current == 100.0


@pytest.mark.property
def test_parallel_branches_do_not_contaminate_each_other() -> None:
    """AC-005: Divergent branches executing different policies maintain strict isolation."""
    base_state = WorldState(
        resources=[
            Resource(id="fuel", current=50.0, min_value=0.0, max_value=100.0),
            Resource(id="reserve", current=50.0, min_value=0.0, max_value=100.0),
        ],
    )
    world = World(
        state=base_state,
        dynamics=DeterministicTransferDynamics(action_type="transfer_resource"),
    )

    scen_a = Scenario(
        scenario_id="branch_burn",
        horizon=1,
        samples=1,
        seed=1,
        scheduled_actions=(
            ScheduledAction(
                step=0,
                action=Action(
                    id="t1",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "reserve",
                        "target_resource": "fuel",
                        "quantity": 30.0,
                    },
                ),
            ),
        ),
    )

    scen_b = Scenario(
        scenario_id="branch_save",
        horizon=1,
        samples=1,
        seed=1,
        scheduled_actions=(
            ScheduledAction(
                step=0,
                action=Action(
                    id="t2",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "fuel",
                        "target_resource": "reserve",
                        "quantity": 20.0,
                    },
                ),
            ),
        ),
    )

    world_a = branch_world(world)
    world_b = branch_world(world)

    res_a = world_a.simulate(scen_a)
    res_b = world_b.simulate(scen_b)

    # In Branch A: reserve dropped from 50 to 20, fuel increased from 50 to 80
    assert res_a.trajectories[0].final_state.get_resource("reserve").current == 20.0
    assert res_a.trajectories[0].final_state.get_resource("fuel").current == 80.0

    # In Branch B: fuel dropped from 50 to 30, reserve increased from 50 to 70
    assert res_b.trajectories[0].final_state.get_resource("fuel").current == 30.0
    assert res_b.trajectories[0].final_state.get_resource("reserve").current == 70.0

    # Original world untouched
    assert world.initial_state.get_resource("fuel").current == 50.0
    assert world.initial_state.get_resource("reserve").current == 50.0
