"""Execution tests for documented runnable examples (AC-014 and AC-021)."""

from __future__ import annotations

import pytest

from ewm_engine import (
    Action,
    Entity,
    Relationship,
    Resource,
    Scenario,
    SimulationEngine,
    World,
    WorldState,
    compare_scenarios,
)
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.scenario import ScheduledAction
from examples.civicflow.run import run_civicflow_simulation
from examples.minimal_warehouse.run import run_minimal_warehouse
from examples.minimal_world.run import run_minimal_world


@pytest.mark.example
def test_readme_quickstart_executes() -> None:
    """AC-021: Ensure the exact 5-minute README Quickstart executes end-to-end and produces stated comparison."""
    # 1. Initialize World State S_0
    state = WorldState(
        entities=[
            Entity(id="wh_north", type="warehouse", attributes={"region": "north"}),
            Entity(id="wh_south", type="warehouse", attributes={"region": "south"}),
        ],
        relationships=[
            Relationship(source="wh_north", target="wh_south", type="connected_to"),
        ],
        resources=[
            Resource(id="stock_north", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_south", current=20.0, min_value=0.0, max_value=200.0),
        ],
    )

    # 2. Attach Constraints and Dynamics
    world = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[
            ResourceCapacityConstraint(resource_id="stock_north"),
            ResourceCapacityConstraint(resource_id="stock_south"),
        ],
    )

    # 3. Simulate Baseline Scenario (Status Quo)
    engine = SimulationEngine()
    scenario_baseline = Scenario(
        scenario_id="baseline",
        name="Status Quo",
        horizon=2,
        samples=1,
        seed=42,
    )
    res_baseline = engine.run(world, scenario_baseline)

    # 4. Branch and Simulate Scenario Intervention
    world_alt = world.branch()
    scenario_intervention = Scenario(
        scenario_id="intervention",
        name="Transfer 30 North -> South",
        horizon=2,
        samples=1,
        seed=42,
        scheduled_actions=(
            ScheduledAction(
                step=1,
                action=Action(
                    id="act_transfer",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "stock_north",
                        "target_resource": "stock_south",
                        "quantity": 30.0,
                    },
                ),
            ),
        ),
    )
    res_intervention = engine.run(world_alt, scenario_intervention)

    # 5. Evaluate and Compare Scenarios
    comparison = compare_scenarios(
        baseline=res_baseline,
        candidates=[res_intervention],
        metrics=["resource_stock_north", "resource_stock_south", "violations_count"],
    )

    # Assert correct trajectory progression
    traj_base = res_baseline.trajectories[0]
    traj_alt = res_intervention.trajectories[0]

    assert traj_base.final_state.get_resource("stock_north").current == 100.0
    assert traj_base.final_state.get_resource("stock_south").current == 20.0

    assert traj_alt.final_state.get_resource("stock_north").current == 70.0
    assert traj_alt.final_state.get_resource("stock_south").current == 50.0

    table = comparison.summary_table()
    assert "Status Quo" in table
    assert "Transfer 30 North -> South" in table
    assert "-30.00" in table
    assert "+30.00" in table


@pytest.mark.example
def test_readme_scenario_branching_executes() -> None:
    """AC-014: Ensure the documented scenario branching example executes without mutation."""
    state = WorldState(
        resources=[
            Resource(id="stock_main", current=50.0, min_value=0.0, max_value=100.0),
        ]
    )
    world = World(state=state, dynamics=DeterministicTransferDynamics())
    engine = SimulationEngine()

    policy_branch = world.branch()
    branch_scenario = Scenario(
        scenario_id="branch_policy_b",
        name="Aggressive Reorder",
        horizon=3,
        samples=2,
        seed=101,
    )

    branch_results = engine.run(policy_branch, branch_scenario)
    assert len(branch_results.trajectories) == 2
    assert world.initial_state.fingerprint == state.fingerprint


@pytest.mark.example
def test_readme_constraint_rejection_executes() -> None:
    """AC-014: Ensure documented constraint pre-action rejection snippet executes properly."""
    state = WorldState(
        resources=[
            Resource(id="stock_a", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_b", current=20.0, min_value=0.0, max_value=200.0),
        ]
    )
    world = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[ActionTransferAvailabilityConstraint()],
    )

    invalid_action_scenario = Scenario(
        scenario_id="reject_excess",
        horizon=1,
        scheduled_actions=(
            ScheduledAction(
                step=0,
                action=Action(
                    id="act_overflow",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "stock_a",
                        "target_resource": "stock_b",
                        "quantity": 150.0,
                    },
                ),
            ),
        ),
    )

    res = SimulationEngine().run(world, invalid_action_scenario)
    step = res.trajectories[0].steps[0]

    assert len(step.actions_proposed) == 1
    assert len(step.actions_accepted) == 0
    assert res.trajectories[0].final_state.get_resource("stock_a").current == 100.0


@pytest.mark.example
def test_readme_systemic_trace_executes() -> None:
    """AC-014: Ensure documented systemic trace export snippet executes properly."""
    state = WorldState(
        resources=[
            Resource(id="stock_north", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="stock_south", current=20.0, min_value=0.0, max_value=200.0),
        ]
    )
    world = World(state=state, dynamics=DeterministicTransferDynamics())
    scenario = Scenario(
        scenario_id="trace_scen",
        horizon=2,
        scheduled_actions=(
            ScheduledAction(
                step=1,
                action=Action(
                    id="act_transfer",
                    type="transfer_resource",
                    parameters={
                        "source_resource": "stock_north",
                        "target_resource": "stock_south",
                        "quantity": 30.0,
                    },
                ),
            ),
        ),
    )

    res = SimulationEngine().run(world, scenario)
    trajectory = res.trajectories[0]
    trace = trajectory.systemic_trace

    mermaid_markup = trace.to_mermaid()
    assert "flowchart TD" in mermaid_markup
    assert "transfer_resource" in mermaid_markup
    assert "drives" in mermaid_markup
    assert "causes" not in mermaid_markup

    nx_graph = trace.to_networkx()
    assert nx_graph.number_of_nodes() >= 2
    assert nx_graph.number_of_edges() >= 1


@pytest.mark.example
def test_minimal_warehouse_run_script_executes() -> None:
    """Ensure minimal_warehouse run.py script executes end-to-end and returns comparison."""
    comparison = run_minimal_warehouse()
    assert comparison is not None
    assert comparison.baseline_name is not None
    assert len(comparison.results_by_scenario) >= 2
    table = comparison.summary_table()
    assert "served_demand" in table
    assert "unserved_demand" in table


@pytest.mark.example
def test_civicflow_run_script_executes() -> None:
    """Ensure civicflow run.py script executes end-to-end and returns comparison."""
    comparison = run_civicflow_simulation()
    assert comparison is not None
    assert comparison.baseline_name is not None
    assert len(comparison.results_by_scenario) >= 2
    table = comparison.summary_table()
    assert "cumulative_unserved" in table or "unserved" in table


@pytest.mark.example
def test_minimal_world_backward_compatibility() -> None:
    """Ensure minimal_world example continues to execute for backward compatibility."""
    comparison = run_minimal_world()
    assert comparison is not None
    assert comparison.baseline_name is not None
