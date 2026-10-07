"""Core-only import and execution test asserting zero optional dependency leaks."""

from __future__ import annotations

import sys

import pytest


@pytest.mark.packaging
def test_root_import_does_not_load_optional_dependencies() -> None:
    """Importing ewm_engine root must NEVER load heavy optional libraries into sys.modules."""
    forbidden_heavy_modules = [
        "torch",
        "z3",
        "ortools",
        "ray",
        "gymnasium",
        "plotly",
        "fastapi",
        "mlflow",
        "salib",
        "simpy",
        "mesa",
        "pysd",
        "nashpy",
        "rtamt",
        "fmpy",
    ]

    import ewm_engine  # noqa: F401

    for mod in forbidden_heavy_modules:
        assert mod not in sys.modules, (
            f"Optional library '{mod}' was eagerly imported on 'import ewm_engine'!"
        )


@pytest.mark.packaging
def test_core_quickstart_runs_with_core_dependencies_alone() -> None:
    """The 5-minute quickstart workflow executes purely on core dependencies (pydantic/numpy/pyyaml)."""
    from ewm_engine import (
        Action,
        Entity,
        Relationship,
        Resource,
        Scenario,
        SimulationEngine,
        TrajectoryStatus,
        World,
        WorldState,
        compare_scenarios,
    )
    from ewm_engine.constraints.standard import ResourceCapacityConstraint
    from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
    from ewm_engine.simulation.scenario import ScheduledAction

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
    assert res_baseline.trajectories[0].status == TrajectoryStatus.COMPLETED

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
                step=0,
                action=Action(
                    id="transfer_0",
                    type="transfer",
                    parameters={
                        "source_resource": "stock_north",
                        "target_resource": "stock_south",
                        "amount": 30.0,
                    },
                ),
            ),
        ),
    )
    res_intervention = engine.run(world_alt, scenario_intervention)
    assert res_intervention.trajectories[0].status == TrajectoryStatus.COMPLETED

    # 5. Evaluate and Compare Scenarios
    comparison = compare_scenarios(res_baseline, [res_intervention])
    assert comparison is not None
    assert "Transfer 30 North -> South" in comparison.deltas
