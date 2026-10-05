"""Integration tests for mathematical solvers and OR action planners.

Verifies:
- Protocol compliance (ConstraintSolver, ActionPlanner, Actor).
- Mandatory finite time limit enforcement (no unbounded solver path).
- Unsat core / infeasibility certificate extraction in Z3 and OR-Tools CP-SAT.
- Propose -> Constraint validation -> State transition pipeline without direct state mutation.
"""

from __future__ import annotations

import pytest

from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.constraints.standard import ActionTransferAvailabilityConstraint
from ewm_engine.core.actions import Action
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.integrations.agents import CallableActorAdapter
from ewm_engine.integrations.ortools import (
    CPSATAllocationPlanner,
    ORToolsAllocationAdapter,
)
from ewm_engine.integrations.protocols import (
    ActionPlanner,
    ConstraintSolver,
    SolverStatus,
)
from ewm_engine.integrations.scipy_planner import SciPyAllocationPlanner
from ewm_engine.integrations.solvers import Z3ConstraintAdapter
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario

# ---------------------------------------------------------------------------
# Z3 SMT Constraint Adapter Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_z3_protocol_and_satisfaction() -> None:
    """Verify Z3ConstraintAdapter satisfies both Constraint and ConstraintSolver protocols."""

    def verify_positive_cash(
        state: WorldState,
        action: Action | None,
        z3_mod: object,
        timeout_ms: int = 5000,
    ) -> tuple[bool, str, dict[str, object]]:
        import z3

        s = z3.Solver()
        s.set("timeout", timeout_ms)
        cash_val = state.get_resource("cash").current
        x = z3.Real("cash")
        s.add(x == cash_val)
        s.add(x < 0)  # Check if cash can be negative

        res = s.check()
        if res == z3.sat:
            return False, "Negative cash detected", {"deficit": True}
        return True, "Cash invariant holds", {}

    adapter = Z3ConstraintAdapter(
        constraint_id="positive_cash",
        solver_fn=verify_positive_cash,
        severity=ConstraintSeverity.HARD,
        timeout_ms=3000,
    )

    assert isinstance(adapter, ConstraintSolver)
    assert adapter.timeout_ms == 3000

    valid_state = WorldState(
        resources=[Resource(id="cash", current=100.0, min_value=0.0, max_value=500.0)]
    )
    result = adapter.check(state=valid_state)
    assert result.status == SolverStatus.OPTIMAL
    assert result.satisfied
    assert not result.timed_out
    assert result.solve_time_seconds >= 0.0

    eval_result = adapter.evaluate(valid_state)
    assert eval_result.satisfied


@pytest.mark.integration
def test_z3_unsat_core_extraction() -> None:
    """Verify Z3ConstraintAdapter surfaces unsatisfiable core when constraints conflict."""

    def unsatisfiable_budget_solver(
        state: WorldState,
        action: Action | None,
        z3_mod: object,
        timeout_ms: int = 5000,
    ) -> tuple[bool, str, dict[str, object]]:
        import z3

        s = z3.Solver()
        s.set("timeout", timeout_ms)
        p1 = z3.Bool("min_cash_required")
        p2 = z3.Bool("max_cash_limit")
        x = z3.Real("cash")

        s.assert_and_track(x >= 100.0, p1)
        s.assert_and_track(x <= 50.0, p2)

        check_res = s.check()
        if check_res == z3.unsat:
            core = [str(c) for c in s.unsat_core()]
            return False, "Conflicting budget boundaries", {"unsat_core": core}
        return True, "Consistent", {}

    adapter = Z3ConstraintAdapter(
        constraint_id="budget_core",
        solver_fn=unsatisfiable_budget_solver,
    )

    state = WorldState(
        resources=[Resource(id="cash", current=10.0, min_value=0.0, max_value=500.0)]
    )
    result = adapter.check(state=state)
    assert result.status == SolverStatus.INFEASIBLE
    assert not result.satisfied
    assert "min_cash_required" in result.unsat_core
    assert "max_cash_limit" in result.unsat_core

    eval_result = adapter.evaluate(state)
    assert not eval_result.satisfied
    assert "unsat_core" in eval_result.violating_values


@pytest.mark.integration
def test_z3_timeout_enforcement() -> None:
    """Verify Z3ConstraintAdapter enforces finite timeout rather than hanging indefinitely."""

    def slow_combinatorial_solver(
        state: WorldState,
        action: Action | None,
        z3_mod: object,
        timeout_ms: int = 1,
    ) -> tuple[bool, str, dict[str, object]]:
        import z3

        s = z3.Solver()
        s.set("timeout", timeout_ms)
        # Create a hard pigeonhole problem that times out under 1ms
        n_pigeons = 12
        n_holes = 11
        vars_map = {(i, j): z3.Bool(f"p_{i}_{j}") for i in range(n_pigeons) for j in range(n_holes)}
        for i in range(n_pigeons):
            s.add(z3.Or([vars_map[(i, j)] for j in range(n_holes)]))
        for j in range(n_holes):
            for i1 in range(n_pigeons):
                for i2 in range(i1 + 1, n_pigeons):
                    s.add(z3.Or(z3.Not(vars_map[(i1, j)]), z3.Not(vars_map[(i2, j)])))

        check_res = s.check()
        if check_res == z3.unknown and s.reason_unknown() == "timeout":
            return False, "Solver timed out", {"timed_out": True, "reason": "timeout"}
        return check_res == z3.sat, "Done", {}

    adapter = Z3ConstraintAdapter(
        constraint_id="slow_problem",
        solver_fn=slow_combinatorial_solver,
        timeout_ms=1,  # 1ms timeout
    )

    state = WorldState(resources=[])
    result = adapter.check(state=state, time_limit_seconds=0.001)
    assert result.timed_out
    assert result.status == SolverStatus.UNKNOWN
    assert not result.satisfied


@pytest.mark.integration
def test_z3_invalid_timeout_raises() -> None:
    """Verify negative or zero timeout raises ValueError."""
    with pytest.raises(ValueError, match="timeout_ms must be positive"):
        Z3ConstraintAdapter(
            constraint_id="inv",
            solver_fn=lambda s, a, z: (True, "", {}),
            timeout_ms=0,
        )


# ---------------------------------------------------------------------------
# OR-Tools Allocation Adapter Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ortools_allocation_adapter_optimal_solve() -> None:
    """Verify ORToolsAllocationAdapter synthesizes optimal transfer actions."""
    adapter = ORToolsAllocationAdapter(default_time_limit_seconds=5.0)
    assert isinstance(adapter, ConstraintSolver)

    state = WorldState(
        resources=[
            Resource(id="wh_a", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="wh_b", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="store_1", current=0.0, min_value=0.0, max_value=200.0),
            Resource(id="store_2", current=0.0, min_value=0.0, max_value=200.0),
        ]
    )

    actions = adapter.optimize_transfers(
        state=state,
        sources=["wh_a", "wh_b"],
        destinations=["store_1", "store_2"],
        demands={"store_1": 40.0, "store_2": 30.0},
        costs={("wh_a", "store_1"): 1.0, ("wh_b", "store_2"): 1.0},
    )

    assert len(actions) == 2
    total_qty = sum(float(a.parameters["quantity"]) for a in actions)
    assert total_qty == pytest.approx(70.0)
    assert adapter.last_result is not None
    assert adapter.last_result.status == SolverStatus.OPTIMAL


@pytest.mark.integration
def test_ortools_allocation_adapter_check_protocol() -> None:
    """Verify ORToolsAllocationAdapter.check() validates candidate action feasibility."""
    adapter = ORToolsAllocationAdapter()
    state = WorldState(
        resources=[Resource(id="stock", current=50.0, min_value=0.0, max_value=100.0)]
    )

    valid_actions = [
        Action(
            id="act_valid",
            type="transfer_resource",
            parameters={"source_resource": "stock", "quantity": 30.0},
        )
    ]
    res_valid = adapter.check(state=state, actions=valid_actions)
    assert res_valid.satisfied
    assert res_valid.status == SolverStatus.OPTIMAL

    excess_actions = [
        Action(
            id="act_excess",
            type="transfer_resource",
            parameters={"source_resource": "stock", "quantity": 80.0},
        )
    ]
    res_invalid = adapter.check(state=state, actions=excess_actions)
    assert not res_invalid.satisfied
    assert res_invalid.status == SolverStatus.INFEASIBLE


# ---------------------------------------------------------------------------
# CP-SAT Allocation Planner Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cpsat_planner_protocol_and_solve() -> None:
    """Verify CPSATAllocationPlanner conforms to ActionPlanner and Actor protocols."""
    planner = CPSATAllocationPlanner(
        actor_id="cpsat_logistics",
        sources=["depot_1", "depot_2"],
        destinations=["hosp_a", "hosp_b"],
        demands={"hosp_a": 50, "hosp_b": 40},
        capacities={("depot_1", "hosp_a"): 60, ("depot_2", "hosp_b"): 50},
        costs={("depot_1", "hosp_a"): 2, ("depot_2", "hosp_b"): 3},
        time_limit_seconds=5.0,
    )

    assert isinstance(planner, ActionPlanner)
    assert hasattr(planner, "act")
    assert planner.actor_id == "cpsat_logistics"

    state = WorldState(
        resources=[
            Resource(id="depot_1", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="depot_2", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="hosp_a", current=0.0, min_value=0.0, max_value=100.0),
            Resource(id="hosp_b", current=0.0, min_value=0.0, max_value=100.0),
        ]
    )

    actions = planner.propose(state=state)
    assert len(actions) == 2
    assert planner.last_result is not None
    assert planner.last_result.status == SolverStatus.OPTIMAL
    assert not planner.last_result.timed_out
    assert sum(float(a.parameters["quantity"]) for a in actions) == 90.0


@pytest.mark.integration
def test_cpsat_planner_unsat_core_infeasibility_certificate() -> None:
    """Verify CP-SAT planner extracts unsat core when physical supply cannot meet demand."""
    planner = CPSATAllocationPlanner(
        actor_id="cpsat_infeasible",
        sources=["depot_empty"],
        destinations=["hosp_urgent"],
        demands={"hosp_urgent": 200},  # Requires 200 units
        time_limit_seconds=5.0,
    )

    # Depot only has 50 units
    state = WorldState(
        resources=[
            Resource(id="depot_empty", current=50.0, min_value=0.0, max_value=100.0),
            Resource(id="hosp_urgent", current=0.0, min_value=0.0, max_value=500.0),
        ]
    )

    actions = planner.propose(state=state)
    assert actions == []
    assert planner.last_result is not None
    assert planner.last_result.status == SolverStatus.INFEASIBLE
    assert not planner.last_result.satisfied
    assert len(planner.last_result.unsat_core) > 0
    # Core identifies the demand and supply conflict
    core_text = " ".join(planner.last_result.unsat_core)
    assert "demand_target_hosp_urgent_200" in core_text
    assert "supply_capacity_depot_empty_50" in core_text


@pytest.mark.integration
def test_cpsat_planner_in_simulation_engine_rollout() -> None:
    """Verify CPSATAllocationPlanner acts as an interventional Actor in SimulationEngine."""
    planner = CPSATAllocationPlanner(
        actor_id="cpsat_actor",
        sources=["wh_main"],
        destinations=["clinic"],
        demands={"clinic": 25},
        time_limit_seconds=5.0,
    )

    state = WorldState(
        resources=[
            Resource(id="wh_main", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="clinic", current=0.0, min_value=0.0, max_value=100.0),
        ]
    )

    world = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[ActionTransferAvailabilityConstraint()],
        actors=[planner],
    )

    scenario = Scenario(
        scenario_id="cpsat_rollout",
        horizon=2,
        samples=1,
        seed=42,
    )

    result = SimulationEngine().run(world, scenario)
    traj = result.trajectories[0]
    assert traj.status.name == "COMPLETED"
    # Action proposed by planner was validated and executed
    step0 = traj.steps[0]
    assert len(step0.actions_proposed) == 1
    assert len(step0.actions_accepted) == 1
    assert traj.final_state.get_resource("clinic").current >= 25.0


# ---------------------------------------------------------------------------
# SciPy Continuous Allocation Planner Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_scipy_planner_protocol_and_solve() -> None:
    """Verify SciPyAllocationPlanner continuous linear programming allocation."""
    planner = SciPyAllocationPlanner(
        actor_id="scipy_continuous",
        sources=["hub_a", "hub_b"],
        destinations=["region_1", "region_2"],
        demands={"region_1": 45.5, "region_2": 35.0},
        costs={("hub_a", "region_1"): 1.5, ("hub_b", "region_2"): 1.2},
        time_limit_seconds=5.0,
    )

    assert isinstance(planner, ActionPlanner)
    assert hasattr(planner, "act")

    state = WorldState(
        resources=[
            Resource(id="hub_a", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="hub_b", current=100.0, min_value=0.0, max_value=200.0),
            Resource(id="region_1", current=0.0, min_value=0.0, max_value=100.0),
            Resource(id="region_2", current=0.0, min_value=0.0, max_value=100.0),
        ]
    )

    actions = planner.propose(state=state)
    assert len(actions) >= 2
    assert planner.last_result is not None
    assert planner.last_result.status == SolverStatus.OPTIMAL
    total_flow = sum(float(a.parameters["quantity"]) for a in actions)
    assert total_flow == pytest.approx(80.5, abs=1e-3)


@pytest.mark.integration
def test_scipy_planner_infeasible() -> None:
    """Verify SciPyAllocationPlanner handles infeasible problems cleanly."""
    planner = SciPyAllocationPlanner(
        actor_id="scipy_infeasible",
        sources=["low_hub"],
        destinations=["high_demand"],
        demands={"high_demand": 500.0},
        time_limit_seconds=5.0,
    )

    state = WorldState(
        resources=[
            Resource(id="low_hub", current=50.0, min_value=0.0, max_value=100.0),
            Resource(id="high_demand", current=0.0, min_value=0.0, max_value=1000.0),
        ]
    )

    actions = planner.propose(state=state)
    assert actions == []
    assert planner.last_result is not None
    assert planner.last_result.status == SolverStatus.INFEASIBLE
    assert not planner.last_result.satisfied


@pytest.mark.integration
def test_scipy_planner_time_limit_rejection() -> None:
    """Verify SciPyAllocationPlanner enforces positive time limits."""
    with pytest.raises(ValueError, match="time_limit_seconds must be positive"):
        SciPyAllocationPlanner(time_limit_seconds=0.0)


# ---------------------------------------------------------------------------
# CallableActorAdapter as ActionPlanner Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_callable_actor_adapter_as_action_planner() -> None:
    """Verify CallableActorAdapter implements ActionPlanner protocol."""

    def custom_policy(state: WorldState, ctx: object) -> list[Action]:
        return [
            Action(
                id="act_custom",
                type="transfer_resource",
                parameters={"source_resource": "s", "quantity": 10.0},
            )
        ]

    adapter = CallableActorAdapter(actor_id="custom_agent", act_fn=custom_policy)
    assert isinstance(adapter, ActionPlanner)

    state = WorldState(resources=[])
    actions = adapter.propose(state=state)
    assert len(actions) == 1
    assert actions[0].id == "act_custom"
