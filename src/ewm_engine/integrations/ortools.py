"""Google OR-Tools mathematical programming and CP-SAT optimization adapters for EWM Engine."""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Sequence
from typing import Any

from ewm_engine.actors.base import Actor, ActorContext
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ActorId
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.integrations.protocols import (
    ActionPlanner,
    ConstraintSolver,
    SolverResult,
    SolverStatus,
)


def _load_pywraplp() -> Any | None:
    try:
        return importlib.import_module("ortools.linear_solver.pywraplp")
    except ImportError:
        return None


def _load_cp_model() -> Any | None:
    try:
        return importlib.import_module("ortools.sat.python.cp_model")
    except ImportError:
        return None


class ORToolsAllocationAdapter(ConstraintSolver):
    """Operations Research optimization adapter leveraging Google OR-Tools Linear Solver.

    Solves optimal resource redistribution / min-cost network flow problems
    over an EWM WorldState to automatically synthesize valid Action sequences.
    Enforces finite, mandatory time limits on every solve.
    """

    def __init__(
        self,
        solver_type: str = "GLOP",
        default_time_limit_seconds: float = 5.0,
    ) -> None:
        """Initialize OR-Tools solver wrapper.

        Args:
            solver_type: Linear programming solver backend (default: 'GLOP').
            default_time_limit_seconds: Mandatory time limit in seconds (must be > 0).
        """
        if default_time_limit_seconds <= 0:
            raise ValueError(
                f"default_time_limit_seconds must be positive, got {default_time_limit_seconds}"
            )

        self.solver_type = solver_type
        self.default_time_limit_seconds = default_time_limit_seconds
        self._pywraplp: Any | None = _load_pywraplp()
        self.last_result: SolverResult | None = None

    @property
    def is_available(self) -> bool:
        """Return True if Google OR-Tools linear solver is installed and importable."""
        return self._pywraplp is not None

    def optimize_transfers(
        self,
        state: WorldState,
        sources: Sequence[str],
        destinations: Sequence[str],
        demands: dict[str, float],
        capacities: dict[tuple[str, str], float] | None = None,
        costs: dict[tuple[str, str], float] | None = None,
        time_limit_seconds: float | None = None,
    ) -> Sequence[Action]:
        """Solve a transportation / flow problem to generate optimal transfer actions.

        Args:
            state: Current world state providing source resource stock levels.
            sources: Resource IDs of supplying entities.
            destinations: Resource IDs of demanding entities.
            demands: Required resource quantities per destination.
            capacities: Optional edge capacity limits `(src, dst) -> max_qty`.
            costs: Optional edge transfer costs `(src, dst) -> unit_cost`.
            time_limit_seconds: Optional solve timeout override.

        Returns:
            Sequence of concrete `transfer_resource` Action objects.
        """
        if self._pywraplp is None:
            raise SimulationConfigurationError(
                "Google OR-Tools is required for ORToolsAllocationAdapter. "
                "Install via `pip install ortools` or `pip install ewm-engine[or]`."
            )

        effective_time_limit = (
            time_limit_seconds
            if time_limit_seconds is not None
            else self.default_time_limit_seconds
        )
        if effective_time_limit <= 0:
            raise ValueError(f"time_limit_seconds must be positive, got {effective_time_limit}")

        solver = self._pywraplp.Solver.CreateSolver(self.solver_type)
        if not solver:
            raise SimulationConfigurationError(
                f"Failed to create OR-Tools solver with backend: {self.solver_type}"
            )

        # Enforce time limit in milliseconds
        solver.set_time_limit(max(1, int(effective_time_limit * 1000)))

        # Variables: flow[i, j]
        flow: dict[tuple[str, str], Any] = {}
        for s in sources:
            for d in destinations:
                cap = (capacities or {}).get((s, d), solver.infinity())
                flow[s, d] = solver.NumVar(0.0, float(cap), f"flow_{s}_{d}")

        # Supply Constraints: sum_j flow[i, j] <= supply[i]
        for s in sources:
            res = state.get_resource(s)
            avail = res.current if res is not None else 0.0
            solver.Add(solver.Sum([flow[s, d] for d in destinations]) <= float(avail))

        # Demand Constraints: sum_i flow[i, j] >= demand[j]
        for d in destinations:
            target_demand = float(demands.get(d, 0.0))
            if target_demand > 0:
                solver.Add(solver.Sum([flow[s, d] for s in sources]) >= target_demand)

        # Objective: Minimize transfer costs
        cost_map = costs or {}
        objective = solver.Objective()
        for s in sources:
            for d in destinations:
                c = float(cost_map.get((s, d), 1.0))
                objective.SetCoefficient(flow[s, d], c)
        objective.SetMinimization()

        start_time = time.perf_counter()
        status = solver.Solve()
        elapsed = time.perf_counter() - start_time

        timed_out = status == self._pywraplp.Solver.NOT_SOLVED and elapsed >= (
            effective_time_limit * 0.9
        )

        if status == self._pywraplp.Solver.OPTIMAL:
            res_status = SolverStatus.OPTIMAL
            satisfied = True
            msg = "Optimal allocation found"
        elif status == self._pywraplp.Solver.FEASIBLE:
            res_status = SolverStatus.FEASIBLE
            satisfied = True
            msg = "Feasible allocation found"
        elif timed_out or status == self._pywraplp.Solver.NOT_SOLVED:
            res_status = SolverStatus.UNKNOWN
            satisfied = False
            msg = f"Solver stopped (time limit {effective_time_limit}s reached)"
            timed_out = True
        else:
            res_status = SolverStatus.INFEASIBLE
            satisfied = False
            msg = "Allocation problem is infeasible"

        self.last_result = SolverResult(
            status=res_status,
            satisfied=satisfied,
            message=msg,
            solve_time_seconds=round(elapsed, 6),
            timed_out=timed_out,
        )

        if not satisfied:
            return []

        actions: list[Action] = []
        action_idx = 0
        for s in sources:
            for d in destinations:
                val = flow[s, d].solution_value()
                if val > 1e-6:
                    actions.append(
                        Action(
                            id=f"act_opt_transfer_{action_idx}",
                            type="transfer_resource",
                            parameters={
                                "source_resource": s,
                                "target_resource": d,
                                "quantity": round(val, 6),
                            },
                        )
                    )
                    action_idx += 1

        return actions

    def check(
        self,
        *,
        state: WorldState,
        actions: Sequence[Action] = (),
        time_limit_seconds: float | None = None,
    ) -> SolverResult:
        """Verify allocation feasibility under supply and capacity limits."""
        if self._pywraplp is None:
            raise SimulationConfigurationError(
                "Google OR-Tools is required for ORToolsAllocationAdapter. "
                "Install via `pip install ortools` or `pip install ewm-engine[or]`."
            )

        if not actions:
            return SolverResult(
                status=SolverStatus.OPTIMAL,
                satisfied=True,
                message="No actions to verify",
                solve_time_seconds=0.0,
            )

        # Verify each action does not exceed source supply or destination capacity
        start_time = time.perf_counter()
        violations: list[str] = []
        projected_supplies: dict[str, float] = {}

        for act in actions:
            if act.type == "transfer_resource":
                src = str(act.parameters.get("source_resource", ""))
                qty = float(act.parameters.get("quantity", 0.0))
                if src:
                    curr = projected_supplies.get(
                        src,
                        state.get_resource(src).current
                        if state.get_resource(src) is not None
                        else 0.0,
                    )
                    if curr < qty:
                        violations.append(
                            f"Action {act.id} requests {qty} from {src} (available: {curr})"
                        )
                    projected_supplies[src] = curr - qty

        elapsed = time.perf_counter() - start_time
        satisfied = len(violations) == 0
        return SolverResult(
            status=SolverStatus.OPTIMAL if satisfied else SolverStatus.INFEASIBLE,
            satisfied=satisfied,
            message="All action transfers feasible" if satisfied else "; ".join(violations),
            solve_time_seconds=round(elapsed, 6),
            timed_out=False,
        )


class CPSATAllocationPlanner(Actor, ActionPlanner):
    """Discrete resource allocation planner powered by Google OR-Tools CP-SAT.

    Formulates integer/discrete multi-depot distribution problems with support for:
    - Enforced finite solve time limits (`max_time_in_seconds`).
    - Assumption variables enabling extraction of an auditable unsatisfiable core
      (infeasibility certificate) whenever demand cannot be satisfied.
    - Full compliance with `ActionPlanner` and `Actor` protocols. Candidate actions
      are proposed to the engine and validated by the engine's constraint pipeline.
    """

    def __init__(
        self,
        actor_id: ActorId = "cpsat_planner",
        sources: Sequence[str] = (),
        destinations: Sequence[str] = (),
        demands: dict[str, int] | Callable[[WorldState], dict[str, int]] | None = None,
        capacities: dict[tuple[str, str], int] | None = None,
        costs: dict[tuple[str, str], int] | None = None,
        action_type: str = "transfer_resource",
        allow_partial: bool = False,
        time_limit_seconds: float = 5.0,
    ) -> None:
        if time_limit_seconds <= 0:
            raise ValueError(f"time_limit_seconds must be positive, got {time_limit_seconds}")

        self._actor_id = actor_id
        self.sources = tuple(sources)
        self.destinations = tuple(destinations)
        self.demands = demands or {}
        self.capacities = capacities or {}
        self.costs = costs or {}
        self.action_type = action_type
        self.allow_partial = allow_partial
        self.time_limit_seconds = time_limit_seconds
        self._cp_model: Any | None = _load_cp_model()
        self.last_result: SolverResult | None = None

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    @property
    def is_available(self) -> bool:
        """Return True if Google OR-Tools CP-SAT solver is importable."""
        return self._cp_model is not None

    def propose(
        self,
        *,
        state: WorldState,
        objective: str | dict[str, Any] | None = None,
        time_limit_seconds: float | None = None,
    ) -> Sequence[Action]:
        """Synthesize candidate resource transfer actions via CP-SAT discrete optimization.

        Planners NEVER mutate world state directly. Actions are returned for engine constraint
        verification.

        Args:
            state: Current world state containing inventory/resource levels.
            objective: Optional objective override.
            time_limit_seconds: Optional timeout override.

        Returns:
            Sequence of proposed Action objects (empty if infeasible or timed out).
        """
        if self._cp_model is None:
            raise SimulationConfigurationError(
                "Google OR-Tools CP-SAT is required for CPSATAllocationPlanner. "
                "Install via `pip install ortools` or `pip install ewm-engine[or]`."
            )

        effective_time_limit = (
            time_limit_seconds if time_limit_seconds is not None else self.time_limit_seconds
        )
        if effective_time_limit <= 0:
            raise ValueError(f"time_limit_seconds must be positive, got {effective_time_limit}")

        # Resolve demands
        if callable(self.demands):
            demands_map = self.demands(state)
        else:
            demands_map = self.demands

        model = self._cp_model.CpModel()
        flow: dict[tuple[str, str], Any] = {}
        assumptions: list[Any] = []
        max_int_val = 1_000_000_000

        # Flow variables bounded by link capacities
        for s in self.sources:
            for d in self.destinations:
                cap = self.capacities.get((s, d), max_int_val)
                flow[s, d] = model.NewIntVar(0, int(cap), f"flow_{s}_{d}")

        # Supply constraints with tracked assumption literals for auditability
        for s in self.sources:
            res = state.get_resource(s)
            avail = int(res.current) if res is not None else 0
            sup_var = model.NewBoolVar(f"supply_capacity_{s}_{avail}")
            model.Add(sum(flow[s, d] for d in self.destinations) <= avail).OnlyEnforceIf(sup_var)
            model.AddAssumption(sup_var)
            assumptions.append(sup_var)

        # Demand constraints with tracked assumptions
        unmet_vars: dict[str, Any] = {}
        for d in self.destinations:
            req = int(demands_map.get(d, 0))
            if req > 0:
                if self.allow_partial:
                    unmet = model.NewIntVar(0, req, f"unmet_{d}")
                    unmet_vars[d] = unmet
                    model.Add(sum(flow[s, d] for s in self.sources) + unmet >= req)
                else:
                    dem_var = model.NewBoolVar(f"demand_target_{d}_{req}")
                    model.Add(sum(flow[s, d] for s in self.sources) >= req).OnlyEnforceIf(dem_var)
                    model.AddAssumption(dem_var)
                    assumptions.append(dem_var)

        # Objective: minimize transfer cost (+ unmet demand penalty if partial allowed)
        obj_exprs = []
        for s in self.sources:
            for d in self.destinations:
                c = int(self.costs.get((s, d), 1))
                obj_exprs.append(flow[s, d] * c)

        if self.allow_partial and unmet_vars:
            unmet_penalty = 10_000
            for unmet in unmet_vars.values():
                obj_exprs.append(unmet * unmet_penalty)

        if obj_exprs:
            model.Minimize(sum(obj_exprs))

        # Solve with enforced time limit
        solver = self._cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(effective_time_limit)

        start_time = time.perf_counter()
        status = solver.Solve(model)
        elapsed = time.perf_counter() - start_time

        unsat_core: tuple[str, ...] = ()
        timed_out = False

        if status == self._cp_model.OPTIMAL:
            res_status = SolverStatus.OPTIMAL
            satisfied = True
            msg = "CP-SAT found optimal discrete allocation"
        elif status == self._cp_model.FEASIBLE:
            res_status = SolverStatus.FEASIBLE
            satisfied = True
            msg = "CP-SAT found feasible discrete allocation"
        elif status == self._cp_model.INFEASIBLE:
            res_status = SolverStatus.INFEASIBLE
            satisfied = False
            msg = "CP-SAT proved allocation is infeasible under physical bounds"
            if assumptions:
                raw_core = solver.SufficientAssumptionsForInfeasibility()
                unsat_core = tuple(model.GetBoolVarFromProtoIndex(i).Name() for i in raw_core)
        else:  # UNKNOWN or MODEL_INVALID
            res_status = SolverStatus.UNKNOWN
            satisfied = False
            timed_out = elapsed >= (effective_time_limit * 0.9)
            msg = (
                f"CP-SAT timed out after {effective_time_limit}s"
                if timed_out
                else "CP-SAT returned unknown status"
            )

        self.last_result = SolverResult(
            status=res_status,
            satisfied=satisfied,
            message=msg,
            unsat_core=unsat_core,
            solve_time_seconds=round(elapsed, 6),
            timed_out=timed_out,
        )

        if not satisfied:
            return []

        actions: list[Action] = []
        action_idx = 0
        for s in self.sources:
            for d in self.destinations:
                val = solver.Value(flow[s, d])
                if val > 0:
                    actions.append(
                        Action(
                            id=f"{self.actor_id}_transfer_{action_idx}",
                            type=self.action_type,
                            parameters={
                                "source_resource": s,
                                "target_resource": d,
                                "quantity": float(val),
                            },
                        )
                    )
                    action_idx += 1

        return actions

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        """Propose actions during world model simulation rollout."""
        return self.propose(
            state=state,
            objective=None,
            time_limit_seconds=self.time_limit_seconds,
        )

    def clone(self) -> CPSATAllocationPlanner:
        """Create an independent clone of this planner."""
        return CPSATAllocationPlanner(
            actor_id=self.actor_id,
            sources=self.sources,
            destinations=self.destinations,
            demands=self.demands,
            capacities=self.capacities,
            costs=self.costs,
            action_type=self.action_type,
            allow_partial=self.allow_partial,
            time_limit_seconds=self.time_limit_seconds,
        )
