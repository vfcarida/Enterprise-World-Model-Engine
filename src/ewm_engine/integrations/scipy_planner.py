"""SciPy continuous linear programming allocation planner for EWM Engine."""

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
    SolverResult,
    SolverStatus,
)


def _load_scipy_optimize() -> Any | None:
    try:
        return importlib.import_module("scipy.optimize")
    except ImportError:
        return None


class SciPyAllocationPlanner(Actor, ActionPlanner):
    """Continuous resource allocation planner powered by SciPy linear programming (HiGHS).

    Solves continuous min-cost network flow and transportation problems over an EWM WorldState
    to propose concrete Action sequences. Enforces mandatory finite solve time limits.
    Candidate actions are validated through the engine's constraint pipeline.
    """

    def __init__(
        self,
        actor_id: ActorId = "scipy_planner",
        sources: Sequence[str] = (),
        destinations: Sequence[str] = (),
        demands: dict[str, float] | Callable[[WorldState], dict[str, float]] | None = None,
        capacities: dict[tuple[str, str], float] | None = None,
        costs: dict[tuple[str, str], float] | None = None,
        action_type: str = "transfer_resource",
        time_limit_seconds: float = 5.0,
        method: str = "highs",
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
        self.time_limit_seconds = time_limit_seconds
        self.method = method
        self._scipy_optimize: Any | None = _load_scipy_optimize()
        self.last_result: SolverResult | None = None

    @property
    def actor_id(self) -> ActorId:
        return self._actor_id

    @property
    def is_available(self) -> bool:
        """Return True if SciPy optimize is installed and importable."""
        return self._scipy_optimize is not None

    def propose(
        self,
        *,
        state: WorldState,
        objective: str | dict[str, Any] | None = None,
        time_limit_seconds: float | None = None,
    ) -> Sequence[Action]:
        """Propose continuous resource transfers via SciPy linear programming.

        Planners NEVER mutate world state directly. Actions are returned for engine constraint
        verification.

        Args:
            state: Current world state containing inventory/resource levels.
            objective: Optional objective specification or parameters.
            time_limit_seconds: Optional timeout override in seconds.

        Returns:
            Sequence of proposed Action objects (empty if infeasible or timed out).
        """
        if self._scipy_optimize is None:
            raise SimulationConfigurationError(
                "SciPy is required for SciPyAllocationPlanner. "
                "Install via `pip install scipy` or `pip install ewm-engine[or]`."
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

        pairs = [(s, d) for s in self.sources for d in self.destinations]
        if not pairs:
            self.last_result = SolverResult(
                status=SolverStatus.OPTIMAL,
                satisfied=True,
                message="No source-destination pairs configured",
                solve_time_seconds=0.0,
            )
            return []

        n_vars = len(pairs)
        pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}

        # Objective cost coefficients
        c = [float(self.costs.get(pair, 1.0)) for pair in pairs]

        # Variable bounds (0 <= x <= capacity)
        bounds = []
        for pair in pairs:
            cap = self.capacities.get(pair, None)
            bounds.append((0.0, float(cap) if cap is not None else None))

        A_ub_rows: list[list[float]] = []
        b_ub_rows: list[float] = []

        # Supply constraints: sum_d flow(s, d) <= avail(s)
        for s in self.sources:
            res = state.get_resource(s)
            avail = float(res.current) if res is not None else 0.0
            row = [0.0] * n_vars
            for d in self.destinations:
                row[pair_to_idx[(s, d)]] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(avail)

        # Demand constraints: sum_s flow(s, d) >= demand(d) -> -sum_s flow(s, d) <= -demand(d)
        for d in self.destinations:
            req = float(demands_map.get(d, 0.0))
            if req > 0:
                row = [0.0] * n_vars
                for s in self.sources:
                    row[pair_to_idx[(s, d)]] = -1.0
                A_ub_rows.append(row)
                b_ub_rows.append(-req)

        opts: dict[str, Any] = {"time_limit": float(effective_time_limit)}

        start_time = time.perf_counter()
        try:
            res = self._scipy_optimize.linprog(
                c,
                A_ub=A_ub_rows if A_ub_rows else None,
                b_ub=b_ub_rows if b_ub_rows else None,
                bounds=bounds,
                method=self.method,
                options=opts,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            self.last_result = SolverResult(
                status=SolverStatus.UNKNOWN,
                satisfied=False,
                message=f"SciPy linprog execution error: {exc}",
                details={"error": str(exc)},
                solve_time_seconds=round(elapsed, 6),
                timed_out=False,
            )
            return []

        elapsed = time.perf_counter() - start_time

        timed_out = res.status == 1 or (
            not res.success and "time limit" in str(res.message).lower()
        )

        if res.success:
            res_status = SolverStatus.OPTIMAL
            satisfied = True
            msg = "SciPy linprog found optimal continuous allocation"
        elif timed_out:
            res_status = SolverStatus.UNKNOWN
            satisfied = False
            msg = f"SciPy linprog timed out after {effective_time_limit}s"
        elif res.status == 2:
            res_status = SolverStatus.INFEASIBLE
            satisfied = False
            msg = f"SciPy linprog proved problem is infeasible: {res.message}"
        else:
            res_status = SolverStatus.UNKNOWN
            satisfied = False
            msg = f"SciPy linprog terminated with status {res.status}: {res.message}"

        self.last_result = SolverResult(
            status=res_status,
            satisfied=satisfied,
            message=msg,
            details={"scipy_status": res.status, "message": str(res.message)},
            solve_time_seconds=round(elapsed, 6),
            timed_out=timed_out,
        )

        if not satisfied or res.x is None:
            return []

        actions: list[Action] = []
        action_idx = 0
        for pair, idx in pair_to_idx.items():
            val = float(res.x[idx])
            if val > 1e-6:
                s, d = pair
                actions.append(
                    Action(
                        id=f"{self.actor_id}_transfer_{action_idx}",
                        type=self.action_type,
                        parameters={
                            "source_resource": s,
                            "target_resource": d,
                            "quantity": round(val, 6),
                        },
                    )
                )
                action_idx += 1

        return actions

    def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]:
        """Propose actions during simulation rollout."""
        return self.propose(
            state=state,
            objective=None,
            time_limit_seconds=self.time_limit_seconds,
        )

    def clone(self) -> SciPyAllocationPlanner:
        """Create an independent copy of this planner."""
        return SciPyAllocationPlanner(
            actor_id=self.actor_id,
            sources=self.sources,
            destinations=self.destinations,
            demands=self.demands,
            capacities=self.capacities,
            costs=self.costs,
            action_type=self.action_type,
            time_limit_seconds=self.time_limit_seconds,
            method=self.method,
        )
