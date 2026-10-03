"""Google OR-Tools mathematical programming optimization adapter for EWM Engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.exceptions import SimulationConfigurationError


class ORToolsAllocationAdapter:
    """Operations Research optimization adapter leveraging Google OR-Tools.

    Solves optimal resource redistribution / min-cost network flow problems
    over an EWM WorldState to automatically synthesize valid Action sequences.
    """

    def __init__(self, solver_type: str = "GLOP") -> None:
        """Initialize OR-Tools solver wrapper.

        Args:
            solver_type: Linear programming solver backend (default: 'GLOP').
        """
        self.solver_type = solver_type

        # Verify OR-Tools availability dynamically
        import importlib

        try:
            pywraplp = importlib.import_module("ortools.linear_solver.pywraplp")
            self._pywraplp: Any | None = pywraplp
        except ImportError:
            self._pywraplp = None

    @property
    def is_available(self) -> bool:
        """Return True if Google OR-Tools is installed and importable."""
        return self._pywraplp is not None

    def optimize_transfers(
        self,
        state: WorldState,
        sources: Sequence[str],
        destinations: Sequence[str],
        demands: dict[str, float],
        capacities: dict[tuple[str, str], float] | None = None,
        costs: dict[tuple[str, str], float] | None = None,
    ) -> Sequence[Action]:
        """Solve a transportation / flow problem to generate optimal transfer actions.

        Args:
            state: Current world state providing source resource stock levels.
            sources: Resource IDs of supplying entities.
            destinations: Resource IDs of demanding entities.
            demands: Required resource quantities per destination.
            capacities: Optional edge capacity limits `(src, dst) -> max_qty`.
            costs: Optional edge transfer costs `(src, dst) -> unit_cost`.

        Returns:
            Sequence of concrete `transfer_resource` Action objects.
        """
        if self._pywraplp is None:
            raise SimulationConfigurationError(
                "Google OR-Tools is required for ORToolsAllocationAdapter. "
                "Install via `pip install ortools` or `pip install ewm-engine[solvers]`."
            )

        solver = self._pywraplp.Solver.CreateSolver(self.solver_type)
        if not solver:
            raise SimulationConfigurationError(
                f"Failed to create OR-Tools solver with backend: {self.solver_type}"
            )

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

        # Demand Constraints: sum_i flow[i, j] >= demand[j] (or as much as possible)
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

        status = solver.Solve()
        if status not in (
            self._pywraplp.Solver.OPTIMAL,
            self._pywraplp.Solver.FEASIBLE,
        ):
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
