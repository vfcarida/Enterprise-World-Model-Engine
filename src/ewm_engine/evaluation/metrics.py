"""Evaluation metrics for measuring performance, equity, and constraint compliance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ewm_engine.simulation.trajectory import Trajectory


@runtime_checkable
class Metric(Protocol):
    """Protocol for computing an evaluation metric across a simulation trajectory."""

    @property
    def name(self) -> str:
        """Unique metric name."""
        ...

    def compute(self, trajectory: Trajectory) -> float:
        """Compute metric value from a trajectory."""
        ...


class FinalResourceLevelMetric:
    """Measures the final level of a given resource."""

    def __init__(self, resource_id: str, name: str | None = None) -> None:
        self.resource_id = resource_id
        self._name = name or f"final_{resource_id}"

    @property
    def name(self) -> str:
        return self._name

    def compute(self, trajectory: Trajectory) -> float:
        res = trajectory.final_state.resources.get(self.resource_id)
        return res.current if res is not None else 0.0


class TotalViolationsMetric:
    """Counts the total constraint violations incurred along the trajectory."""

    def __init__(self, name: str = "constraint_violations") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, trajectory: Trajectory) -> float:
        return float(trajectory.total_violations)


class HardViolationsMetric:
    """Counts only hard (fatal/invariable) constraint violations."""

    def __init__(self, name: str = "hard_violations") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, trajectory: Trajectory) -> float:
        return float(trajectory.hard_violations)


class DistributionalEquityMetric:
    """Measures equity/disparity in resource allocation across entities (Gini coefficient)."""

    def __init__(self, resource_ids: Sequence[str], name: str = "allocation_equity") -> None:
        self.resource_ids = list(resource_ids)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def compute(self, trajectory: Trajectory) -> float:
        values = [
            trajectory.final_state.resources[r_id].current
            for r_id in self.resource_ids
            if r_id in trajectory.final_state.resources
        ]
        if not values or all(v == 0.0 for v in values):
            return 1.0  # Equal when all zero or empty

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        index = 0.0
        for i, val in enumerate(sorted_vals, 1):
            index += (2 * i - n - 1) * val
        gini = index / (n * sum(sorted_vals))
        # Equity = 1 - Gini (1.0 = perfectly equal, 0.0 = completely unequal)
        return max(0.0, min(1.0, 1.0 - gini))
