"""Formal SMT and mathematical solver adapters for EWM Engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ewm_engine.constraints.base import Constraint
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ConstraintId
from ewm_engine.exceptions import SimulationConfigurationError


class Z3ConstraintAdapter(Constraint):
    """Adapter bridging formal Z3 SMT symbolic expressions with the Constraint protocol.

    If z3-solver is not installed, raises an informative SimulationConfigurationError
    upon instantiation or evaluation.
    """

    def __init__(
        self,
        constraint_id: ConstraintId,
        solver_fn: Callable[[WorldState, Action | None, Any], tuple[bool, str, dict[str, Any]]],
        severity: ConstraintSeverity = ConstraintSeverity.HARD,
        version: str = "1.0.0",
    ) -> None:
        self._constraint_id = constraint_id
        self._solver_fn = solver_fn
        self._severity = severity
        self._version = version

        # Verify z3 availability dynamically
        import importlib

        try:
            self._z3_module: Any | None = importlib.import_module("z3")
        except ImportError:
            self._z3_module = None

    @property
    def constraint_id(self) -> ConstraintId:
        return self._constraint_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def severity(self) -> ConstraintSeverity:
        return self._severity

    @property
    def description(self) -> str:
        return f"Formal Z3 SMT constraint verification: {self._constraint_id}"

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.POST_TRANSITION,
    ) -> ConstraintResult:
        """Evaluate constraint via Z3 SMT solver."""
        if self._z3_module is None:
            raise SimulationConfigurationError(
                "z3-solver is not installed. Install via `pip install z3-solver` "
                "or `pip install ewm-engine[solvers]`."
            )

        action = actions[0] if actions else None
        satisfied, message, violating_values = self._solver_fn(state, action, self._z3_module)
        return ConstraintResult(
            satisfied=satisfied,
            constraint_id=self.constraint_id,
            constraint_version=self.version,
            severity=self.severity,
            phase=phase,
            message=message,
            violating_values=violating_values,
        )
