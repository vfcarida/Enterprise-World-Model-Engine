"""Formal SMT and mathematical solver adapters for EWM Engine."""

from __future__ import annotations

import importlib
import inspect
import time
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
from ewm_engine.integrations.protocols import (
    ConstraintSolver,
    SolverResult,
    SolverStatus,
)


class Z3ConstraintAdapter(Constraint, ConstraintSolver):
    """Adapter bridging formal Z3 SMT symbolic expressions with the Constraint protocol.

    Implements both the EWM Engine `Constraint` protocol (for in-simulation verification)
    and the `ConstraintSolver` protocol (for stand-alone formal checks).
    Every evaluation enforces a finite, mandatory timeout to prevent unbounded solver hangs.

    If z3-solver is not installed, raises an informative SimulationConfigurationError
    upon evaluation.
    """

    def __init__(
        self,
        constraint_id: ConstraintId,
        solver_fn: Callable[..., tuple[bool, str, dict[str, Any]] | SolverResult],
        severity: ConstraintSeverity = ConstraintSeverity.HARD,
        version: str = "1.0.0",
        timeout_ms: int = 5000,
    ) -> None:
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {timeout_ms}ms")

        self._constraint_id = constraint_id
        self._solver_fn = solver_fn
        self._severity = severity
        self._version = version
        self._timeout_ms = timeout_ms

        # Verify z3 availability dynamically
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
    def timeout_ms(self) -> int:
        """Enforced solver timeout in milliseconds."""
        return self._timeout_ms

    @property
    def description(self) -> str:
        return (
            f"Formal Z3 SMT constraint verification: {self._constraint_id} "
            f"(timeout={self._timeout_ms}ms)"
        )

    def check(
        self,
        *,
        state: WorldState,
        actions: Sequence[Action] = (),
        time_limit_seconds: float | None = None,
    ) -> SolverResult:
        """Verify whether proposed actions and state satisfy formal SMT constraints.

        Args:
            state: Current world state to verify.
            actions: Optional sequence of proposed actions.
            time_limit_seconds: Override timeout in seconds. If None, uses configured timeout_ms.

        Returns:
            A structured SolverResult including satisfaction status, unsat core,
            and execution time.
        """
        if self._z3_module is None:
            raise SimulationConfigurationError(
                "z3-solver is not installed. Install via `pip install z3-solver` "
                "or `pip install ewm-engine[solvers]`."
            )

        if time_limit_seconds is not None:
            if time_limit_seconds <= 0:
                raise ValueError(f"time_limit_seconds must be positive, got {time_limit_seconds}")
            effective_timeout_ms = max(1, int(time_limit_seconds * 1000))
        else:
            effective_timeout_ms = self._timeout_ms

        action = actions[0] if actions else None
        start_time = time.perf_counter()

        # Check signature of solver_fn to provide effective_timeout_ms if supported
        sig = inspect.signature(self._solver_fn)
        params_count = len(sig.parameters)
        has_varargs = any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in sig.parameters.values()
        )

        try:
            if params_count >= 4 or has_varargs:
                res = self._solver_fn(state, action, self._z3_module, effective_timeout_ms)
            else:
                res = self._solver_fn(state, action, self._z3_module)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            return SolverResult(
                status=SolverStatus.UNKNOWN,
                satisfied=False,
                message=f"Solver execution error: {exc}",
                details={"error": str(exc)},
                solve_time_seconds=round(elapsed, 6),
                timed_out=False,
            )

        elapsed = time.perf_counter() - start_time

        if isinstance(res, SolverResult):
            return res

        satisfied, message, details = res
        timed_out = bool(details.get("timed_out", False)) or (details.get("reason") == "timeout")

        if timed_out:
            status = SolverStatus.UNKNOWN
        elif satisfied:
            status = SolverStatus.OPTIMAL
        else:
            status = SolverStatus.INFEASIBLE

        unsat_core = tuple(str(item) for item in details.get("unsat_core", ()))

        return SolverResult(
            status=status,
            satisfied=satisfied and not timed_out,
            message=message,
            details=details,
            unsat_core=unsat_core,
            solve_time_seconds=round(elapsed, 6),
            timed_out=timed_out,
        )

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase = ConstraintPhase.POST_TRANSITION,
    ) -> ConstraintResult:
        """Evaluate constraint via Z3 SMT solver within the simulation pipeline."""
        solver_res = self.check(
            state=state,
            actions=actions,
            time_limit_seconds=self._timeout_ms / 1000.0,
        )

        violating_values = dict(solver_res.details)
        if solver_res.timed_out:
            violating_values["timed_out"] = True
            message = f"Z3 solver timed out after {self._timeout_ms}ms: {solver_res.message}"
        else:
            message = solver_res.message

        if solver_res.unsat_core:
            violating_values["unsat_core"] = list(solver_res.unsat_core)

        return ConstraintResult(
            satisfied=solver_res.satisfied,
            constraint_id=self.constraint_id,
            constraint_version=self.version,
            severity=self.severity,
            phase=phase,
            message=message,
            violating_values=violating_values,
        )
