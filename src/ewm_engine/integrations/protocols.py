"""Protocols and result models for mathematical solvers and action planners.

Formalizes the architectural boundary:
- Solvers (`ConstraintSolver`) verify state and action invariants and return structured `SolverResult` payloads.
- Planners (`ActionPlanner`) synthesize candidate actions to optimize objectives.
- Planners and solvers NEVER mutate state directly; proposed actions flow through the engine's constraint engine.
- Every solver and planner must accept and enforce finite time/resource limits.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ewm_engine.core.actions import Action
    from ewm_engine.core.state import WorldState


class SolverStatus(StrEnum):
    """Execution status returned by formal constraint solvers and optimizers."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


class SolverResult(BaseModel):
    """Immutable result payload returned by a ConstraintSolver evaluation.

    Attributes:
        status: High-level solver status (OPTIMAL, FEASIBLE, INFEASIBLE, UNKNOWN).
        satisfied: Boolean indicating whether all verified constraints hold.
        message: Human-readable solver message or diagnostic summary.
        details: Diagnostic key-value map containing solution values or slack.
        unsat_core: Tuple of constraint or assumption identifiers forming the minimal
            unsatisfiable core (for auditability when a solver finds infeasibility).
        solve_time_seconds: Measured wall-clock duration of the solver call.
        timed_out: True if solver terminated due to reaching time or resource limits.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: SolverStatus = Field(description="Solver outcome status.")
    satisfied: bool = Field(description="Whether the tested constraints hold.")
    message: str = Field(default="", description="Descriptive diagnostic message.")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic payload.")
    unsat_core: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Identified unsatisfiable core constraints when status is INFEASIBLE.",
    )
    solve_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Solver execution time in seconds.",
    )
    timed_out: bool = Field(
        default=False,
        description="Whether solver terminated because time limit expired.",
    )


@runtime_checkable
class ConstraintSolver(Protocol):
    """Protocol for formal SMT, SAT, and mathematical constraint verification engines."""

    def check(
        self,
        *,
        state: WorldState,
        actions: Sequence[Action] = (),
        time_limit_seconds: float | None = None,
    ) -> SolverResult:
        """Verify whether proposed actions and state satisfy formal constraints.

        Args:
            state: Current world state to verify.
            actions: Optional sequence of proposed actions.
            time_limit_seconds: Maximum allowable execution time. If reached,
                the solver must return SolverStatus.UNKNOWN with timed_out=True
                rather than hanging indefinitely.

        Returns:
            A structured SolverResult.
        """
        ...


@runtime_checkable
class ActionPlanner(Protocol):
    """Protocol for operations-research and mathematical programming action planners."""

    def propose(
        self,
        *,
        state: WorldState,
        objective: str | dict[str, Any] | None = None,
        time_limit_seconds: float | None = None,
    ) -> Sequence[Action]:
        """Propose candidate operational actions to optimize the specified objective.

        Planners NEVER mutate world state directly. Candidate actions are returned
        for verification by the engine's constraint pipeline.

        Args:
            state: Current world state providing supply, demand, and network topology.
            objective: Optimization goal specification or parameter map.
            time_limit_seconds: Maximum solver execution time. If exceeded, returns
                a fallback or empty action sequence.

        Returns:
            Sequence of proposed Action objects.
        """
        ...
