"""Constraint protocol and foundational abstractions for organizational rules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ConstraintId


@runtime_checkable
class Constraint(Protocol):
    """First-class representation of an organizational, physical, or policy rule.

    Constraints evaluate whether states and proposed actions conform to
    operational invariants.
    """

    @property
    def constraint_id(self) -> ConstraintId:
        """Unique identifier of the constraint."""
        ...

    @property
    def version(self) -> str:
        """Semantic version of the rule definition."""
        ...

    @property
    def severity(self) -> ConstraintSeverity:
        """Severity classification: HARD or SOFT."""
        ...

    @property
    def description(self) -> str:
        """Human-readable explanation of the constraint."""
        ...

    def evaluate(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        *,
        phase: ConstraintPhase,
    ) -> ConstraintResult:
        """Evaluate the constraint against a state and optional actions within a specific lifecycle phase."""
        ...
