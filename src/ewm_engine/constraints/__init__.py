"""First-class constraint verification and provenance engine."""

from __future__ import annotations

from ewm_engine.constraints.base import Constraint
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.results import (
    ConstraintPhase,
    ConstraintResult,
    ConstraintSeverity,
)
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
    ResourceNonNegativeConstraint,
)

__all__ = [
    "ActionTransferAvailabilityConstraint",
    "Constraint",
    "ConstraintPhase",
    "ConstraintRegistry",
    "ConstraintResult",
    "ConstraintSeverity",
    "ResourceCapacityConstraint",
    "ResourceNonNegativeConstraint",
]
