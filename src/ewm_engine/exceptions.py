"""Domain-specific exceptions for Enterprise World Model Engine (EWM Engine).

All exceptions inherit from EWMError to provide a consistent error hierarchy
across simulation, dynamics, constraints, and state management.
"""

from __future__ import annotations


class EWMError(Exception):
    """Base class for all Enterprise World Model Engine exceptions."""


class InvalidWorldStateError(EWMError):
    """Raised when a world state violates internal schema, structure, or invariant requirements."""


class ConstraintViolationError(EWMError):
    """Raised when a hard constraint is violated during pre-action or post-state evaluation."""

    def __init__(
        self,
        message: str,
        constraint_id: str,
        step: int | None = None,
        violating_entities: list[str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.constraint_id = constraint_id
        self.step = step
        self.violating_entities = violating_entities or []
        self.metadata = metadata or {}


class InvalidActionError(EWMError):
    """Raised when an actor proposes an action that is malformed or invalid."""


class ResourceBoundsError(EWMError):
    """Raised when a resource update breaches min/max bounds under hard limits."""


class SimulationConfigurationError(EWMError):
    """Raised when simulation parameters (horizon, samples, seeds) are invalid."""


class DynamicsError(EWMError):
    """Raised when a dynamics model fails to compute a state transition."""


class ScenarioError(EWMError):
    """Raised when scenario definition, initialization, or execution fails."""


class BranchingError(EWMError):
    """Raised when scenario branching from a state snapshot fails."""


class ProvenanceError(EWMError):
    """Raised when provenance tracking or causal evidence metadata is inconsistent."""


class SerializationError(SimulationConfigurationError):
    """Raised when serialization or deserialization fails."""


class SerializationSecurityError(SerializationError):
    """Raised when an unsafe serialization or deserialization operation is detected."""
