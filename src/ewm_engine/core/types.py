"""Central typing definitions, error hierarchy root, and core protocol interfaces for EWM Engine.

Defines strict type aliases (utilizing ``typing.Dict``, ``typing.List``, ``typing.Optional``),
the root ``EngineError`` exception base class, and runtime-checkable protocol contracts
(``StateInterface``, ``DynamicsInterface``, ``ConstraintInterface``, ``ActionInterface``,
``EventInterface``, ``EventSourceInterface``, ``ActorInterface``, ``TransitionResultInterface``)
governing deterministic simulation and reasoning boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Self,
    TypeAlias,
    runtime_checkable,
)

import numpy as np

# Identifier Type Aliases
EntityId: TypeAlias = str
ResourceId: TypeAlias = str
RelationshipId: TypeAlias = str
ActionId: TypeAlias = str
EventId: TypeAlias = str
ActorId: TypeAlias = str
ConstraintId: TypeAlias = str
ScenarioId: TypeAlias = str
TrajectoryId: TypeAlias = str

# Random Number Generator Type
RandomGenerator: TypeAlias = np.random.Generator

# Strict Container & Seed Type Aliases
StateDict: TypeAlias = Dict[str, Any]
ParameterMap: TypeAlias = Dict[str, Any]
DiagnosticsMap: TypeAlias = Dict[str, Any]
ViolationList: TypeAlias = List[str]
OptionalSeed: TypeAlias = Optional[int]


class EngineError(Exception):
    """Root base exception class for all Enterprise World Model (EWM) Engine errors.

    All domain, state, constraint, dynamics, temporal, bitemporal, serialization,
    and epistemic trust exceptions explicitly subclass ``EngineError`` to enforce a
    strict, unified error contract across the engine.
    """

    def __init__(
        self,
        *args: object,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(*args)
        self.details: Dict[str, Any] = dict(details) if details is not None else {}


@runtime_checkable
class StateInterface(Protocol):
    """Core protocol interface for immutable world state representations.

    Concrete state implementations (such as ``WorldState``) must expose a discrete
    non-negative step index, continuous simulation timestamp, canonical SHA-256
    cryptographic fingerprint (`state_hash` and `fingerprint`), deep cloning, and
    deterministic dictionary serialization.
    """

    @property
    def step(self) -> int:
        """Return the discrete non-negative simulation time step index."""
        ...

    @property
    def timestamp(self) -> float:
        """Return the continuous simulation timestamp."""
        ...

    @property
    def state_hash(self) -> str:
        """Return the canonical SHA-256 hex digest of the normalized state."""
        ...

    @property
    def fingerprint(self) -> str:
        """Return the canonical SHA-256 fingerprint alias for state_hash."""
        ...

    def clone(self) -> Self:
        """Return an isolated deep copy of this state instance with disjoint container IDs."""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the state into a standard JSON-compatible dictionary."""
        ...


@runtime_checkable
class ActionInterface(Protocol):
    """Core protocol interface for operational actions proposed or executed in the world.

    Concrete action types (such as ``Action``) expose an action identifier,
    categorical type, optional initiating actor identifier, and typed parameter payload.
    """

    @property
    def id(self) -> ActionId:
        """Return the unique action instance identifier."""
        ...

    @property
    def type(self) -> str:
        """Return the categorical action type identifier."""
        ...

    @property
    def actor_id(self) -> Optional[ActorId]:
        """Return the optional identifier of the actor initiating this action."""
        ...

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return the immutable action parameter dictionary."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Safely fetch a parameter value by key with an optional fallback."""
        ...


@runtime_checkable
class EventInterface(Protocol):
    """Core protocol interface for exogenous shocks and environmental events.

    Concrete event representations (such as ``ExogenousEvent``) expose a unique
    event identifier, classification type, severity measure, and parameter map.
    """

    @property
    def id(self) -> EventId:
        """Return the unique exogenous event identifier."""
        ...

    @property
    def type(self) -> str:
        """Return the event classification type."""
        ...

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return the event payload parameter map."""
        ...

    @property
    def severity(self) -> float:
        """Return the normalized or absolute event severity measure."""
        ...


@runtime_checkable
class TransitionResultInterface(Protocol):
    """Core protocol interface for the output of a dynamics state transition step.

    Encapsulates the evolved next world state, a dictionary of applied state changes,
    and execution diagnostics.
    """

    @property
    def next_state(self) -> Any:
        """Return the evolved world state after dynamics application."""
        ...

    @property
    def applied_changes(self) -> Dict[str, Any]:
        """Return the dictionary of state modifications enacted during transition."""
        ...

    @property
    def diagnostics(self) -> Dict[str, Any]:
        """Return execution metadata and diagnostic metrics from the transition."""
        ...


@runtime_checkable
class DynamicsInterface(Protocol):
    """Core protocol interface for pluggable state-transition dynamics models.

    Dynamics models accept the current state, accepted actions, exogenous events,
    and a deterministically seeded ``RandomGenerator``, returning a transition result.
    """

    def transition(
        self,
        state: Any,
        actions: Sequence[Any],
        exogenous_events: Sequence[Any],
        rng: RandomGenerator,
    ) -> Any:
        """Compute the next state transition given current state, actions, events, and seeded RNG."""
        ...


@runtime_checkable
class ConstraintInterface(Protocol):
    """Core protocol interface for hard and soft operational invariants.

    Constraints evaluate whether states and proposed actions conform to physical,
    accounting, and policy invariants within a lifecycle phase.
    """

    @property
    def constraint_id(self) -> ConstraintId:
        """Return the unique identifier of the constraint."""
        ...

    def evaluate(
        self,
        state: Any,
        actions: Sequence[Any] = (),
        *,
        phase: Any,
    ) -> Any:
        """Evaluate the constraint against a state and optional actions."""
        ...


@runtime_checkable
class EventSourceInterface(Protocol):
    """Core protocol interface for seeded exogenous event generators.

    Implementations sample exogenous shocks for a given state and discrete step
    using an explicit, deterministically seeded ``RandomGenerator``.
    """

    def sample(
        self,
        state: Any,
        step: int,
        rng: RandomGenerator,
    ) -> Sequence[Any]:
        """Sample exogenous events for the current state and simulation step."""
        ...


@runtime_checkable
class ActorInterface(Protocol):
    """Core protocol interface for policy agents and decision-making actors.

    Actors inspect the observed world state and execution context to propose a
    sequence of candidate actions at each simulation step.
    """

    @property
    def actor_id(self) -> ActorId:
        """Return the unique actor identifier."""
        ...

    def act(
        self,
        state: Any,
        context: Any,
    ) -> Sequence[Any]:
        """Propose actions given the observed world state and actor context."""
        ...


__all__ = [
    "ActionId",
    "ActionInterface",
    "ActorId",
    "ActorInterface",
    "ConstraintId",
    "ConstraintInterface",
    "DiagnosticsMap",
    "DynamicsInterface",
    "EngineError",
    "EntityId",
    "EventId",
    "EventInterface",
    "EventSourceInterface",
    "OptionalSeed",
    "ParameterMap",
    "RandomGenerator",
    "RelationshipId",
    "ResourceId",
    "ScenarioId",
    "StateDict",
    "StateInterface",
    "TrajectoryId",
    "TransitionResultInterface",
    "ViolationList",
]
