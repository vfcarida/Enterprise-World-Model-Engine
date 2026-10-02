"""Actor protocols and execution context for decision-making agents."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ActorId, RandomGenerator


class ActorContext(BaseModel):
    """Contextual metadata passed to an actor during decision evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    step: int = Field(default=0, description="Current simulation step index.")
    timestamp: float = Field(default=0.0, description="Continuous simulation time.")
    rng: RandomGenerator = Field(description="Deterministic seeded random number generator.")
    active_interventions: tuple[str, ...] = Field(
        default_factory=tuple,
        description="IDs of currently active policy interventions.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary simulation context.",
    )


@runtime_checkable
class Actor(Protocol):
    """Protocol for entities or policies capable of selecting actions within the world.

    An Actor can be heuristic, rule-based, stochastic, optimization-based,
    or adapted from an external agentic framework.
    """

    @property
    def actor_id(self) -> ActorId:
        """Unique actor identifier."""
        ...

    def act(
        self,
        state: WorldState,
        context: ActorContext,
    ) -> Sequence[Action]:
        """Propose actions given the observed world state and context."""
        ...
