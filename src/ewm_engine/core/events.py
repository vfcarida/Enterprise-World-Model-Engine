"""Exogenous events affecting the world outside the direct control of simulated policies."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.types import EventId, RandomGenerator

if TYPE_CHECKING:
    from ewm_engine.core.state import WorldState


class ExogenousEvent(BaseModel):
    """An external shock or event outside direct policy control.

    Examples include weather disruptions, sudden supplier bankruptcy, market demand surges,
    traffic accidents, or regulatory shifts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: EventId = Field(description="Unique event identifier.")
    type: str = Field(
        description="Event classification type (e.g., 'rainfall_surge', 'road_closure')."
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload attributes.",
    )
    timestamp: float = Field(default=0.0, description="Simulation timestamp when event occurs.")
    severity: float = Field(
        default=1.0, description="Normalized or absolute event severity measure."
    )
    description: str = Field(default="", description="Descriptive context for the event.")

    def __init__(self, **kwargs: Any) -> None:
        if "parameters" in kwargs and kwargs["parameters"] is not None:
            kwargs["parameters"] = copy.deepcopy(kwargs["parameters"])
        super().__init__(**kwargs)

    def __getattribute__(self, name: str) -> Any:
        val = super().__getattribute__(name)
        if name == "parameters" and isinstance(val, dict):
            return copy.deepcopy(val)
        return val

    def get(self, key: str, default: Any = None) -> Any:
        """Safely fetch an event parameter value."""
        return self.parameters.get(key, default)


class ExogenousEventSource(Protocol):
    """Protocol for generating exogenous events over simulation steps."""

    def sample(
        self,
        state: WorldState,
        step: int,
        rng: RandomGenerator,
    ) -> Sequence[ExogenousEvent]:
        """Sample exogenous events for the current state and step."""
        ...
