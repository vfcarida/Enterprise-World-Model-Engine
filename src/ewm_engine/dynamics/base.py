"""Base protocols and transition types for pluggable dynamics in EWM Engine."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.provenance.evidence import EvidenceLevel


class TransitionResult(BaseModel):
    """The result of executing a dynamics transition step.

    Contains the resulting next world state, a summary of applied changes,
    the epistemic evidence tier, and execution diagnostics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    next_state: WorldState = Field(
        description="The evolved world state after dynamics application."
    )
    applied_changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Key deltas, resource flows, and state modifications enacted during transition.",
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.STRUCTURAL,
        description="Epistemic status of the transition mechanism.",
    )
    diagnostics: dict[str, Any] = Field(
        default_factory=dict,
        description="Execution metadata, internal model states, and performance metrics.",
    )
    model_name: str = Field(
        default="", description="Name of the dynamics model producing this result."
    )


@runtime_checkable
class DynamicsModel(Protocol):
    """Pluggable interface for defining how a world state evolves over time.

    Dynamics models accept the current WorldState, accepted Actions, exogenous shocks,
    and a seeded random generator, returning a TransitionResult.
    """

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        exogenous_events: Sequence[ExogenousEvent],
        rng: RandomGenerator,
    ) -> TransitionResult:
        """Compute the next state given actions, events, and randomness."""
        ...
