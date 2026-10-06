"""Multi-agent observation views, action proposals, and adjudication records.

Provides per-actor perspectives over immutable WorldState and typed proposals
for deterministic multi-agent adjudication.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState


class ActorObservationView(BaseModel):
    """Filtered, role-specific observation view over immutable WorldState."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: str = Field(description="Identifier of the observing actor.")
    step: int = Field(ge=0, description="Current simulation step.")
    timestamp: float = Field(ge=0.0, description="Current simulation timestamp.")
    visible_entities: tuple[Entity, ...] = Field(
        default=(), description="Entities visible to this actor."
    )
    visible_resources: dict[str, Resource] = Field(
        default_factory=dict, description="Resources visible to this actor."
    )
    visible_memory: dict[str, Any] = Field(
        default_factory=dict, description="State memory visible to this actor."
    )

    @classmethod
    def from_world_state(
        cls,
        state: WorldState,
        actor_id: str,
        entity_filter: Sequence[str] | None = None,
        resource_filter: Sequence[str] | None = None,
        memory_filter: Sequence[str] | None = None,
    ) -> ActorObservationView:
        """Create a filtered observation view from WorldState."""
        ent_set = set(entity_filter) if entity_filter is not None else None
        res_set = set(resource_filter) if resource_filter is not None else None
        mem_set = set(memory_filter) if memory_filter is not None else None

        ents = [e for e in state.entities.values() if ent_set is None or e.id in ent_set]
        res = {k: v for k, v in state.resources.items() if res_set is None or k in res_set}
        mem = {k: v for k, v in state.memory.items() if mem_set is None or k in mem_set}

        return cls(
            actor_id=actor_id,
            step=state.step,
            timestamp=state.timestamp,
            visible_entities=tuple(ents),
            visible_resources=res,
            visible_memory=mem,
        )


class ActorActionProposal(BaseModel):
    """Action proposed by an actor for execution during the current step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: str = Field(description="Proposing actor identifier.")
    action: Action = Field(description="Concrete action proposed.")
    priority: int = Field(default=0, description="Priority level (higher executed first).")
    intent: str | None = Field(
        default=None, description="Optional symbolic or natural language intent."
    )
    utility: float | None = Field(
        default=None, description="Expected utility of the action to the actor."
    )


class AdjudicationResult(BaseModel):
    """Result of deterministic mediation among competing action proposals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted_actions: tuple[Action, ...] = Field(
        default=(), description="Actions validated and accepted for execution."
    )
    rejected_actions: tuple[tuple[Action, str], ...] = Field(
        default=(), description="Actions rejected alongside reasons."
    )
    penalties: dict[str, float] = Field(
        default_factory=dict, description="Soft constraint penalties incurred."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Adjudication algorithm metadata."
    )

    @property
    def adjudication_fingerprint(self) -> str:
        """Deterministic canonical fingerprint of the adjudication outcome."""
        normalized = {
            "accepted": [a.id for a in self.accepted_actions],
            "rejected": [(a.id, reason) for a, reason in self.rejected_actions],
            "penalties": {k: float(v) for k, v in sorted(self.penalties.items())},
        }
        return canonical_sha256(normalized)
