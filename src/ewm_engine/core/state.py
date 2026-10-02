"""Canonical immutable WorldState representation for enterprise dynamics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.types import EntityId, ResourceId
from ewm_engine.exceptions import InvalidWorldStateError


class WorldState(BaseModel):
    r"""Canonical representation of the enterprise or ecosystem state at a point in time.

    Formal Conceptual Representation:
        S_t = (G_t, R_t, M_t, \Gamma_t, C_t)
    where:
        G_t = Entities and typed relationships
        R_t = Measurable finite resources and capacities
        M_t = Temporal memory and operational history
        \Gamma_t = Active operational rules and policy parameters
        C_t = Exogenous and environmental context

    WorldState is immutable (frozen=True), branchable, serializable, and auditable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entities: dict[EntityId, Entity] = Field(
        default_factory=dict,
        description="Active participating entities keyed by EntityId.",
    )
    relationships: tuple[Relationship, ...] = Field(
        default_factory=tuple,
        description="Immutable sequence of directed relational edges between entities.",
    )
    resources: dict[ResourceId, Resource] = Field(
        default_factory=dict,
        description="Finite or measurable resources keyed by ResourceId.",
    )
    memory: dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant temporal memory, running statistics, or history lags.",
    )
    active_rules: dict[str, Any] = Field(
        default_factory=dict,
        description="Active policy parameters, operational thresholds, and rule flags.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Environmental, economic, or meteorological context variables.",
    )
    timestamp: float = Field(default=0.0, description="Continuous simulation time.")
    step: int = Field(default=0, description="Discrete simulation step index.")

    def __init__(
        self,
        entities: Sequence[Entity] | Mapping[EntityId, Entity] = (),
        relationships: Sequence[Relationship] = (),
        resources: Sequence[Resource] | Mapping[ResourceId, Resource] = (),
        memory: Mapping[str, Any] | None = None,
        active_rules: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        timestamp: float = 0.0,
        step: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            entities=entities,
            relationships=relationships,
            resources=resources,
            memory=dict(memory or {}),
            active_rules=dict(active_rules or {}),
            context=dict(context or {}),
            timestamp=timestamp,
            step=step,
            **kwargs,
        )

    @field_validator("entities", mode="before")
    @classmethod
    def _coerce_entities(cls, v: Any) -> dict[EntityId, Entity]:
        if isinstance(v, (list, tuple)):
            res: dict[EntityId, Entity] = {}
            for item in v:
                entity = item if isinstance(item, Entity) else Entity(**item)
                res[entity.id] = entity
            return res
        if isinstance(v, dict):
            res = {}
            for k, val in v.items():
                res[k] = val if isinstance(val, Entity) else Entity(**val)
            return res
        return {}

    @field_validator("resources", mode="before")
    @classmethod
    def _coerce_resources(cls, v: Any) -> dict[ResourceId, Resource]:
        if isinstance(v, (list, tuple)):
            res: dict[ResourceId, Resource] = {}
            for item in v:
                resource = item if isinstance(item, Resource) else Resource(**item)
                res[resource.id] = resource
            return res
        if isinstance(v, dict):
            res = {}
            for k, val in v.items():
                res[k] = val if isinstance(val, Resource) else Resource(**val)
            return res
        return {}

    @field_validator("relationships", mode="before")
    @classmethod
    def _coerce_relationships(cls, v: Any) -> tuple[Relationship, ...]:
        if isinstance(v, (list, tuple)):
            return tuple(
                item if isinstance(item, Relationship) else Relationship(**item) for item in v
            )
        return ()

    @property
    def state_hash(self) -> str:
        """Deterministic cryptographic SHA-256 fingerprint of the normalized state."""
        canonical_dict = {
            "step": self.step,
            "timestamp": round(self.timestamp, 6),
            "entities": sorted(
                [
                    {"id": e.id, "type": e.type, "attributes": e.attributes}
                    for e in self.entities.values()
                ],
                key=lambda x: str(x["id"]),
            ),
            "relationships": sorted(
                [
                    {
                        "source": r.source,
                        "target": r.target,
                        "type": r.type,
                        "attributes": r.attributes,
                    }
                    for r in self.relationships
                ],
                key=lambda x: (str(x["source"]), str(x["target"]), str(x["type"])),
            ),
            "resources": sorted(
                [
                    {
                        "id": r.id,
                        "current": round(r.current, 6),
                        "min": r.min_value,
                        "max": r.max_value,
                    }
                    for r in self.resources.values()
                ],
                key=lambda x: str(x["id"]),
            ),
            "memory": self.memory,
            "active_rules": self.active_rules,
            "context": self.context,
        }
        encoded = json.dumps(canonical_dict, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def get_entity(self, entity_id: EntityId) -> Entity:
        """Fetch an entity by ID or raise InvalidWorldStateError."""
        if entity_id not in self.entities:
            raise InvalidWorldStateError(f"Entity '{entity_id}' not found in WorldState.")
        return self.entities[entity_id]

    def get_resource(self, resource_id: ResourceId) -> Resource:
        """Fetch a resource by ID or raise InvalidWorldStateError."""
        if resource_id not in self.resources:
            raise InvalidWorldStateError(f"Resource '{resource_id}' not found in WorldState.")
        return self.resources[resource_id]

    def get_relationships(
        self,
        source: EntityId | None = None,
        target: EntityId | None = None,
        type: str | None = None,
    ) -> list[Relationship]:
        """Query relational edges matching source, target, and/or type filters."""
        matches: list[Relationship] = []
        for rel in self.relationships:
            if source is not None and rel.source != source:
                continue
            if target is not None and rel.target != target:
                continue
            if type is not None and rel.type != type:
                continue
            matches.append(rel)
        return matches

    def update_resource(
        self,
        resource_id: ResourceId,
        delta: float = 0.0,
        new_value: float | None = None,
        clamp: bool = False,
        enforce_bounds: bool = False,
    ) -> WorldState:
        """Produce an updated WorldState with a modified resource level."""
        current_res = self.get_resource(resource_id)
        if new_value is not None:
            updated_res = current_res.with_value(
                new_value, clamp=clamp, enforce_bounds=enforce_bounds
            )
        else:
            updated_res = current_res.with_delta(delta, clamp=clamp, enforce_bounds=enforce_bounds)

        new_resources = dict(self.resources)
        new_resources[resource_id] = updated_res
        return self.model_copy(update={"resources": new_resources})

    def with_entity(self, entity: Entity) -> WorldState:
        """Return a new WorldState with an added or updated entity."""
        new_entities = dict(self.entities)
        new_entities[entity.id] = entity
        return self.model_copy(update={"entities": new_entities})

    def with_relationship(self, relationship: Relationship) -> WorldState:
        """Return a new WorldState with an added relationship."""
        return self.model_copy(update={"relationships": (*self.relationships, relationship)})

    def with_memory(self, key: str, value: Any) -> WorldState:
        """Return a new WorldState with an updated memory entry."""
        new_mem = dict(self.memory)
        new_mem[key] = value
        return self.model_copy(update={"memory": new_mem})

    def with_context(self, key: str, value: Any) -> WorldState:
        """Return a new WorldState with an updated context variable."""
        new_ctx = dict(self.context)
        new_ctx[key] = value
        return self.model_copy(update={"context": new_ctx})

    def advance_time(self, delta_t: float = 1.0) -> WorldState:
        """Return a new WorldState with advanced step and timestamp."""
        return self.model_copy(
            update={
                "step": self.step + 1,
                "timestamp": self.timestamp + delta_t,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert state to a standard JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldState:
        """Reconstruct WorldState from a dictionary."""
        return cls.model_validate(data)
