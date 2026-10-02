"""Entity and Relationship domain primitives for organizational world modeling."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.types import EntityId


class Entity(BaseModel):
    """Represents an autonomous, structural, or physical object participating in the world.

    Entities represent actors, physical facilities, geographic regions, vehicles,
    suppliers, or any discrete organizational participant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: EntityId = Field(description="Unique entity identifier.")
    type: str = Field(
        description="Domain-specific entity classification (e.g., 'warehouse', 'hospital')."
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Immutable property map associated with this entity.",
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Categorical tags for grouping and querying entities.",
    )

    def get(self, key: str, default: Any = None) -> Any:
        """Safely fetch an attribute value."""
        return self.attributes.get(key, default)


class Relationship(BaseModel):
    """Represents a directed, typed relational edge between two entities.

    Models structural, logistical, dependency, or jurisdictional relations
    (e.g., 'supplies', 'depends_on', 'connected_to', 'serves').
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: EntityId = Field(description="Source entity identifier.")
    target: EntityId = Field(description="Target entity identifier.")
    type: str = Field(description="Relation type identifier.")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Attributes associated with this relation (e.g., distance, bandwidth).",
    )

    @property
    def key(self) -> tuple[EntityId, EntityId, str]:
        """Unique relational key tuple."""
        return (self.source, self.target, self.type)

    def get(self, key: str, default: Any = None) -> Any:
        """Safely fetch a relational attribute value."""
        return self.attributes.get(key, default)
