"""Entity and Relationship domain primitives for organizational world modeling."""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.trust import validate_dependency_for_query
from ewm_engine.core.types import EntityId
from ewm_engine.provenance.evidence import (
    DependencyEvidenceNature,
    DependencyQueryKind,
    EvidenceLevel,
    evidence_level_to_nature,
    evidence_nature_to_level,
)


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

    def __init__(self, **kwargs: Any) -> None:
        if "attributes" in kwargs and kwargs["attributes"] is not None:
            kwargs["attributes"] = copy.deepcopy(kwargs["attributes"])
        super().__init__(**kwargs)

    def __getattribute__(self, name: str) -> Any:
        val = super().__getattribute__(name)
        if name == "attributes" and isinstance(val, dict):
            return copy.deepcopy(val)
        return val

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

    def __init__(self, **kwargs: Any) -> None:
        if "attributes" in kwargs and kwargs["attributes"] is not None:
            kwargs["attributes"] = copy.deepcopy(kwargs["attributes"])
        super().__init__(**kwargs)

    def __getattribute__(self, name: str) -> Any:
        val = super().__getattribute__(name)
        if name == "attributes" and isinstance(val, dict):
            return copy.deepcopy(val)
        return val

    @property
    def key(self) -> tuple[EntityId, EntityId, str]:
        """Unique relational key tuple."""
        return (self.source, self.target, self.type)

    def get(self, key: str, default: Any = None) -> Any:
        """Safely fetch a relational attribute value."""
        return self.attributes.get(key, default)

    @property
    def evidence_nature(self) -> DependencyEvidenceNature:
        """Resolve the declared DependencyEvidenceNature for this relationship.

        Resolution order (fail-closed):
        1. Explicit ``attributes['evidence_nature']`` if present.
        2. Mapped from ``attributes['evidence_level']`` if present.
        3. Defaults to ``DependencyEvidenceNature.STATISTICAL_ASSOCIATION`` so
           untagged dependencies are never silently treated as causal.
        """
        raw_attrs = super().__getattribute__("attributes")
        if "evidence_nature" in raw_attrs and raw_attrs["evidence_nature"] is not None:
            return DependencyEvidenceNature(str(raw_attrs["evidence_nature"]))
        if "evidence_level" in raw_attrs and raw_attrs["evidence_level"] is not None:
            return evidence_level_to_nature(EvidenceLevel(str(raw_attrs["evidence_level"])))
        return DependencyEvidenceNature.STATISTICAL_ASSOCIATION

    @property
    def effective_evidence_level(self) -> EvidenceLevel:
        """Resolve the effective EvidenceLevel for this relationship."""
        raw_attrs = super().__getattribute__("attributes")
        is_identifiable = bool(raw_attrs.get("is_identifiable", False))
        overlap_satisfied = bool(raw_attrs.get("overlap_satisfied", True))
        provenance = raw_attrs.get("identification_provenance")
        has_id = (
            is_identifiable
            and overlap_satisfied
            and isinstance(provenance, str)
            and bool(provenance.strip())
        )
        return evidence_nature_to_level(self.evidence_nature, has_identification_evidence=has_id)

    def with_evidence_nature(
        self,
        evidence_nature: DependencyEvidenceNature,
        *,
        is_identifiable: bool | None = None,
        overlap_satisfied: bool | None = None,
        identification_provenance: str | None = None,
    ) -> Relationship:
        """Return a new Relationship with updated evidence-nature metadata in attributes."""
        attrs = self.attributes
        attrs["evidence_nature"] = evidence_nature.value
        if is_identifiable is not None:
            attrs["is_identifiable"] = is_identifiable
        if overlap_satisfied is not None:
            attrs["overlap_satisfied"] = overlap_satisfied
        if identification_provenance is not None:
            attrs["identification_provenance"] = identification_provenance
        return Relationship(
            source=self.source,
            target=self.target,
            type=self.type,
            attributes=attrs,
        )

    def validate_for_query(
        self,
        query_kind: DependencyQueryKind,
        *,
        strict: bool = True,
    ) -> EvidenceLevel:
        """Validate that this relationship's evidence nature permits ``query_kind``."""
        raw_attrs = super().__getattribute__("attributes")
        is_identifiable = bool(raw_attrs.get("is_identifiable", False))
        overlap_satisfied = bool(raw_attrs.get("overlap_satisfied", True))
        provenance = raw_attrs.get("identification_provenance")
        dep_id = f"{self.source}-[{self.type}]->{self.target}"
        return validate_dependency_for_query(
            evidence_nature=self.evidence_nature,
            query_kind=query_kind,
            is_identifiable=is_identifiable,
            overlap_satisfied=overlap_satisfied,
            identification_provenance=str(provenance) if provenance is not None else None,
            dependency_id=dep_id,
            strict=strict,
        )
