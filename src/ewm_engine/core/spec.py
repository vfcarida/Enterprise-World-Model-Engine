"""Safe declarative specification schema parser for World and WorldState.

Enables defining entities, resources, relationships, and constraints declaratively
via strictly validated JSON or YAML without arbitrary code execution.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
    ResourceNonNegativeConstraint,
)
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.exceptions import SimulationConfigurationError


class EntitySpec(BaseModel):
    """Specification schema for an entity."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: str = "entity"
    attributes: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class RelationshipSpec(BaseModel):
    """Specification schema for an entity relationship."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    type: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class ResourceSpec(BaseModel):
    """Specification schema for a measurable resource."""

    model_config = ConfigDict(extra="forbid")

    id: str
    current: float = 0.0
    min_value: float = 0.0
    max_value: float = float("inf")
    unit: str = "units"
    entity_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class ConstraintSpec(BaseModel):
    """Specification schema for standard constraints."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="'capacity', 'non_negative', or 'transfer_availability'")
    resource_id: str | None = None
    max_capacity: float | None = None
    severity: str = "hard"
    constraint_id: str | None = None


class WorldMetadataSpec(BaseModel):
    """Metadata specification schema."""

    model_config = ConfigDict(extra="forbid")

    name: str = "EnterpriseWorld"
    step: int = 0
    timestamp: float = 0.0
    memory: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class WorldSpecification(BaseModel):
    """Root declarative specification for an Enterprise World Model.

    Provides safe, non-executable serialization and deserialization from YAML and JSON.
    """

    model_config = ConfigDict(extra="forbid")

    world: WorldMetadataSpec = Field(default_factory=WorldMetadataSpec)
    entities: list[EntitySpec] = Field(default_factory=list)
    relationships: list[RelationshipSpec] = Field(default_factory=list)
    resources: list[ResourceSpec] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldSpecification:
        """Parse and validate a dictionary against the world specification schema."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> WorldSpecification:
        """Parse world specification from a JSON string."""
        try:
            data = json.loads(json_str)
        except Exception as exc:
            raise SimulationConfigurationError(f"Invalid JSON specification: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> WorldSpecification:
        """Safely parse world specification from a YAML string without code execution."""
        try:
            import yaml
        except ImportError as exc:
            raise SimulationConfigurationError(
                "PyYAML is required for YAML world specifications. Install via `pip install PyYAML`."
            ) from exc

        try:
            # Strictly safe loading to prevent arbitrary code execution
            data = yaml.safe_load(yaml_str)
        except Exception as exc:
            raise SimulationConfigurationError(f"Invalid YAML specification: {exc}") from exc

        if not isinstance(data, dict):
            raise SimulationConfigurationError(
                f"YAML specification must define a top-level dictionary, got {type(data).__name__}"
            )
        return cls.from_dict(data)

    def build_state(self) -> WorldState:
        """Build an immutable WorldState from this specification."""
        entities = [
            Entity(id=e.id, type=e.type, attributes=e.attributes, tags=tuple(e.tags))
            for e in self.entities
        ]
        relationships = [
            Relationship(source=r.source, target=r.target, type=r.type, attributes=r.attributes)
            for r in self.relationships
        ]
        resources = [
            Resource(
                id=res.id,
                current=res.current,
                min_value=res.min_value,
                max_value=res.max_value,
                unit=res.unit,
                entity_id=res.entity_id,
            )
            for res in self.resources
        ]

        return WorldState(
            step=self.world.step,
            timestamp=self.world.timestamp,
            entities=entities,
            relationships=relationships,
            resources=resources,
            memory=self.world.memory,
            context=self.world.context,
        )

    def build_world(self) -> World:
        """Compile the specification into an active, runnable World container."""
        initial_state = self.build_state()
        registry = ConstraintRegistry()

        for c_spec in self.constraints:
            severity = (
                ConstraintSeverity.HARD
                if c_spec.severity.lower() == "hard"
                else ConstraintSeverity.SOFT
            )
            if c_spec.type == "capacity":
                if c_spec.resource_id is None:
                    raise SimulationConfigurationError(
                        "Capacity constraint requires 'resource_id'."
                    )
                registry.register(
                    ResourceCapacityConstraint(
                        resource_id=c_spec.resource_id,
                        severity=severity,
                        constraint_id=c_spec.constraint_id,
                    )
                )
            elif c_spec.type == "non_negative":
                if c_spec.resource_id is None:
                    raise SimulationConfigurationError(
                        "NonNegative constraint requires 'resource_id'."
                    )
                registry.register(
                    ResourceNonNegativeConstraint(
                        resource_id=c_spec.resource_id,
                        severity=severity,
                        constraint_id=c_spec.constraint_id,
                    )
                )
            elif c_spec.type == "transfer_availability":
                registry.register(ActionTransferAvailabilityConstraint(severity=severity))
            else:
                raise SimulationConfigurationError(
                    f"Unknown constraint type in specification: '{c_spec.type}'"
                )

        return World(state=initial_state, constraints=registry)
