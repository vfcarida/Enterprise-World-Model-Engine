"""Safe declarative World Specification Language (WSL) implementation (v2.0 candidate).

Formalizes a versioned, declarative format for enterprise world models.
Conforms to ADR-023:
- Strictly declarative data (no eval, no exec, no expression languages).
- Rejects custom YAML tags and pickle opcodes via StrictSafeLoader.
- Validates strictly through Pydantic (extra="forbid").
- Resolves executable components through trusted programmatic ComponentRegistry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.spec import ComponentRegistry
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.exceptions import (
    SerializationError,
)
from ewm_engine.serialization.yaml import safe_dump_yaml, safe_load_yaml


class WSLMetadata(BaseModel):
    """Metadata describing the enterprise world model specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default="enterprise_world", description="Unique world model identifier.")
    name: str = Field(default="Enterprise World", description="Human-readable model name.")
    version: str = Field(default="2.0.0", description="Model semantic version.")
    description: str = Field(default="", description="High-level narrative description.")
    author: str = Field(default="", description="Author or organization name.")
    tags: list[str] = Field(default_factory=list, description="Categorical classification tags.")


class WSLTemporalConfig(BaseModel):
    """Temporal horizon and stepping configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    time_unit: str = Field(default="step", description="Logical time unit (e.g., 'step', 'day').")
    step_duration: float = Field(default=1.0, description="Duration per step in time units.")
    default_horizon: int = Field(default=10, description="Default rollout steps.")


class WSLEntitySpec(BaseModel):
    """Declarative specification of a participating entity node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Unique entity identifier.")
    type: str = Field(description="Categorical entity classification.")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Static attributes.")
    tags: list[str] = Field(default_factory=list, description="Categorical tags.")
    temporal_valid_from: float | None = Field(
        default=None,
        description="Timestamp when entity becomes active.",
    )
    temporal_valid_until: float | None = Field(
        default=None,
        description="Timestamp when entity expires.",
    )


class WSLRelationshipSpec(BaseModel):
    """Declarative specification of a directed relational edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(description="Source entity ID.")
    target: str = Field(description="Target entity ID.")
    type: str = Field(description="Relational edge type.")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Edge attributes.")
    weight: float = Field(default=1.0, description="Scalar relational weight.")
    temporal_valid_from: float | None = Field(
        default=None,
        description="Timestamp when edge becomes active.",
    )
    temporal_valid_until: float | None = Field(
        default=None,
        description="Timestamp when edge expires.",
    )


class WSLResourceSpec(BaseModel):
    """Declarative specification of a bounded state resource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Unique resource identifier.")
    current: float = Field(description="Initial numeric resource level.")
    min_value: float = Field(default=0.0, description="Hard lower bound.")
    max_value: float = Field(default=1e9, description="Hard upper bound.")
    unit: str = Field(default="", description="Measurement unit.")
    entity_id: str | None = Field(
        default=None,
        description="Optional entity node this resource attaches to.",
    )


class WSLComponentSpec(BaseModel):
    """Reference to a registered executable component resolved via ComponentRegistry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type_id: str = Field(description="Registered component type identifier.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Initialization parameters passed to component factory.",
    )
    evidence_level: str = Field(
        default="ASSUMED",
        description="Declared epistemic evidence level.",
    )


class WSLConstraintSpec(BaseModel):
    """Declarative constraint specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Unique constraint identifier.")
    type_id: str = Field(description="Registered constraint type identifier.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Constraint parameters.")
    phase: str = Field(default="POST_DYNAMICS", description="Evaluation phase.")
    severity: str = Field(default="HARD", description="Constraint severity: 'HARD' or 'SOFT'.")


class WSLScenarioSpec(BaseModel):
    """Declarative simulation scenario specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Unique scenario identifier.")
    name: str = Field(default="", description="Scenario title.")
    seed: int = Field(default=42, description="Pseudorandom generator seed.")
    horizon: int = Field(default=10, description="Simulation step count.")
    interventions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Scheduled interventions.",
    )


class WSLDocument(BaseModel):
    """Root World Specification Language document (v2.0 grammar)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default="wsl/2.0.0",
        description="WSL schema version string.",
    )
    metadata: WSLMetadata = Field(default_factory=WSLMetadata)
    temporal: WSLTemporalConfig = Field(default_factory=WSLTemporalConfig)
    entities: list[WSLEntitySpec] = Field(default_factory=list)
    relationships: list[WSLRelationshipSpec] = Field(default_factory=list)
    resources: list[WSLResourceSpec] = Field(default_factory=list)
    dynamics: WSLComponentSpec | None = Field(
        default=None,
        description="Primary dynamics model component.",
    )
    constraints: list[WSLConstraintSpec] = Field(default_factory=list)
    event_sources: list[WSLComponentSpec] = Field(default_factory=list)
    scenarios: list[WSLScenarioSpec] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, v: str) -> str:
        if v not in ("wsl/2.0.0", "2.0.0", "wsl/2.0"):
            raise ValueError(
                f"Unsupported WSL schema_version '{v}'. Expected 'wsl/2.0.0' or '2.0.0'."
            )
        return v


def parse_wsl_yaml(yaml_str: str) -> WSLDocument:
    """Parse a YAML string into a strictly validated WSLDocument.

    Security guarantees:
    - Rejects any custom YAML tags (e.g. !python/object) via StrictSafeLoader.
    - Rejects pickle byte sequences.
    - Validates through Pydantic with extra='forbid'.
    """
    data = safe_load_yaml(yaml_str)
    if not isinstance(data, dict):
        raise SerializationError(f"WSL document must be a YAML mapping, got {type(data).__name__}")
    return WSLDocument.model_validate(data)


def parse_wsl_file(path: str | Path) -> WSLDocument:
    """Safely parse a WSL file from disk."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"WSL file not found: {p}")
    return parse_wsl_yaml(p.read_text(encoding="utf-8"))


def dump_wsl_yaml(doc: WSLDocument) -> str:
    """Serialize a WSLDocument to safe canonical YAML."""
    return safe_dump_yaml(doc)


def compile_wsl(doc: WSLDocument, registry: ComponentRegistry | None = None) -> World:
    """Compile a validated WSLDocument into an executable World container.

    Resolves dynamics models and constraints through the trusted ComponentRegistry.
    """
    reg = registry if registry is not None else ComponentRegistry()

    # 1. Build entities
    entities: list[Entity] = []
    for e in doc.entities:
        attrs = dict(e.attributes)
        if e.temporal_valid_from is not None:
            attrs["temporal_valid_from"] = e.temporal_valid_from
        if e.temporal_valid_until is not None:
            attrs["temporal_valid_until"] = e.temporal_valid_until
        entities.append(Entity(id=e.id, type=e.type, attributes=attrs, tags=tuple(e.tags)))

    # 2. Build relationships
    relationships: list[Relationship] = []
    for r in doc.relationships:
        attrs = dict(r.attributes)
        attrs["weight"] = r.weight
        if r.temporal_valid_from is not None:
            attrs["temporal_valid_from"] = r.temporal_valid_from
        if r.temporal_valid_until is not None:
            attrs["temporal_valid_until"] = r.temporal_valid_until
        relationships.append(
            Relationship(source=r.source, target=r.target, type=r.type, attributes=attrs)
        )

    # 3. Build resources
    resources: list[Resource] = [
        Resource(
            id=res.id,
            current=res.current,
            min_value=res.min_value,
            max_value=res.max_value,
            unit=res.unit,
            entity_id=res.entity_id,
        )
        for res in doc.resources
    ]

    initial_state = WorldState(
        step=0,
        timestamp=0.0,
        schema_version="2.0.0",
        entities=entities,
        relationships=relationships,
        resources=resources,
    )

    # 4. Resolve dynamics
    if doc.dynamics is not None:
        dynamics_model = reg.build_dynamics(doc.dynamics.type_id, doc.dynamics.parameters)
    else:
        # Default pass-through dynamics if none specified
        from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics

        dynamics_model = DeterministicTransferDynamics()

    world = World(initial_state=initial_state, dynamics=dynamics_model)

    # 5. Resolve constraints
    for c_spec in doc.constraints:
        c_instance = reg.build_constraint(c_spec.type_id, c_spec.parameters)
        world.add_constraint(c_instance)

    return world


def export_wsl(world: World, metadata: WSLMetadata | None = None) -> WSLDocument:
    """Export an active World container to a declarative WSLDocument."""
    state = world.current_state

    entities: list[WSLEntitySpec] = []
    for e in state.entities.values():
        attrs = dict(e.attributes)
        t_from = attrs.pop("temporal_valid_from", None)
        t_until = attrs.pop("temporal_valid_until", None)
        entities.append(
            WSLEntitySpec(
                id=e.id,
                type=e.type,
                attributes=attrs,
                tags=list(e.tags),
                temporal_valid_from=t_from,
                temporal_valid_until=t_until,
            )
        )

    relationships: list[WSLRelationshipSpec] = []
    for r in state.relationships:
        attrs = dict(r.attributes)
        weight = float(attrs.pop("weight", 1.0))
        t_from = attrs.pop("temporal_valid_from", None)
        t_until = attrs.pop("temporal_valid_until", None)
        relationships.append(
            WSLRelationshipSpec(
                source=r.source,
                target=r.target,
                type=r.type,
                attributes=attrs,
                weight=weight,
                temporal_valid_from=t_from,
                temporal_valid_until=t_until,
            )
        )

    resources: list[WSLResourceSpec] = [
        WSLResourceSpec(
            id=res.id,
            current=res.current,
            min_value=res.min_value,
            max_value=res.max_value,
            unit=res.unit,
            entity_id=res.entity_id,
        )
        for res in state.resources.values()
    ]

    return WSLDocument(
        schema_version="wsl/2.0.0",
        metadata=metadata or WSLMetadata(),
        temporal=WSLTemporalConfig(),
        entities=entities,
        relationships=relationships,
        resources=resources,
        dynamics=None,
        constraints=[],
    )
