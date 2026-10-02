"""Safe declarative specification schema parser and trusted factory for World and WorldState.

Enforces the secure pipeline:
YAML text -> safe parser -> plain JSON data -> Pydantic validation -> WorldSpec -> trusted registry -> runtime objects.

Never executes arbitrary code, never dynamically imports from user data, and fails closed on unknown keys.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.events import ExogenousEventSource
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.exceptions import SimulationConfigurationError

if TYPE_CHECKING:
    from ewm_engine.constraints.base import Constraint
    from ewm_engine.dynamics.base import DynamicsModel


class ComponentSpec(BaseModel):
    """Specification schema for a modular runtime component (dynamics, constraint, or event source).

    Refers to components exclusively via registered type IDs and parameters, never importable paths.
    """

    model_config = ConfigDict(extra="forbid")

    type: str = Field(description="Registered type identifier in ComponentRegistry.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration parameters passed to the component factory.",
    )
    enabled: bool = Field(default=True, description="Whether this component is active.")
    id: str | None = Field(default=None, description="Optional unique identifier.")


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


class WorldMetadataSpec(BaseModel):
    """Metadata specification schema."""

    model_config = ConfigDict(extra="forbid")

    name: str = "EnterpriseWorld"
    description: str = ""
    step: int = 0
    timestamp: float = 0.0
    memory: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class WorldSpec(BaseModel):
    """Root declarative specification for an Enterprise World Model.

    Provides safe, non-executable serialization and deserialization from YAML and JSON.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Semantic schema version.")
    world: WorldMetadataSpec = Field(default_factory=WorldMetadataSpec)
    entities: list[EntitySpec] = Field(default_factory=list)
    relationships: list[RelationshipSpec] = Field(default_factory=list)
    resources: list[ResourceSpec] = Field(default_factory=list)
    dynamics: ComponentSpec | None = Field(
        default=None,
        description="Active dynamics model component specification.",
    )
    constraints: list[ComponentSpec] = Field(
        default_factory=list,
        description="Active constraint component specifications.",
    )
    event_sources: list[ComponentSpec] = Field(
        default_factory=list,
        description="Active exogenous event sources specifications.",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorldSpec:
        """Parse and validate a dictionary against the world specification schema."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> WorldSpec:
        """Parse world specification from a JSON string using safe deserialization."""
        from ewm_engine.serialization.json import canonical_loads

        data = canonical_loads(json_str)
        if not isinstance(data, dict):
            raise SimulationConfigurationError(
                f"JSON specification must be a dictionary, got {type(data).__name__}"
            )
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, yaml_input: str | Path) -> WorldSpec:
        """Safely parse world specification from a YAML string or file Path without code execution."""
        from ewm_engine.serialization.yaml import safe_load_yaml

        p = Path(yaml_input) if isinstance(yaml_input, (str, Path)) else None
        if p is not None and p.is_file():
            content = p.read_text(encoding="utf-8")
        else:
            content = str(yaml_input)

        data = safe_load_yaml(content)
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

    def build_world(self, factory: WorldFactory | None = None) -> World:
        """Compile the specification into an active, runnable World container."""
        wf = factory or WorldFactory()
        return wf.create_world(self)


# Backward-compatible aliases
WorldSpecification = WorldSpec
ConstraintSpec = ComponentSpec


class ComponentRegistry:
    """Trusted programmatic registry mapping component type IDs to factory callables.

    Prohibits arbitrary module importing and arbitrary code execution from user input.
    """

    def __init__(self) -> None:
        self._dynamics_factories: dict[str, Callable[[dict[str, Any]], DynamicsModel]] = {}
        self._constraint_factories: dict[str, Callable[[dict[str, Any]], Constraint]] = {}
        self._event_source_factories: dict[
            str, Callable[[dict[str, Any]], ExogenousEventSource]
        ] = {}

        self._register_default_components()

    def register_dynamics(
        self,
        type_id: str,
        factory: Callable[[dict[str, Any]], DynamicsModel],
    ) -> None:
        """Register a trusted dynamics factory callable."""
        self._dynamics_factories[type_id.strip().lower()] = factory

    def register_constraint(
        self,
        type_id: str,
        factory: Callable[[dict[str, Any]], Constraint],
    ) -> None:
        """Register a trusted constraint factory callable."""
        self._constraint_factories[type_id.strip().lower()] = factory

    def register_event_source(
        self,
        type_id: str,
        factory: Callable[[dict[str, Any]], ExogenousEventSource],
    ) -> None:
        """Register a trusted event source factory callable."""
        self._event_source_factories[type_id.strip().lower()] = factory

    def build_dynamics(self, type_id: str, parameters: dict[str, Any]) -> DynamicsModel:
        """Instantiate a dynamics model from registered type ID."""
        key = type_id.strip().lower()
        if key not in self._dynamics_factories:
            raise SimulationConfigurationError(
                f"Unknown dynamics component type: '{type_id}'. "
                f"Registered types: {sorted(self._dynamics_factories.keys())}"
            )
        return self._dynamics_factories[key](parameters)

    def build_constraint(self, type_id: str, parameters: dict[str, Any]) -> Constraint:
        """Instantiate a constraint from registered type ID."""
        key = type_id.strip().lower()
        if key not in self._constraint_factories:
            raise SimulationConfigurationError(
                f"Unknown constraint type: '{type_id}'. "
                f"Registered types: {sorted(self._constraint_factories.keys())}"
            )
        return self._constraint_factories[key](parameters)

    def build_event_source(self, type_id: str, parameters: dict[str, Any]) -> ExogenousEventSource:
        """Instantiate an event source from registered type ID."""
        key = type_id.strip().lower()
        if key not in self._event_source_factories:
            raise SimulationConfigurationError(
                f"Unknown event source component type: '{type_id}'. "
                f"Registered types: {sorted(self._event_source_factories.keys())}"
            )
        return self._event_source_factories[key](parameters)

    def _register_default_components(self) -> None:
        """Register standard out-of-the-box EWM Engine components."""
        from ewm_engine.constraints.results import ConstraintSeverity
        from ewm_engine.constraints.standard import (
            ActionTransferAvailabilityConstraint,
            ResourceCapacityConstraint,
            ResourceNonNegativeConstraint,
        )
        from ewm_engine.dynamics.composite import CompositeDynamics
        from ewm_engine.dynamics.deterministic import (
            DeterministicDemandDynamics,
            DeterministicDynamics,
            DeterministicTransferDynamics,
        )
        from ewm_engine.dynamics.stochastic import StochasticDemandDynamics

        # Dynamics
        def _make_transfer_dynamics(p: dict[str, Any]) -> DynamicsModel:
            return DeterministicTransferDynamics(
                action_type=p.get("action_type", "transfer_resource"),
                name=p.get("name", "TransferDynamics"),
            )

        def _make_deterministic_dynamics(p: dict[str, Any]) -> DynamicsModel:
            return DeterministicDynamics(name=p.get("name", "DeterministicDynamics"))

        def _make_demand_dynamics(p: dict[str, Any]) -> DynamicsModel:
            return DeterministicDemandDynamics(
                resource_id=str(p.get("resource_id", "stock")),
                demand=float(p.get("demand", 0.0)),
                name=p.get("name", "DeterministicDemandDynamics"),
            )

        def _make_stochastic_demand(p: dict[str, Any]) -> DynamicsModel:
            return StochasticDemandDynamics(
                resource_id=str(p.get("resource_id", "stock")),
                mean_demand=float(p.get("mean_demand", 0.0)),
                std_demand=float(p.get("std_demand", 0.0)),
                name=p.get("name", "StochasticDemandDynamics"),
            )

        def _make_composite_dynamics(p: dict[str, Any]) -> DynamicsModel:
            sub_specs = p.get("models", [])
            sub_models = [
                self.build_dynamics(m["type"], m.get("parameters", {})) for m in sub_specs
            ]
            return CompositeDynamics(models=sub_models, name=p.get("name", "CompositeDynamics"))

        self.register_dynamics("deterministic_transfer", _make_transfer_dynamics)
        self.register_dynamics("transfer", _make_transfer_dynamics)
        self.register_dynamics("deterministic", _make_deterministic_dynamics)
        self.register_dynamics("deterministic_demand", _make_demand_dynamics)
        self.register_dynamics("demand", _make_demand_dynamics)
        self.register_dynamics("stochastic_demand", _make_stochastic_demand)
        self.register_dynamics("composite", _make_composite_dynamics)

        # Constraints
        def _parse_severity(s: str | None) -> ConstraintSeverity:
            return (
                ConstraintSeverity.SOFT
                if (s or "hard").strip().lower() == "soft"
                else ConstraintSeverity.HARD
            )

        def _make_capacity_constraint(p: dict[str, Any]) -> Constraint:
            res_id = p.get("resource_id")
            if not res_id:
                raise SimulationConfigurationError("Capacity constraint requires 'resource_id'.")
            return ResourceCapacityConstraint(
                resource_id=str(res_id),
                severity=_parse_severity(p.get("severity")),
                constraint_id=p.get("constraint_id") or p.get("id"),
            )

        def _make_non_negative_constraint(p: dict[str, Any]) -> Constraint:
            res_id = p.get("resource_id")
            if not res_id:
                raise SimulationConfigurationError("NonNegative constraint requires 'resource_id'.")
            return ResourceNonNegativeConstraint(
                resource_id=str(res_id),
                severity=_parse_severity(p.get("severity")),
                constraint_id=p.get("constraint_id") or p.get("id"),
            )

        def _make_transfer_avail_constraint(p: dict[str, Any]) -> Constraint:
            cid = p.get("constraint_id") or p.get("id") or "transfer_stock_available"
            return ActionTransferAvailabilityConstraint(
                action_type=p.get("action_type", "transfer_resource"),
                constraint_id=str(cid),
                severity=_parse_severity(p.get("severity")),
            )

        self.register_constraint("capacity", _make_capacity_constraint)
        self.register_constraint("resource_capacity", _make_capacity_constraint)
        self.register_constraint("non_negative", _make_non_negative_constraint)
        self.register_constraint("resource_non_negative", _make_non_negative_constraint)
        self.register_constraint("transfer_availability", _make_transfer_avail_constraint)
        self.register_constraint("action_transfer_availability", _make_transfer_avail_constraint)


class WorldFactory:
    """Secure factory constructing executable World instances from WorldSpec.

    Ensures that all runtime components (dynamics, constraints, event sources)
    are constructed exclusively via trusted registrations.
    """

    def __init__(self, registry: ComponentRegistry | None = None) -> None:
        self.registry = registry or ComponentRegistry()

    def create_world(self, spec: WorldSpec) -> World:
        """Compile a validated WorldSpec into a runnable World container."""
        initial_state = spec.build_state()

        # Dynamics
        dynamics: DynamicsModel | None = None
        if spec.dynamics and spec.dynamics.enabled:
            dynamics = self.registry.build_dynamics(
                type_id=spec.dynamics.type,
                parameters=spec.dynamics.parameters,
            )

        # Constraints
        from ewm_engine.constraints.registry import ConstraintRegistry

        constraint_registry = ConstraintRegistry()
        for c_spec in spec.constraints:
            if not c_spec.enabled:
                continue
            constraint = self.registry.build_constraint(
                type_id=c_spec.type,
                parameters=c_spec.parameters,
            )
            constraint_registry.register(constraint)

        # Event sources
        event_sources: list[ExogenousEventSource] = []
        for es_spec in spec.event_sources:
            if not es_spec.enabled:
                continue
            event_source = self.registry.build_event_source(
                type_id=es_spec.type,
                parameters=es_spec.parameters,
            )
            event_sources.append(event_source)

        return World(
            state=initial_state,
            dynamics=dynamics,
            constraints=constraint_registry,
            event_sources=event_sources,
        )

    def create_from_yaml(self, yaml_str: str) -> World:
        """Safely parse YAML and build a validated World container."""
        spec = WorldSpec.from_yaml(yaml_str)
        return self.create_world(spec)

    def create_from_json(self, json_str: str) -> World:
        """Safely parse JSON and build a validated World container."""
        spec = WorldSpec.from_json(json_str)
        return self.create_world(spec)
