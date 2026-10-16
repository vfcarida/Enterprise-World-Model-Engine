"""Safe declarative specification schema parser and trusted factory for World and WorldState.

Enforces the secure pipeline:
YAML text -> safe parser -> plain JSON data -> Pydantic validation -> WorldSpec -> trusted registry -> runtime objects.

Never executes arbitrary code, never dynamically imports from user data, and fails closed on unknown keys.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        schema: Any | None = None,
        as_of: Any | None = None,
        name: str = "InducedFromDataFrame",
        seed: int = 42,
    ) -> WorldSpec:
        """Induce a validated, point-in-time correct WorldSpec from a DataFrame or Arrow Table.

        Accepts pyarrow.Table, pandas.DataFrame, polars.DataFrame, or list of dicts.
        """
        from ewm_engine.induction.connectors import induce_from_dataframe

        spec, _report = induce_from_dataframe(
            df=df,
            schema=schema,
            as_of=as_of,
            name=name,
            seed=seed,
        )
        return spec

    @classmethod
    def from_sql(
        cls,
        query: str,
        con: Any,
        as_of: Any | None = None,
        schema: Any | None = None,
        name: str = "InducedFromSQL",
        seed: int = 42,
    ) -> WorldSpec:
        """Execute a SQL query with point-in-time correctness and induce a WorldSpec.

        Supports DuckDB, SQLite, SQLAlchemy, and Python DB-API 2.0 connections.
        """
        from ewm_engine.induction.connectors import induce_from_sql

        spec, _report = induce_from_sql(
            query=query,
            con=con,
            as_of=as_of,
            schema=schema,
            name=name,
            seed=seed,
        )
        return spec

    @classmethod
    def from_ocel(
        cls,
        ocel_data: str | Path | dict[str, Any],
        engine: Literal["native", "pm4py"] = "native",
        name: str = "InducedFromOCEL",
        seed: int = 42,
    ) -> WorldSpec:
        """Ingest an OCEL 2.0 event log and induce a validated WorldSpec.

        Uses Inductive Miner (soundness-guaranteed) to discover transitions,
        mines bounded-resource conservation laws, and assigns EvidenceLevels
        derived from replay alignment conformance checking.
        """
        from ewm_engine.induction.process import induce_process

        spec, _report = induce_process(
            ocel_data=ocel_data,
            engine=engine,
            name=name,
            seed=seed,
        )
        return spec

    @classmethod
    def from_causal(
        cls,
        data: Any,
        var_names: list[str] | None = None,
        interventional_data: dict[str, Any] | None = None,
        name: str = "InducedFromCausal",
        seed: int = 42,
    ) -> WorldSpec:
        """Induce a WorldSpec from observational and interventional data.

        Runs an ensemble of constraint-based (PC/FCI), score-based (DAGMA),
        and functional (DirectLiNGAM) discovery algorithms reconciled via DiBS
        Bayesian aggregation into an equivalence class.
        """
        from ewm_engine.induction.causal import CausalEdgeType, induce_causal

        equiv_class, _report = induce_causal(
            data=data,
            var_names=var_names,
            interventional_data=interventional_data,
            seed=seed,
        )
        entities = [
            EntitySpec(id=v, type="variable", attributes={"domain": "continuous"})
            for v in equiv_class.variables
        ]
        relationships = [
            RelationshipSpec(
                source=e.source,
                target=e.target,
                type=(
                    "directed_dependency"
                    if e.edge_type == CausalEdgeType.DIRECTED
                    else "associated_with"
                ),
                attributes={
                    "posterior_support": e.posterior_support,
                    "evidence_level": e.evidence_level.value,
                    "edge_type": e.edge_type.value,
                },
            )
            for e in equiv_class.edges
        ]
        return cls(
            world=WorldMetadataSpec(name=name),
            entities=entities,
            relationships=relationships,
        )

    @classmethod
    def from_dynamics(
        cls,
        trajectories: Any,
        target_resources: list[str] | None = None,
        degree: int = 2,
        threshold: float = 0.05,
        name: str = "InducedFromDynamics",
        seed: int = 42,
    ) -> WorldSpec:
        """Induce a WorldSpec with governing rate laws from state trajectories via SINDy."""
        from ewm_engine.induction.dynamics import induce_dynamics

        rate_laws, _report = induce_dynamics(
            trajectories_or_dataset=trajectories,
            target_resources=target_resources,
            degree=degree,
            threshold=threshold,
            seed=seed,
        )
        entities = [
            EntitySpec(
                id=r.target_variable,
                type="resource_entity",
                attributes={"domain": "continuous"},
            )
            for r in rate_laws
        ]
        resources = [
            ResourceSpec(
                id=r.target_variable,
                current=100.0,
                min_value=0.0,
                max_value=1e9,
                unit="units",
            )
            for r in rate_laws
        ]
        return cls(
            world=WorldMetadataSpec(name=name),
            entities=entities,
            resources=resources,
            dynamics=ComponentSpec(
                type="DeterministicDynamics",
                id="InducedDeterministicDynamics",
            ),
        )

    @classmethod
    def from_invariants(
        cls,
        trajectories: Any,
        resource_ids: list[str] | None = None,
        tolerance: float = 1e-3,
        mc_samples: int = 10,
        name: str = "InducedFromInvariants",
        seed: int = 42,
    ) -> WorldSpec:
        """Induce a WorldSpec with conservation laws verified via Monte Carlo stress testing."""
        from ewm_engine.induction.invariants import induce_invariants

        hypotheses, _report = induce_invariants(
            trajectories_or_dataset=trajectories,
            resource_ids=resource_ids,
            tolerance=tolerance,
            mc_samples=mc_samples,
            seed=seed,
        )
        res_names = (
            list(resource_ids)
            if resource_ids
            else (list(hypotheses[0].weights.keys()) if hypotheses else [])
        )
        entities = [
            EntitySpec(id=v, type="resource_entity", attributes={"domain": "continuous"})
            for v in res_names
        ]
        resources = [
            ResourceSpec(
                id=v,
                current=100.0,
                min_value=0.0,
                max_value=1e9,
                unit="units",
            )
            for v in res_names
        ]
        constraints = [h.to_constraint_spec() for h in hypotheses]
        return cls(
            world=WorldMetadataSpec(name=name),
            entities=entities,
            resources=resources,
            constraints=constraints,
        )

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

    def as_scm(self, horizon: int = 1) -> Any:
        """Compile an additive Structural Causal Model (SCM) view from this declarative specification."""
        from ewm_engine.core.scm import StructuralCausalModel

        return StructuralCausalModel.from_world_spec(self, horizon=horizon)

    def _repr_html_(self) -> str:
        """Render a rich, styled HTML card representation for interactive notebooks."""
        dyn_str = self.dynamics.type if self.dynamics else "None"
        constraints_str = f"{len(self.constraints)} active" if self.constraints else "None"
        events_str = f"{len(self.event_sources)} sources" if self.event_sources else "None"

        res_pills = "".join(
            f"<span style='background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 4px; padding: 2px 6px; font-size: 11px; margin: 2px 4px 2px 0; display: inline-block;'><code>{r.id}</code> ({r.current:.1f}/{r.max_value:.1f})</span>"
            for r in self.resources[:6]
        )
        if len(self.resources) > 6:
            res_pills += f"<span style='color: #64748b; font-size: 11px;'> +{len(self.resources) - 6} more</span>"
        none_span = "<span style='color: #94a3b8; font-size: 12px;'>None</span>"
        res_content = res_pills if self.resources else none_span

        return (
            f"<div style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; "
            f"border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; background: #ffffff; "
            f"box-shadow: 0 1px 3px rgba(0,0,0,0.05); max-width: 640px; margin: 8px 0; color: #1e293b;'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px;'>"
            f"<div>"
            f"<span style='font-weight: 700; font-size: 15px; color: #0f172a;'>WorldSpec</span>"
            f"<span style='background: #f0fdf4; color: #166534; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px; margin-left: 8px;'>v{self.schema_version}</span>"
            f"</div>"
            f"<span style='font-size: 13px; font-weight: 600; color: #334155;'>{self.world.name}</span>"
            f"</div>"
            f"<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 12px 0; font-size: 12px;'>"
            f"<div style='background: #f8fafc; padding: 8px; border-radius: 6px;'><strong>Entities:</strong> {len(self.entities)}<br><small style='color: #64748b;'>{len(self.relationships)} relations</small></div>"
            f"<div style='background: #f8fafc; padding: 8px; border-radius: 6px;'><strong>Dynamics:</strong> <code>{dyn_str}</code><br><small style='color: #64748b;'>{events_str}</small></div>"
            f"<div style='background: #f8fafc; padding: 8px; border-radius: 6px;'><strong>Constraints:</strong> {constraints_str}<br><small style='color: #64748b;'>Step: {self.world.step}</small></div>"
            f"</div>"
            f"<div style='margin-top: 8px;'><strong>Resources:</strong><div style='margin-top: 4px;'>{res_content}</div></div>"
            f"</div>"
        )


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

        def _make_leakage_dynamics(p: dict[str, Any]) -> DynamicsModel:
            from ewm_engine.dynamics.faults import UnmeteredLeakageDynamics

            return UnmeteredLeakageDynamics(
                resource_id=str(p.get("resource_id", "warehouse_a")),
                leak_rate=float(p.get("leak_rate", 15.0)),
                name=p.get("name", "UnmeteredLeakageDynamics"),
            )

        def _make_jitter_dynamics(p: dict[str, Any]) -> DynamicsModel:
            from ewm_engine.dynamics.faults import UnseededJitterDynamics

            return UnseededJitterDynamics(
                resource_id=str(p.get("resource_id", "warehouse_a")),
                name=p.get("name", "UnseededJitterDynamics"),
            )

        def _make_bias_dynamics(p: dict[str, Any]) -> DynamicsModel:
            from ewm_engine.dynamics.faults import PermutationBiasDynamics

            return PermutationBiasDynamics(
                resource_id=str(p.get("resource_id", "warehouse_a")),
                name=p.get("name", "PermutationBiasDynamics"),
            )

        def _make_fixed_transfer_dynamics(p: dict[str, Any]) -> DynamicsModel:
            from ewm_engine.dynamics.deterministic import DeterministicFixedTransferDynamics

            return DeterministicFixedTransferDynamics(
                source_resource=str(p.get("source_resource", "stock_a")),
                target_resource=str(p.get("target_resource", "stock_b")),
                quantity=float(p.get("quantity", 0.0)),
                name=p.get("name", "DeterministicFixedTransferDynamics"),
            )

        def _make_drift_dynamics(p: dict[str, Any]) -> DynamicsModel:
            from ewm_engine.dynamics.deterministic import DeterministicDriftDynamics

            return DeterministicDriftDynamics(
                resource_id=str(p.get("resource_id", "liquid_tgt")),
                drift=float(p.get("drift", 0.0)),
                name=p.get("name", "DeterministicDriftDynamics"),
            )

        self.register_dynamics("deterministic_transfer", _make_transfer_dynamics)
        self.register_dynamics("deterministictransferdynamics", _make_transfer_dynamics)
        self.register_dynamics("transfer", _make_transfer_dynamics)
        self.register_dynamics("fixed_transfer", _make_fixed_transfer_dynamics)
        self.register_dynamics("deterministic_fixed_transfer", _make_fixed_transfer_dynamics)
        self.register_dynamics("deterministicfixedtransferdynamics", _make_fixed_transfer_dynamics)
        self.register_dynamics("drift", _make_drift_dynamics)
        self.register_dynamics("deterministic_drift", _make_drift_dynamics)
        self.register_dynamics("deterministicdriftdynamics", _make_drift_dynamics)
        self.register_dynamics("deterministic", _make_deterministic_dynamics)
        self.register_dynamics("deterministicdynamics", _make_deterministic_dynamics)
        self.register_dynamics("deterministic_demand", _make_demand_dynamics)
        self.register_dynamics("deterministicdemanddynamics", _make_demand_dynamics)
        self.register_dynamics("demand", _make_demand_dynamics)
        self.register_dynamics("stochastic_demand", _make_stochastic_demand)
        self.register_dynamics("stochasticdemanddynamics", _make_stochastic_demand)
        self.register_dynamics("composite", _make_composite_dynamics)
        self.register_dynamics("compositedynamics", _make_composite_dynamics)
        self.register_dynamics("unmetered_leakage", _make_leakage_dynamics)
        self.register_dynamics("unmeteredleakagedynamics", _make_leakage_dynamics)
        self.register_dynamics("unseeded_jitter", _make_jitter_dynamics)
        self.register_dynamics("unseededjitterdynamics", _make_jitter_dynamics)
        self.register_dynamics("permutation_bias", _make_bias_dynamics)
        self.register_dynamics("permutationbiasdynamics", _make_bias_dynamics)

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
        self.register_constraint("resourcecapacityconstraint", _make_capacity_constraint)
        self.register_constraint("non_negative", _make_non_negative_constraint)
        self.register_constraint("resource_non_negative", _make_non_negative_constraint)
        self.register_constraint("resourcenonnegativeconstraint", _make_non_negative_constraint)
        self.register_constraint("transfer_availability", _make_transfer_avail_constraint)
        self.register_constraint("action_transfer_availability", _make_transfer_avail_constraint)
        self.register_constraint(
            "actiontransferavailabilityconstraint", _make_transfer_avail_constraint
        )


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
