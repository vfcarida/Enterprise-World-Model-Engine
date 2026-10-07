"""Core domain primitives for Enterprise World Model Engine (EWM Engine)."""

from __future__ import annotations

from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.deprecation import deprecate, deprecated
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.events import ExogenousEvent, ExogenousEventSource
from ewm_engine.core.resources import Resource
from ewm_engine.core.spec import (
    ComponentRegistry,
    ComponentSpec,
    ConstraintSpec,
    EntitySpec,
    RelationshipSpec,
    ResourceSpec,
    WorldFactory,
    WorldMetadataSpec,
    WorldSpec,
    WorldSpecification,
)
from ewm_engine.core.state import WorldState
from ewm_engine.core.trust import (
    BarePointEstimateWarning,
    DecisionBlockedError,
    ExtrapolationWarning,
    GateValidationReport,
    InsufficientCausalEvidenceWarning,
    ResponsibleAIGate,
    ResponsibleAIWarning,
    ScenarioForecastResult,
    UncertaintyRequiredError,
    UnqualifiedCausalClaimError,
    WideUncertaintyWarning,
    validate_causal_claim,
)
from ewm_engine.core.types import (
    ActionId,
    ActorId,
    ConstraintId,
    EntityId,
    EventId,
    RandomGenerator,
    RelationshipId,
    ResourceId,
    ScenarioId,
    TrajectoryId,
)
from ewm_engine.core.world import World

__all__ = [
    "Action",
    "ActionId",
    "ActorId",
    "BarePointEstimateWarning",
    "ComponentRegistry",
    "ComponentSpec",
    "ConstraintId",
    "ConstraintSpec",
    "DecisionBlockedError",
    "Entity",
    "EntityId",
    "EntitySpec",
    "EventId",
    "ExogenousEvent",
    "ExogenousEventSource",
    "ExtrapolationWarning",
    "GateValidationReport",
    "InsufficientCausalEvidenceWarning",
    "Intervention",
    "RandomGenerator",
    "Relationship",
    "RelationshipId",
    "RelationshipSpec",
    "Resource",
    "ResourceId",
    "ResourceSpec",
    "ResponsibleAIGate",
    "ResponsibleAIWarning",
    "ScenarioForecastResult",
    "ScenarioId",
    "TrajectoryId",
    "UncertaintyRequiredError",
    "UnqualifiedCausalClaimError",
    "WideUncertaintyWarning",
    "World",
    "WorldFactory",
    "WorldMetadataSpec",
    "WorldSpec",
    "WorldSpecification",
    "WorldState",
    "deprecate",
    "deprecated",
    "validate_causal_claim",
]
