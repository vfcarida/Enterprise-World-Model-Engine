"""Core domain primitives for Enterprise World Model Engine (EWM Engine)."""

from __future__ import annotations

from ewm_engine.core._canonical import (
    canonical_sha256,
    fast_canonical_sha256,
    is_native_accel_available,
)
from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.deprecation import deprecate, deprecated
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.events import ExogenousEvent, ExogenousEventSource
from ewm_engine.core.resources import Resource
from ewm_engine.core.scm import (
    AbductionResult,
    ExogenousVariable,
    InvertibilityType,
    NonUniqueAbductionWarning,
    StructuralAssignment,
    StructuralCausalModel,
)
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
    INDUCTION_EVIDENCE_LADDER,
    BarePointEstimateWarning,
    DecisionBlockedError,
    ExtrapolationWarning,
    GateValidationReport,
    InductionRefusalError,
    InsufficientCausalEvidenceWarning,
    ResponsibleAIGate,
    ResponsibleAIWarning,
    ScenarioForecastResult,
    UncertaintyRequiredError,
    UnqualifiedCausalClaimError,
    WideUncertaintyWarning,
    is_evidence_at_least,
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
    "INDUCTION_EVIDENCE_LADDER",
    "AbductionResult",
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
    "ExogenousVariable",
    "ExtrapolationWarning",
    "GateValidationReport",
    "InductionRefusalError",
    "InsufficientCausalEvidenceWarning",
    "Intervention",
    "InvertibilityType",
    "NonUniqueAbductionWarning",
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
    "StructuralAssignment",
    "StructuralCausalModel",
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
    "canonical_sha256",
    "deprecate",
    "deprecated",
    "fast_canonical_sha256",
    "is_evidence_at_least",
    "is_native_accel_available",
    "validate_causal_claim",
]
