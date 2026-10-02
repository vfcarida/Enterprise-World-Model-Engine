"""Constraint evaluation results and provenance metadata."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.types import ActionId, ConstraintId, EntityId, ResourceId


class ConstraintSeverity(StrEnum):
    """Classification of constraint enforcement severity.

    - HARD: Inviolable physical, legal, or logistical invariant. Violations invalidate
            an action or abort/penalize a trajectory fatally.
    - SOFT: Operational preference, guideline, or non-fatal quality degradation.
            Violations yield warnings and numerical penalty scores.
    """

    HARD = "hard"
    SOFT = "soft"


class ConstraintPhase(StrEnum):
    """Execution lifecycle phase at which constraint evaluation is performed.

    - PRE_ACTION: Evaluated before action dispatch to validate operational feasibility.
    - POST_TRANSITION: Evaluated after dynamics transition to check state invariants.
    """

    PRE_ACTION = "pre_action"
    POST_TRANSITION = "post_transition"


class ConstraintResult(BaseModel):
    """Detailed audit record produced when evaluating an operational constraint.

    Preserves full constraint provenance:
      - Which constraint was evaluated
      - Its version and semantic definition
      - Whether it was satisfied
      - What entities, resources, and offending values caused the violation
      - What action preceded it
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    satisfied: bool = Field(description="True if constraint conditions are met.")
    constraint_id: ConstraintId = Field(description="Unique constraint identifier.")
    constraint_version: str = Field(default="1.0.0", description="Semantic version of constraint.")
    severity: ConstraintSeverity = Field(
        default=ConstraintSeverity.HARD,
        description="Hard (fatal/invariable) or Soft (penalty).",
    )
    phase: ConstraintPhase = Field(
        default=ConstraintPhase.POST_TRANSITION,
        description="Lifecycle evaluation phase (PRE_ACTION or POST_TRANSITION).",
    )
    message: str = Field(default="", description="Descriptive explanation of evaluation outcome.")
    violating_entities: tuple[EntityId, ...] = Field(
        default_factory=tuple,
        description="Identifiers of entities involved in constraint breach.",
    )
    violating_resources: tuple[ResourceId, ...] = Field(
        default_factory=tuple,
        description="Identifiers of resources involved in constraint breach.",
    )
    violating_values: dict[str, Any] = Field(
        default_factory=dict,
        description="Offending values (e.g. {'observed': 150, 'limit': 100}).",
    )
    step: int | None = Field(default=None, description="Simulation step when evaluated.")
    preceding_action_id: ActionId | None = Field(
        default=None,
        description="Identifier of preceding action that prompted evaluation.",
    )
    penalty: float = Field(default=0.0, description="Numerical penalty score for soft violations.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible audit metadata.",
    )
