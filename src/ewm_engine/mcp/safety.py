"""Safety constraints, resource caps, and input validation models for the EWM MCP server.

Guarantees:
- Zero arbitrary code execution: no eval(), exec(), or custom YAML tags.
- Hard resource bounds (max Monte Carlo samples, max horizon, max entities).
- Callers are untrusted; all inputs pass through strict Pydantic schemas.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_HORIZON = 100
MAX_SAMPLES = 1000
MAX_ENTITIES = 500


class SimulateInput(BaseModel):
    """Input payload for ewm.simulate tool."""

    model_config = ConfigDict(extra="forbid")

    world_yaml: str = Field(description="Declarative World specification in YAML format.")
    horizon: int = Field(
        default=5,
        ge=1,
        le=MAX_HORIZON,
        description="Forward simulation discrete time steps (1 to 100).",
    )
    samples: int = Field(
        default=1,
        ge=1,
        le=MAX_SAMPLES,
        description="Number of Monte Carlo rollout sample paths (1 to 1000).",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Deterministic integer random seed for exact reproducibility.",
    )


class CounterfactualInput(BaseModel):
    """Input payload for ewm.counterfactual tool."""

    model_config = ConfigDict(extra="forbid")

    world_yaml: str = Field(description="Declarative World specification in YAML format.")
    intervention: dict[str, Any] = Field(
        description="Intervention definition specifying target entity/resource and do-action mutations."
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Deterministic integer random seed for coupled noise abduction.",
    )
    horizon: int = Field(
        default=5,
        ge=1,
        le=MAX_HORIZON,
        description="Simulation horizon steps.",
    )


class CheckConstraintsInput(BaseModel):
    """Input payload for ewm.check_constraints tool."""

    model_config = ConfigDict(extra="forbid")

    world_yaml: str = Field(description="Declarative World specification in YAML format.")
    action: dict[str, Any] = Field(
        description="Candidate action dictionary containing 'type', 'parameters', and optional 'actor_id'."
    )


class DecideInput(BaseModel):
    """Input payload for ewm.decide tool."""

    model_config = ConfigDict(extra="forbid")

    world_yaml: str = Field(description="Declarative World specification in YAML format.")
    candidates: dict[str, Any] = Field(
        description="Mapping of candidate action/policy names to action parameter definitions."
    )
    satisficing_threshold: float = Field(
        default=0.0,
        description="Satisficing boundary threshold for robustness evaluation.",
    )
    robustness_metric: Literal["regret", "satisficing", "domain"] = Field(
        default="regret",
        description="Robustness metric for ranking alternatives under uncertainty.",
    )
    horizon: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Simulation horizon steps (1 to 50).",
    )
    samples: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Monte Carlo samples per candidate (1 to 200).",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Deterministic integer random seed.",
    )


class ExplainInput(BaseModel):
    """Input payload for ewm.explain tool."""

    model_config = ConfigDict(extra="forbid")

    identifier: str = Field(
        description="Execution run_id, state fingerprint, or trace_id to inspect."
    )


class VerifyExplanationInput(BaseModel):
    """Input payload for ewm.verify_explanation tool."""

    model_config = ConfigDict(extra="forbid")

    trace_id: str | None = Field(
        default=None,
        description="Stored trace_id or run_id to verify against (optional if explanation_dict is provided).",
    )
    explanation_dict: dict[str, Any] | None = Field(
        default=None,
        description="Serialized VerifiableMechanisticExplanation dictionary when not looking up by trace_id.",
    )
    narrative_text: str = Field(
        min_length=1,
        max_length=10000,
        description="Candidate natural-language explanation text produced by an agent or reporting layer.",
    )
    claimed_mechanisms: list[str] = Field(
        default_factory=list,
        description="Optional explicit list of mechanism IDs claimed by the narrative.",
    )
    known_universe_mechanisms: list[str] = Field(
        default_factory=list,
        description="Optional universe of known domain mechanism IDs to audit for ungrounded mentions.",
    )


class CheckAbstentionInput(BaseModel):
    """Input payload for ewm.check_abstention tool."""

    model_config = ConfigDict(extra="forbid")

    aleatoric_std: float | None = Field(default=None, ge=0.0)
    parameter_std: float | None = Field(default=None, ge=0.0)
    structural_model_std: float | None = Field(default=None, ge=0.0)
    runtime_context_std: float | None = Field(default=None, ge=0.0)
    monte_carlo_se: float | None = Field(default=None, ge=0.0)
    model_discrepancy_vs_reality: float | None = Field(default=None, ge=0.0)
    max_parameter_std: float = Field(default=0.25, gt=0.0)
    max_structural_model_std: float = Field(default=0.25, gt=0.0)
    max_model_discrepancy: float = Field(default=0.30, gt=0.0)
    performative_invalidation: bool = Field(default=False)
    insufficient_causal_evidence: bool = Field(default=False)
