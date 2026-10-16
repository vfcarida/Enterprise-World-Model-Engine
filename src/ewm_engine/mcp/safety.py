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
