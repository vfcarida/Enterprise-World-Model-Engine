"""Programmatic runtime metrics collected across simulation rollouts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RunMetrics(BaseModel):
    """Programmatic runtime metrics collected during simulation execution.

    Tracks execution performance, rollout counts, step counts, constraint evaluations,
    and trace edges without exposing sensitive data or entity payloads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Total simulation wall-clock execution duration in seconds.",
    )
    rollout_count: int = Field(
        default=0,
        ge=0,
        description="Total number of Monte Carlo rollouts executed.",
    )
    completed_rollout_count: int = Field(
        default=0,
        ge=0,
        description="Number of rollouts that completed their full horizon without fatal invalidation.",
    )
    invalid_rollout_count: int = Field(
        default=0,
        ge=0,
        description="Number of rollouts invalidated by hard post-transition constraint violations.",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Total number of discrete simulation steps executed across all rollouts.",
    )
    constraint_evaluation_count: int = Field(
        default=0,
        ge=0,
        description="Total number of individual constraint evaluations performed (pre and post).",
    )
    hard_violation_count: int = Field(
        default=0,
        ge=0,
        description="Total count of hard constraint violations encountered.",
    )
    soft_violation_count: int = Field(
        default=0,
        ge=0,
        description="Total count of soft constraint violations encountered.",
    )
    dynamics_transition_count: int = Field(
        default=0,
        ge=0,
        description="Total number of state dynamics transitions performed across all steps.",
    )
    trace_edge_count: int = Field(
        default=0,
        ge=0,
        description="Total number of dependency edges added to systemic traces across all rollouts.",
    )


__all__ = ["RunMetrics"]
