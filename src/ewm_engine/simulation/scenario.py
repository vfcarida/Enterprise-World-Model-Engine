"""Scenario specification for simulation experiments and counterfactual evaluation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Intervention
from ewm_engine.core.types import ScenarioId


class Scenario(BaseModel):
    """Specification of an experimental simulation rollout.

    Defines the time horizon, Monte Carlo sample count, random seed,
    and optional counterfactual intervention.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="DefaultScenario", description="Human-readable scenario title.")
    scenario_id: ScenarioId = Field(
        default_factory=lambda: "scenario_run",
        description="Unique identifier for this scenario specification.",
    )
    horizon: int = Field(default=10, ge=1, description="Number of simulation steps to execute.")
    samples: int = Field(default=1, ge=1, description="Number of Monte Carlo rollout trajectories.")
    seed: int = Field(default=42, description="Master pseudo-random generator seed.")
    intervention: Intervention | None = Field(
        default=None,
        description="Optional policy or structural intervention to apply.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary experiment parameters and scenario notes.",
    )

    def with_intervention(self, intervention: Intervention) -> Scenario:
        """Create a new scenario instance with the specified intervention."""
        return self.model_copy(
            update={
                "intervention": intervention,
                "scenario_id": f"{self.scenario_id}_{intervention.id}",
            }
        )
