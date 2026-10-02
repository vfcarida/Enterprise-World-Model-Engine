"""Scenario specification for simulation experiments and counterfactual evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ewm_engine.core.actions import Action, Intervention
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import ScenarioId


class ScheduledAction(BaseModel):
    """An action pre-scheduled to execute deterministically at a specific simulation time step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(ge=0, description="Simulation step at which action must be proposed.")
    action: Action = Field(description="Action to propose at this step.")


class Scenario(BaseModel):
    """Specification of an experimental simulation rollout.

    Defines the time horizon, Monte Carlo sample count, random seed,
    optional counterfactual intervention, pre-scheduled actions, and metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Semantic schema version.")
    name: str = Field(default="DefaultScenario", description="Human-readable scenario title.")
    scenario_id: ScenarioId = Field(
        default_factory=lambda: "scenario_run",
        description="Unique identifier for this scenario specification.",
    )
    initial_state: WorldState | None = Field(
        default=None,
        description="Optional baseline initial world state tied to this scenario.",
    )
    horizon: int = Field(default=10, ge=1, description="Number of simulation steps to execute.")
    samples: int = Field(default=1, ge=1, description="Number of Monte Carlo rollout trajectories.")
    seed: int = Field(default=42, description="Master pseudo-random generator seed.")
    intervention: Intervention | None = Field(
        default=None,
        description="Optional policy or structural intervention to apply.",
    )
    scheduled_actions: tuple[ScheduledAction, ...] = Field(
        default_factory=tuple,
        description="Deterministic pre-scheduled actions executed at specific time steps.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary experiment parameters and scenario notes.",
    )

    @field_validator("scheduled_actions", mode="before")
    @classmethod
    def _coerce_scheduled_actions(cls, v: Any) -> tuple[ScheduledAction, ...]:
        if isinstance(v, (list, tuple)):
            return tuple(
                item if isinstance(item, ScheduledAction) else ScheduledAction(**item) for item in v
            )
        return ()

    def with_intervention(self, intervention: Intervention) -> Scenario:
        """Create a new scenario instance with the specified intervention."""
        return self.branch(
            scenario_id=f"{self.scenario_id}_{intervention.id}",
            intervention=intervention,
        )

    def branch(
        self,
        *,
        scenario_id: str,
        name: str | None = None,
        scheduled_actions: Sequence[ScheduledAction] | None = None,
        seed: int | None = None,
        intervention: Intervention | None = None,
        metadata: Mapping[str, Any] | None = None,
        horizon: int | None = None,
        samples: int | None = None,
    ) -> Scenario:
        """Create an independent counterfactual scenario branch without mutating the parent.

        Preserves the initial_state (and its canonical fingerprint) while allowing
        alternative interventions, scheduled actions, seeds, or parameters.
        """
        updates: dict[str, Any] = {"scenario_id": scenario_id}
        if name is not None:
            updates["name"] = name
        if scheduled_actions is not None:
            updates["scheduled_actions"] = tuple(scheduled_actions)
        if seed is not None:
            updates["seed"] = seed
        if intervention is not None:
            updates["intervention"] = intervention
        if metadata is not None:
            merged_meta = dict(self.metadata)
            merged_meta.update(metadata)
            updates["metadata"] = merged_meta
        if horizon is not None:
            updates["horizon"] = horizon
        if samples is not None:
            updates["samples"] = samples

        return self.model_copy(update=updates)

    @property
    def fingerprint(self) -> str:
        """Deterministic canonical SHA-256 fingerprint capturing scenario specification."""
        from ewm_engine.core._canonical import canonical_sha256

        scenario_dict = {
            "name": self.name,
            "scenario_id": self.scenario_id,
            "horizon": self.horizon,
            "samples": self.samples,
            "seed": self.seed,
            "initial_state_fingerprint": self.initial_state.fingerprint
            if self.initial_state is not None
            else None,
            "intervention": {
                "id": self.intervention.id,
                "description": self.intervention.description,
                "parameters": self.intervention.parameters,
            }
            if self.intervention
            else None,
            "scheduled_actions": [
                {"step": sa.step, "action": sa.action.model_dump(mode="json")}
                for sa in sorted(self.scheduled_actions, key=lambda x: (x.step, str(x.action.id)))
            ],
            "metadata": self.metadata,
        }
        return canonical_sha256(scenario_dict)


__all__ = ["Scenario", "ScheduledAction"]
