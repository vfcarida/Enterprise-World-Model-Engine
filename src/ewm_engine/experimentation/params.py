"""Parameter space specifications for Design of Experiments and parameter sweeps.

Conforms to Track T4: Core Substrate (Zero-dep).
Allows declaring continuous, integer, and categorical parameter spaces over
World, Scenario, and component configurations with deterministic sampling and
canonical cryptographic hashing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.world import World
from ewm_engine.simulation.scenario import Scenario

ParameterType = Literal["continuous", "integer", "categorical"]


class ParameterDef(BaseModel):
    """Specification of a single parameter dimension."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Parameter name / identifier.")
    type: ParameterType = Field(description="Variable type: continuous, integer, or categorical.")
    bounds: tuple[float, float] | None = Field(
        default=None,
        description="Numeric [lower, upper] interval for continuous and integer parameters.",
    )
    categories: tuple[Any, ...] | None = Field(
        default=None,
        description="Discrete set of permitted values for categorical parameters.",
    )
    default: Any | None = Field(
        default=None,
        description="Nominal baseline value.",
    )
    target: Literal[
        "resource",
        "memory",
        "context",
        "variable",
        "scenario_horizon",
        "scenario_attribute",
        "custom",
    ] = Field(
        default="resource",
        description="Location or semantics of the parameter within World/Scenario.",
    )
    target_id: str | None = Field(
        default=None,
        description="Specific entity/resource ID or dot-path within target.",
    )
    description: str = Field(default="", description="Human-readable parameter description.")

    @model_validator(mode="after")
    def _validate_bounds_or_categories(self) -> ParameterDef:
        if self.type in ("continuous", "integer"):
            if self.bounds is None or len(self.bounds) != 2:
                raise ValueError(
                    f"Parameter '{self.name}' of type '{self.type}' requires valid bounds (min, max)."
                )
            if self.bounds[0] > self.bounds[1]:
                raise ValueError(
                    f"Parameter '{self.name}' has inverted bounds: {self.bounds[0]} > {self.bounds[1]}."
                )
            if self.default is None:
                object.__setattr__(self, "default", float((self.bounds[0] + self.bounds[1]) / 2.0))
        elif self.type == "categorical":
            if not self.categories or len(self.categories) < 1:
                raise ValueError(
                    f"Categorical parameter '{self.name}' requires non-empty categories."
                )
            if self.default is None:
                object.__setattr__(self, "default", self.categories[0])
        return self


class ParameterSpace(BaseModel):
    """Multi-dimensional parameter search or experimentation space."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Schema version.")
    name: str = Field(default="ParameterSpace", description="Name of the parameter space.")
    parameters: tuple[ParameterDef, ...] = Field(
        description="Ordered sequence of parameter dimensions."
    )

    @property
    def dim(self) -> int:
        """Number of parameter dimensions."""
        return len(self.parameters)

    @property
    def names(self) -> list[str]:
        """Names of all parameter dimensions."""
        return [p.name for p in self.parameters]

    @property
    def space_hash(self) -> str:
        """Canonical SHA-256 hash of the parameter space definition."""
        normalized = {
            "name": self.name,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "bounds": list(p.bounds) if p.bounds is not None else None,
                    "categories": list(p.categories) if p.categories is not None else None,
                    "default": p.default,
                    "target": p.target,
                    "target_id": p.target_id,
                }
                for p in self.parameters
            ],
        }
        return canonical_sha256(normalized)

    def get_baseline_point(self) -> dict[str, Any]:
        """Return the default/nominal baseline parameter point."""
        return {p.name: p.default for p in self.parameters}

    def unit_to_point(self, unit_coords: Sequence[float] | np.ndarray) -> dict[str, Any]:
        """Transform coordinates in unit hypercube [0, 1]^D into a concrete parameter point."""
        if len(unit_coords) != self.dim:
            raise ValueError(f"Expected {self.dim} unit coordinates, got {len(unit_coords)}")

        point: dict[str, Any] = {}
        for p, u in zip(self.parameters, unit_coords, strict=True):
            u_clamped = max(0.0, min(1.0, float(u)))
            if p.type == "continuous":
                assert p.bounds is not None
                val = p.bounds[0] + u_clamped * (p.bounds[1] - p.bounds[0])
                point[p.name] = float(val)
            elif p.type == "integer":
                assert p.bounds is not None
                val = p.bounds[0] + u_clamped * (p.bounds[1] - p.bounds[0])
                point[p.name] = round(val)
            elif p.type == "categorical":
                assert p.categories is not None
                idx = min(int(u_clamped * len(p.categories)), len(p.categories) - 1)
                point[p.name] = p.categories[idx]
        return point

    def sample_random(self, rng: np.random.Generator) -> dict[str, Any]:
        """Uniformly sample a random parameter point using an explicit NumPy Generator."""
        u = rng.uniform(0.0, 1.0, size=self.dim)
        return self.unit_to_point(u)

    def apply_to_world_and_scenario(
        self,
        point: dict[str, Any],
        world: World,
        scenario: Scenario,
    ) -> tuple[World, Scenario]:
        """Apply parameter point values to create parameterized clones of World and Scenario."""
        curr_state = world.initial_state
        resources_dict = dict(curr_state.resources)
        memory_dict = dict(curr_state.memory)
        context_dict = dict(curr_state.context)
        scenario_kwargs: dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "horizon": scenario.horizon,
            "samples": scenario.samples,
            "seed": scenario.seed,
            "intervention": scenario.intervention,
            "scheduled_actions": scenario.scheduled_actions,
            "metadata": dict(scenario.metadata),
        }

        for p in self.parameters:
            if p.name not in point:
                continue
            val = point[p.name]
            target_key = p.target_id or p.name

            if p.target == "resource":
                if target_key in resources_dict:
                    res = resources_dict[target_key]
                    resources_dict[target_key] = res.model_copy(update={"current": float(val)})
            elif p.target in ("memory", "variable"):
                memory_dict[target_key] = val
            elif p.target == "context":
                context_dict[target_key] = val
            elif p.target == "scenario_horizon":
                scenario_kwargs["horizon"] = int(val)
            elif p.target == "scenario_attribute":
                scenario_kwargs[target_key] = val

        updated_state = curr_state.model_copy(
            update={"resources": resources_dict, "memory": memory_dict, "context": context_dict}
        )
        updated_world = World(
            initial_state=updated_state,
            dynamics=world.dynamics,
            constraints=world.constraints,
            event_sources=world.event_sources,
            actors=world.actors,
        )
        updated_scenario = Scenario(**scenario_kwargs)
        return updated_world, updated_scenario
