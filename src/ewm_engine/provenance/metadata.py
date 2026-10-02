"""Provenance and cryptographic audit models for exact simulation reproducibility."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ewm_engine.core._canonical import canonical_sha256


class ComponentVersion(BaseModel):
    """Specification of a participating simulation component and its semantic version."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component_id: str = Field(description="Unique identifier or class name of the component.")
    component_version: str = Field(default="1.0.0", description="Semantic version string.")


class Provenance(BaseModel):
    """Immutable audit record and cryptographic provenance for simulation rollouts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Semantic schema version.")
    engine_version: str = Field(default="0.1.0", description="EWM Engine semantic release version.")
    scenario_fingerprint: str = Field(
        description="Deterministic canonical SHA-256 fingerprint of the scenario configuration."
    )
    initial_state_fingerprint: str = Field(
        description="Deterministic canonical SHA-256 fingerprint of the initial baseline world state."
    )
    seed: int = Field(description="Master random seed for the simulation rollout.")
    horizon: int = Field(description="Temporal rollout horizon (number of steps).")
    samples: int = Field(description="Number of Monte Carlo rollout samples.")
    components: tuple[ComponentVersion, ...] = Field(
        default_factory=tuple,
        description="Active participating engine, dynamics, and event components with versions.",
    )
    constraint_versions: tuple[ComponentVersion, ...] = Field(
        default_factory=tuple,
        description="Active constraint rules and their semantic versions.",
    )
    runtime_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional execution, host, or experiment tracking metadata.",
    )

    def __init__(
        self,
        scenario_fingerprint: str = "",
        initial_state_fingerprint: str = "",
        seed: int = 42,
        horizon: int = 10,
        samples: int = 1,
        components: Sequence[ComponentVersion | tuple[str, str] | dict[str, Any]]
        | Mapping[str, Any] = (),
        constraint_versions: Sequence[ComponentVersion | tuple[str, str] | dict[str, Any]]
        | Mapping[str, Any] = (),
        runtime_metadata: dict[str, Any] | None = None,
        engine_version: str = "0.1.0",
        schema_version: str = "1.0.0",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            schema_version=schema_version,
            engine_version=engine_version,
            scenario_fingerprint=scenario_fingerprint,
            initial_state_fingerprint=initial_state_fingerprint,
            seed=seed,
            horizon=horizon,
            samples=samples,
            components=components,
            constraint_versions=constraint_versions,
            runtime_metadata=copy.deepcopy(dict(runtime_metadata or {})),
            **{k: copy.deepcopy(v) for k, v in kwargs.items()},
        )

    def __getattribute__(self, name: str) -> Any:
        val = super().__getattribute__(name)
        if name == "runtime_metadata" and isinstance(val, dict):
            return copy.deepcopy(val)
        return val

    @field_validator("components", mode="before")
    @classmethod
    def _coerce_components(cls, v: Any) -> tuple[ComponentVersion, ...]:
        if isinstance(v, Mapping):
            return tuple(
                ComponentVersion(component_id=str(k), component_version=str(val))
                for k, val in sorted(v.items())
            )
        if isinstance(v, (list, tuple)):
            res: list[ComponentVersion] = []
            for item in v:
                if isinstance(item, ComponentVersion):
                    res.append(item)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    res.append(
                        ComponentVersion(component_id=str(item[0]), component_version=str(item[1]))
                    )
                elif isinstance(item, dict):
                    res.append(ComponentVersion(**item))
            return tuple(res)
        return ()

    @field_validator("constraint_versions", mode="before")
    @classmethod
    def _coerce_constraint_versions(cls, v: Any) -> tuple[ComponentVersion, ...]:
        if isinstance(v, Mapping):
            return tuple(
                ComponentVersion(component_id=str(k), component_version=str(val))
                for k, val in sorted(v.items())
            )
        if isinstance(v, (list, tuple)):
            res: list[ComponentVersion] = []
            for item in v:
                if isinstance(item, ComponentVersion):
                    res.append(item)
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    res.append(
                        ComponentVersion(component_id=str(item[0]), component_version=str(item[1]))
                    )
                elif isinstance(item, dict):
                    res.append(ComponentVersion(**item))
            return tuple(res)
        return ()

    @property
    def fingerprint(self) -> str:
        """Deterministic canonical SHA-256 fingerprint capturing configuration, state, and seed."""
        canonical_dict = {
            "engine_version": self.engine_version,
            "scenario_fingerprint": self.scenario_fingerprint,
            "initial_state_fingerprint": self.initial_state_fingerprint,
            "seed": self.seed,
            "horizon": self.horizon,
            "samples": self.samples,
            "components": sorted(
                [{"id": c.component_id, "version": c.component_version} for c in self.components],
                key=lambda x: str(x["id"]),
            ),
            "constraint_versions": sorted(
                [
                    {"id": c.component_id, "version": c.component_version}
                    for c in self.constraint_versions
                ],
                key=lambda x: str(x["id"]),
            ),
        }
        return canonical_sha256(canonical_dict)

    # Backward-compatibility property aliases
    @property
    def world_hash(self) -> str:
        """Backward-compatible alias for initial_state_fingerprint."""
        return self.initial_state_fingerprint

    @property
    def random_seed(self) -> int:
        """Backward-compatible alias for seed."""
        return self.seed

    @property
    def scenario_id(self) -> str:
        """Backward-compatible access to scenario identifier."""
        return str(self.runtime_metadata.get("scenario_id", self.scenario_fingerprint))

    @property
    def custom_metadata(self) -> dict[str, Any]:
        """Backward-compatible alias for runtime_metadata."""
        return self.runtime_metadata


class SimulationMetadata(Provenance):
    """Internal and backward-compatible alias for Provenance supporting legacy kwargs."""

    def __init__(
        self,
        scenario_id: str | None = None,
        world_hash: str | None = None,
        dynamics_name: str | None = None,
        constraint_versions: Sequence[Any] | Mapping[str, Any] = (),
        random_seed: int | None = None,
        horizon: int = 10,
        samples: int = 1,
        intervention_id: str | None = None,
        timestamp_utc: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        scenario_fingerprint: str | None = None,
        initial_state_fingerprint: str | None = None,
        seed: int | None = None,
        components: Sequence[Any] | Mapping[str, Any] = (),
        runtime_metadata: dict[str, Any] | None = None,
        engine_version: str = "0.1.0",
        **kwargs: Any,
    ) -> None:
        init_state_fp = initial_state_fingerprint or world_hash or ""
        scen_fp = scenario_fingerprint or (scenario_id if scenario_id else "unknown_scenario")
        actual_seed = seed if seed is not None else (random_seed if random_seed is not None else 42)

        meta = dict(runtime_metadata or custom_metadata or {})
        if scenario_id:
            meta["scenario_id"] = scenario_id
        if dynamics_name:
            meta["dynamics_name"] = dynamics_name
        if intervention_id:
            meta["intervention_id"] = intervention_id
        if timestamp_utc:
            meta["timestamp_utc"] = timestamp_utc

        actual_components = list(components)
        if dynamics_name and not any(
            (c.component_id == dynamics_name if isinstance(c, ComponentVersion) else False)
            for c in actual_components
        ):
            actual_components.append(
                ComponentVersion(component_id=dynamics_name, component_version="1.0.0")
            )

        super().__init__(
            engine_version=engine_version,
            scenario_fingerprint=scen_fp,
            initial_state_fingerprint=init_state_fp,
            seed=actual_seed,
            horizon=horizon,
            samples=samples,
            components=actual_components,
            constraint_versions=constraint_versions,
            runtime_metadata=meta,
            **kwargs,
        )
