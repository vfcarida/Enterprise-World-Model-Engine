"""Simulation metadata and cryptographic fingerprinting for exact reproducibility."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SimulationMetadata(BaseModel):
    """Immutable audit record and cryptographic fingerprint for simulation runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine_version: str = Field(default="0.1.0", description="EWM Engine semantic release version.")
    scenario_id: str = Field(description="Unique scenario run identifier.")
    world_hash: str = Field(description="SHA-256 hash of the initial baseline world state.")
    dynamics_name: str = Field(description="Dynamics model or composition applied.")
    constraint_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Map of active constraint IDs to their semantic versions.",
    )
    random_seed: int = Field(description="Master random seed for the simulation rollout.")
    horizon: int = Field(description="Temporal rollout horizon (number of steps).")
    samples: int = Field(description="Number of Monte Carlo rollout samples.")
    intervention_id: str | None = Field(
        default=None, description="Applied intervention ID if present."
    )
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO 8601 UTC execution timestamp.",
    )
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional experiment tracking metadata.",
    )

    @property
    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint capturing configuration, state, and seed."""
        fingerprint_dict = {
            "engine_version": self.engine_version,
            "scenario_id": self.scenario_id,
            "world_hash": self.world_hash,
            "dynamics_name": self.dynamics_name,
            "constraint_versions": sorted(self.constraint_versions.items()),
            "random_seed": self.random_seed,
            "horizon": self.horizon,
            "samples": self.samples,
            "intervention_id": self.intervention_id,
        }
        encoded = json.dumps(fingerprint_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
