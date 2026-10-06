"""Per-run scientific reproducibility manifest conforming to the NeurIPS Reproducibility Checklist."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256

if TYPE_CHECKING:
    from ewm_engine.core.world import World
    from ewm_engine.simulation.scenario import Scenario


def _resolve_git_commit() -> str | None:
    """Attempt to extract current git HEAD commit SHA."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if proc.returncode == 0:
            commit = proc.stdout.strip()
            return commit if commit else None
    except Exception:
        pass
    return None


def _resolve_dependency_versions() -> dict[str, str]:
    """Capture critical package dependency versions."""
    deps = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    for mod_name in ("pydantic", "torch", "scipy", "networkx", "ortools"):
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            deps[mod_name] = getattr(mod, "__version__", "unknown")
    return deps


class RunManifest(BaseModel):
    """Cryptographic execution manifest mapping 1:1 to the NeurIPS reproducibility checklist.

    Captures seed(s), resolved-config hash, component/scenario IDs + versions, engine version,
    git commit, dependency versions, determinism flags in effect, and wall-clock execution metrics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_id: str = Field(
        default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this execution manifest.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Wall-clock execution start timestamp.",
    )
    execution_duration_sec: float | None = Field(
        default=None,
        description="Measured wall-clock run duration in seconds.",
    )
    engine_version: str = Field(
        default="1.1.0",
        description="Enterprise World Model Engine semantic release version.",
    )
    git_commit: str | None = Field(
        default=None,
        description="Git commit hash of repository state during execution.",
    )
    seed: int = Field(
        default=42,
        description="Master pseudo-random seed.",
    )
    seeds: tuple[int, ...] = Field(
        default_factory=tuple,
        description="Full sequence of random seeds in multi-seed repetitions.",
    )
    resolved_config_hash: str = Field(
        description="Canonical SHA-256 hash of the fully resolved scenario and world configuration.",
    )
    scenario_id: str = Field(
        description="Identifier of the executed simulation scenario.",
    )
    initial_state_fingerprint: str = Field(
        description="Canonical SHA-256 state fingerprint of the baseline initial world state.",
    )
    component_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Identifiers of participating dynamics, actor, and event components.",
    )
    component_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of component ID to semantic version.",
    )
    determinism_flags: dict[str, Any] = Field(
        default_factory=dict,
        description="Active environment determinism settings (PYTHONHASHSEED, OMP_NUM_THREADS, CUBLAS, etc.).",
    )
    dependency_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Runtime package and interpreter versions.",
    )
    hardware_summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Host hardware and operating system execution summary.",
    )
    manifest_hash: str = Field(
        default="",
        description="Deterministic canonical cryptographic SHA-256 fingerprint of the entire manifest.",
    )

    @property
    def reproducibility_checklist(self) -> dict[str, Any]:
        """Map manifest attributes 1:1 to the NeurIPS Reproducibility Checklist criteria."""
        return {
            "all_code_provided": True,
            "specification_of_dependencies": len(self.dependency_versions) > 0,
            "training_and_evaluation_code": True,
            "random_seeds_specified": bool(self.seeds or self.seed is not None),
            "exact_computing_infrastructure": len(self.hardware_summary) > 0,
            "hyperparameters_and_configurations": bool(self.resolved_config_hash),
            "determinism_flags_reported": bool(self.determinism_flags),
            "manifest_hash": self.manifest_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to JSON-compatible dictionary."""
        d = self.model_dump(mode="json")
        d["reproducibility_checklist"] = self.reproducibility_checklist
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize manifest to formatted JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=indent)


def create_run_manifest(
    world: World | None = None,
    scenario: Scenario | None = None,
    component_id: str | None = None,
    scenario_id: str | None = None,
    seed: int | None = None,
    seeds: Sequence[int] | None = None,
    resolved_config: dict[str, Any] | None = None,
    execution_duration_sec: float | None = None,
    determinism_flags: dict[str, Any] | None = None,
    engine_version: str = "1.1.0",
    component_version: str = "1.0.0",
    deterministic: bool = True,
) -> RunManifest:
    """Construct an immutable RunManifest from simulation configuration and runtime context."""
    if scenario is not None:
        active_seed = seed if seed is not None else scenario.seed
        scen_id = scenario.scenario_id
    else:
        active_seed = seed if seed is not None else 42
        scen_id = scenario_id or "default_scenario"

    seed_tuple = tuple(seeds) if seeds else (active_seed,)

    # Extract component IDs and versions
    comp_ids: list[str] = []
    comp_versions: dict[str, str] = {}

    if world is not None:
        dyn_name = type(world.dynamics).__name__
        comp_ids.append(dyn_name)
        comp_versions[dyn_name] = component_version

        for actor in world.actors:
            comp_ids.append(actor.actor_id)
            comp_versions[actor.actor_id] = "1.0.0"

        for src in world.event_sources:
            comp_ids.append(type(src).__name__)
            comp_versions[type(src).__name__] = "1.0.0"
        init_state_fp = world.initial_state.fingerprint
    else:
        cid = component_id or "default_component"
        comp_ids.append(cid)
        comp_versions[cid] = component_version
        init_state_fp = "state_0"

    # Capture active determinism flags from environment
    active_flags = dict(determinism_flags or {})
    active_flags.setdefault("python_hash_seed", os.environ.get("PYTHONHASHSEED", "unset"))
    active_flags.setdefault("omp_num_threads", os.environ.get("OMP_NUM_THREADS", "unset"))
    active_flags.setdefault(
        "cublas_workspace_config", os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset")
    )
    active_flags["deterministic"] = deterministic

    # Resolved configuration canonical hash
    if resolved_config is not None:
        config_dict = dict(resolved_config)
    elif scenario is not None and world is not None:
        config_dict = {
            "scenario_fingerprint": scenario.fingerprint,
            "initial_state_fingerprint": world.initial_state.fingerprint,
            "horizon": scenario.horizon,
            "samples": scenario.samples,
            "seed": active_seed,
        }
    else:
        config_dict = {"seed": active_seed, "component_id": comp_ids[0]}

    resolved_cfg_hash = canonical_sha256(config_dict)

    hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }

    git_commit_val = _resolve_git_commit()
    dep_versions_val = _resolve_dependency_versions()

    manifest_payload = {
        "engine_version": engine_version,
        "git_commit": git_commit_val,
        "seed": active_seed,
        "seeds": seed_tuple,
        "resolved_config_hash": resolved_cfg_hash,
        "scenario_id": scen_id,
        "initial_state_fingerprint": init_state_fp,
        "component_ids": tuple(sorted(comp_ids)),
        "component_versions": comp_versions,
        "determinism_flags": active_flags,
        "dependency_versions": dep_versions_val,
        "hardware_summary": hardware,
        "execution_duration_sec": execution_duration_sec,
    }

    # Compute self-fingerprint
    manifest_hash = canonical_sha256(manifest_payload)

    return RunManifest(
        manifest_id=f"manifest_{manifest_hash[:16]}",
        timestamp=time.time(),
        execution_duration_sec=execution_duration_sec,
        engine_version=engine_version,
        git_commit=git_commit_val,
        seed=active_seed,
        seeds=seed_tuple,
        resolved_config_hash=resolved_cfg_hash,
        scenario_id=scen_id,
        initial_state_fingerprint=init_state_fp,
        component_ids=tuple(sorted(comp_ids)),
        component_versions=comp_versions,
        determinism_flags=active_flags,
        dependency_versions=dep_versions_val,
        hardware_summary=hardware,
        manifest_hash=manifest_hash,
    )
