"""Filesystem backend for ResultStore with JSON and NumPy sidecars.

Conforms to Track T2: Fingerprint-keyed ResultStore (CORE, zero new deps).
Uses JSON/YAML for structured scenario/trajectory metadata and NumPy .npz
sidecars for compact numerical timeseries and metric arrays.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ewm_engine.durability.protocol import ResultStore
from ewm_engine.provenance.metadata import Provenance
from ewm_engine.simulation.metrics import RunMetrics
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import SimulationResult, Trajectory


class FilesystemResultStore(ResultStore):
    """Filesystem implementation of ResultStore keyed by canonical SHA-256 fingerprint.

    Stores:
    - `<cache_dir>/<fingerprint>/result.json`: Complete serialized scenario, metadata, and trajectories.
    - `<cache_dir>/<fingerprint>/arrays.npz`: NumPy sidecar containing tabular numerical metric arrays.
    """

    def __init__(self, root_dir: str | Path) -> None:
        """Initialize filesystem result store at root_dir."""
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, fingerprint: str) -> Path:
        return self.root_dir / fingerprint

    def contains(self, fingerprint: str) -> bool:
        """Check whether a cached simulation result directory exists."""
        entry = self._entry_dir(fingerprint)
        return entry.is_dir() and (entry / "result.json").is_file()

    def get(self, fingerprint: str) -> SimulationResult | None:
        """Retrieve and deserialize a SimulationResult from its fingerprint directory."""
        entry = self._entry_dir(fingerprint)
        result_file = entry / "result.json"
        if not result_file.is_file():
            return None

        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            scenario = Scenario.model_validate(data["scenario"])
            provenance = Provenance.model_validate(data["provenance"])
            run_metrics = (
                RunMetrics.model_validate(data["run_metrics"]) if "run_metrics" in data else None
            )
            trajectories = [
                Trajectory.model_validate(t_data) for t_data in data.get("trajectories", [])
            ]
            for t in trajectories:
                t.finalize()

            return SimulationResult(
                scenario=scenario,
                provenance=provenance,
                trajectories=trajectories,
                run_metrics=run_metrics,
            )
        except Exception:
            return None

    def put(
        self,
        fingerprint: str,
        result: SimulationResult,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist a SimulationResult and numerical arrays atomically."""
        target_dir = self._entry_dir(fingerprint)
        tmp_dir = self.root_dir / f".tmp_{fingerprint}_{os.getpid()}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Prepare JSON payload
            payload = {
                "schema_version": "1.0.0",
                "fingerprint": fingerprint,
                "metadata": dict(metadata or {}),
                "scenario": result.scenario.model_dump(mode="json"),
                "provenance": result.provenance.model_dump(mode="json"),
                "run_metrics": result.run_metrics.model_dump(mode="json"),
                "trajectories": [t.model_dump(mode="json") for t in result.trajectories],
            }
            json_file = tmp_dir / "result.json"
            json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            # 2. Prepare NumPy sidecar
            seeds = np.array([t.seed for t in result.trajectories], dtype=np.int64)
            sample_ids = np.array([t.sample_id for t in result.trajectories], dtype=np.int64)
            step_counts = np.array([len(t.steps) for t in result.trajectories], dtype=np.int64)

            # Metric matrix if steps exist
            all_metric_keys = sorted(
                {k for t in result.trajectories for s in t.steps for k in s.step_metrics.keys()}
            )
            metric_arrays: dict[str, Any] = {
                "seeds": seeds,
                "sample_ids": sample_ids,
                "step_counts": step_counts,
            }
            if all_metric_keys and result.trajectories and result.trajectories[0].steps:
                max_horizon = max((len(t.steps) for t in result.trajectories), default=0)
                for m_key in all_metric_keys:
                    mat = np.full(
                        (len(result.trajectories), max_horizon),
                        fill_value=np.nan,
                        dtype=np.float64,
                    )
                    for r_idx, t in enumerate(result.trajectories):
                        for s_idx, s in enumerate(t.steps):
                            if m_key in s.step_metrics:
                                mat[r_idx, s_idx] = s.step_metrics[m_key]
                    metric_arrays[f"metric_{m_key}"] = mat

            npz_file = tmp_dir / "arrays.npz"
            np.savez_compressed(npz_file, **metric_arrays)

            # 3. Atomic replacement
            if target_dir.exists():
                shutil.rmtree(target_dir)
            tmp_dir.replace(target_dir)

        except Exception:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def delete(self, fingerprint: str) -> bool:
        """Delete cached entry directory by fingerprint."""
        target_dir = self._entry_dir(fingerprint)
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
            return True
        return False

    def list_fingerprints(self) -> list[str]:
        """List all valid cached fingerprints."""
        fps: list[str] = []
        for entry in self.root_dir.iterdir():
            if entry.is_dir() and not entry.name.startswith(".tmp_"):
                if (entry / "result.json").is_file():
                    fps.append(entry.name)
        return sorted(fps)
