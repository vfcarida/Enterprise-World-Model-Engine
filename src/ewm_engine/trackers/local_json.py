"""Zero-dependency local JSON experiment tracker.

Stores structured simulation run logs, metrics timeseries, and cards in
.ewm_runs/{fingerprint}/run.json.
"""

from __future__ import annotations

import datetime
import json
import threading
from pathlib import Path
from typing import Any

from ewm_engine.cards.models import DatasetCard, ModelCard, ScenarioCard
from ewm_engine.trackers.protocol import TrackerBackend


class LocalJsonTracker(TrackerBackend):
    """Zero-dependency file-based experiment tracker keyed by canonical fingerprint."""

    def __init__(self, base_dir: str | Path = ".ewm_runs") -> None:
        self.base_dir = Path(base_dir)
        self._fingerprint: str = "unkeyed_run"
        self._params: dict[str, Any] = {}
        self._metrics_history: list[dict[str, Any]] = []
        self._latest_metrics: dict[str, float] = {}
        self._cards: list[dict[str, Any]] = []
        self._artifacts: list[dict[str, str]] = []
        self._created_at = datetime.datetime.now(datetime.UTC).isoformat()
        self._lock = threading.RLock()

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def run_directory(self) -> Path:
        return self.base_dir / self._fingerprint

    def log_fingerprint(self, fingerprint: str) -> None:
        """Set the canonical run fingerprint and write/update log."""
        with self._lock:
            self._fingerprint = fingerprint
            self._save()

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters dictionary."""
        with self._lock:
            self._params.update(params)
            self._save()

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics point."""
        with self._lock:
            now = datetime.datetime.now(datetime.UTC).isoformat()
            entry = {"timestamp": now, "step": step, "metrics": dict(metrics)}
            self._metrics_history.append(entry)
            self._latest_metrics.update(metrics)
            self._save()

    def log_card(self, card: ModelCard | ScenarioCard | DatasetCard) -> None:
        """Log structured card."""
        with self._lock:
            self._cards.append(card.model_dump())
            self._save()

    def log_artifact(self, path: str, artifact_type: str = "file") -> None:
        """Log an artifact file path."""
        with self._lock:
            self._artifacts.append({"path": str(path), "type": artifact_type})
            self._save()

    def get_run_data(self) -> dict[str, Any]:
        """Return the in-memory run dictionary."""
        with self._lock:
            return {
                "fingerprint": self._fingerprint,
                "created_at": self._created_at,
                "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "params": dict(self._params),
                "latest_metrics": dict(self._latest_metrics),
                "metrics_history": list(self._metrics_history),
                "cards": list(self._cards),
                "artifacts": list(self._artifacts),
            }

    def _save(self) -> None:
        """Write current run data to disk atomically."""
        run_dir = self.run_directory
        run_dir.mkdir(parents=True, exist_ok=True)
        file_path = run_dir / "run.json"
        tmp_path = run_dir / "run.json.tmp"

        payload = self.get_run_data()
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        tmp_path.replace(file_path)
