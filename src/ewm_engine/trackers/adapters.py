"""Experiment tracker adapters (MLflow, Weights & Biases, DVC via CLI).

Quarantined behind optional adapters. Core tests pass with none installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
from pathlib import Path
from typing import Any

from ewm_engine.cards.models import DatasetCard, ModelCard, ScenarioCard
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.trackers.protocol import TrackerBackend


def is_mlflow_available() -> bool:
    """Check if MLflow is installed."""
    return importlib.util.find_spec("mlflow") is not None


def is_wandb_available() -> bool:
    """Check if Weights & Biases is installed."""
    return importlib.util.find_spec("wandb") is not None


class MLflowTracker(TrackerBackend):
    """Experiment tracker logging to MLflow with simulation fingerprint tagging."""

    def __init__(self, experiment_name: str = "ewm_simulation") -> None:
        if not is_mlflow_available():
            raise SimulationConfigurationError(
                "MLflow is required for MLflowTracker. Install via `pip install ewm-engine[trackers]`."
            )
        self.experiment_name = experiment_name
        self._mlflow = importlib.import_module("mlflow")
        self._mlflow.set_experiment(experiment_name)

    def log_fingerprint(self, fingerprint: str) -> None:
        self._mlflow.set_tag("ewm.fingerprint", fingerprint)

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({str(k): str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_card(self, card: ModelCard | ScenarioCard | DatasetCard) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(card.to_json())
            tmp_path = f.name
        self._mlflow.log_artifact(tmp_path, artifact_path="cards")

    def log_artifact(self, path: str, artifact_type: str = "file") -> None:
        self._mlflow.log_artifact(path, artifact_path=artifact_type)


class WandbTracker(TrackerBackend):
    """Experiment tracker logging to Weights & Biases with simulation fingerprint tagging."""

    def __init__(self, project_name: str = "ewm_simulation") -> None:
        if not is_wandb_available():
            raise SimulationConfigurationError(
                "wandb is required for WandbTracker. Install via `pip install wandb`."
            )
        self.project_name = project_name
        self._wandb = importlib.import_module("wandb")

    def log_fingerprint(self, fingerprint: str) -> None:
        self._wandb.config.update({"ewm.fingerprint": fingerprint})

    def log_params(self, params: dict[str, Any]) -> None:
        self._wandb.config.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_card(self, card: ModelCard | ScenarioCard | DatasetCard) -> None:
        self._wandb.summary.update(
            {"card_type": type(card).__name__, "card_data": card.model_dump()}
        )

    def log_artifact(self, path: str, artifact_type: str = "file") -> None:
        art = self._wandb.Artifact(name=Path(path).stem, type=artifact_type)
        art.add_file(path)
        self._wandb.log_artifact(art)


class DvcCliTracker:
    """Zero-dependency DVC tracker using CLI shell-out to avoid vendoring ~40 dependencies."""

    def __init__(self, repo_dir: str | Path = ".") -> None:
        self.repo_dir = Path(repo_dir)

    def track_artifact(self, target_path: str | Path) -> bool:
        """Run `dvc add <target_path>` via subprocess."""
        cmd = ["dvc", "add", str(target_path)]
        try:
            res = subprocess.run(
                cmd, cwd=self.repo_dir, capture_output=True, text=True, check=False
            )
            return res.returncode == 0
        except FileNotFoundError:
            return False

    def push(self) -> bool:
        """Run `dvc push` via subprocess."""
        cmd = ["dvc", "push"]
        try:
            res = subprocess.run(
                cmd, cwd=self.repo_dir, capture_output=True, text=True, check=False
            )
            return res.returncode == 0
        except FileNotFoundError:
            return False
