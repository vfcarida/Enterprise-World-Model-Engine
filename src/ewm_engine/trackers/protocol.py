"""Experiment tracker protocols.

Provides standardized provenance and tracking interfaces for simulation experiments.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ewm_engine.cards.models import DatasetCard, ModelCard, ScenarioCard


@runtime_checkable
class TrackerBackend(Protocol):
    """Protocol for experiment and provenance tracking backends."""

    def log_fingerprint(self, fingerprint: str) -> None:
        """Record the canonical simulation or model fingerprint."""
        ...

    def log_params(self, params: dict[str, Any]) -> None:
        """Record simulation parameters or hyperparameters."""
        ...

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Record scalar simulation metrics at an optional step index."""
        ...

    def log_card(self, card: ModelCard | ScenarioCard | DatasetCard) -> None:
        """Record a structured Pydantic reproducibility card."""
        ...

    def log_artifact(self, path: str, artifact_type: str = "file") -> None:
        """Record an artifact file path and its semantic type."""
        ...
