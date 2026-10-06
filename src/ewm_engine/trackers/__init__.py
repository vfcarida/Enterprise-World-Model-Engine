"""Experiment tracking and provenance module (T9).

Provides TrackerBackend protocol, zero-dependency LocalJsonTracker, OmegaConf loader,
and MLflow/W&B/DVC tracker adapters.
"""

from __future__ import annotations

from ewm_engine.trackers.adapters import (
    DvcCliTracker,
    MLflowTracker,
    WandbTracker,
    is_mlflow_available,
    is_wandb_available,
)
from ewm_engine.trackers.local_json import LocalJsonTracker
from ewm_engine.trackers.omegaconf_adapter import (
    OmegaConfConfigLoader,
    is_omegaconf_available,
)
from ewm_engine.trackers.protocol import TrackerBackend

__all__ = [
    "DvcCliTracker",
    "LocalJsonTracker",
    "MLflowTracker",
    "OmegaConfConfigLoader",
    "TrackerBackend",
    "WandbTracker",
    "is_mlflow_available",
    "is_omegaconf_available",
    "is_wandb_available",
]
