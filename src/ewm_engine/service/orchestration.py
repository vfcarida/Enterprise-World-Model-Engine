"""Workflow orchestration adapters (Prefect, Dagster).

Wires EWM Engine canonical simulation fingerprints directly into orchestrator cache keys.
"""

from __future__ import annotations

import importlib
from typing import Any

from ewm_engine.core.world import World
from ewm_engine.durability.memoize import compute_simulation_fingerprint
from ewm_engine.simulation.scenario import Scenario


def is_prefect_available() -> bool:
    """Check if Prefect workflow engine is installed."""
    return importlib.util.find_spec("prefect") is not None


def is_dagster_available() -> bool:
    """Check if Dagster orchestration engine is installed."""
    return importlib.util.find_spec("dagster") is not None


class PrefectTaskAdapter:
    """Generates Prefect cache-key functions directly from EWM simulation fingerprints."""

    @staticmethod
    def cache_key_fn(context: Any, parameters: dict[str, Any]) -> str:
        """Prefect cache_key_fn extracting world and scenario to compute fingerprint."""
        world: World | None = parameters.get("world")
        scenario: Scenario | None = parameters.get("scenario")
        if world is not None and scenario is not None:
            return compute_simulation_fingerprint(world, scenario)
        return "ewm_uncached"


class DagsterAssetAdapter:
    """Projects EWM simulation fingerprints into Dagster DataVersion."""

    @staticmethod
    def get_data_version(world: World, scenario: Scenario) -> str:
        """Return canonical SHA-256 fingerprint for Dagster DataVersion tagging."""
        return compute_simulation_fingerprint(world, scenario)
