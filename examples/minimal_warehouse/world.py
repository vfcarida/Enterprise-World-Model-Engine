"""Minimal Warehouse World setup and factory loading."""

from __future__ import annotations

from pathlib import Path

from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.core.world import World


def get_warehouse_spec_path() -> Path:
    """Return the filesystem path to the minimal warehouse specification YAML."""
    return Path(__file__).resolve().parent / "spec.yaml"


def load_warehouse_spec() -> WorldSpec:
    """Load and validate the minimal warehouse WorldSpec from YAML."""
    return WorldSpec.from_yaml(get_warehouse_spec_path())


def create_warehouse_world(factory: WorldFactory | None = None) -> World:
    """Construct an executable World instance for the minimal warehouse fixture."""
    spec = load_warehouse_spec()
    wf = factory or WorldFactory()
    return wf.create_world(spec)
