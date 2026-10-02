"""Integration test executing the minimal world runnable demonstration."""

from __future__ import annotations

from examples.minimal_world.run import run_minimal_world


def test_minimal_world_execution() -> None:
    """Ensure minimal_world example runs to completion without errors."""
    run_minimal_world()
