"""Integration test executing the CivicFlow flagship simulation demonstration."""

from __future__ import annotations

from examples.civicflow.run import run_civicflow_simulation


def test_civicflow_simulation_execution() -> None:
    """Ensure CivicFlow flood response simulation runs to completion without errors."""
    run_civicflow_simulation()
