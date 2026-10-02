"""Deterministic simulation engine, scenario definitions, and rollout trajectories."""

from __future__ import annotations

from ewm_engine.simulation.branching import branch_scenario, branch_world
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import SimulationResult, StepRecord, Trajectory

__all__ = [
    "Scenario",
    "SimulationEngine",
    "SimulationResult",
    "StepRecord",
    "Trajectory",
    "branch_scenario",
    "branch_world",
]
