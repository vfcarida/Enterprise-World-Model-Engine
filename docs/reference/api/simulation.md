# Simulation Engine API Reference

This module provides the execution kernel for Monte Carlo rollouts, deterministic seed derivation, scenario execution, and receding-horizon control.

---

## Core Simulation Engine

::: ewm_engine.simulation.engine
    options:
      show_root_heading: true
      show_source: false
      members:
        - SimulationEngine

---

## Scenario & Actions

::: ewm_engine.simulation.scenario
    options:
      show_root_heading: true
      show_source: false
      members:
        - Scenario
        - ScheduledAction

---

## Trajectories & Results

::: ewm_engine.simulation.trajectory
    options:
      show_root_heading: true
      show_source: false
      members:
        - Trajectory
        - SimulationResult
        - StepRecord
        - TrajectoryStatus

---

## Receding Horizon Control (MPC)

::: ewm_engine.simulation.mpc
    options:
      show_root_heading: true
      show_source: false
      members:
        - RecedingHorizonSimulator
        - MPCDecisionRecord
        - RecedingHorizonResult
