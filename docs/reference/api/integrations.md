# Ecosystem Integrations & Adapters API Reference

This module provides hexagonal adapters connecting EWM Engine with external solvers, reinforcement learning environments, observability frameworks, and generative multi-agent systems.

---

## Concordia Game-Master World Arbiter

::: ewm_engine.integrations.concordia
    options:
      show_root_heading: true
      show_source: false
      members:
        - EWMWorldArbiter
        - ArbiterVerdict

---

## OpenTelemetry Observability Hook

::: ewm_engine.integrations.otel
    options:
      show_root_heading: true
      show_source: false
      members:
        - OpenTelemetryHook

---

## Solvers, Planners & RL Environments

::: ewm_engine.integrations.scipy_planner
    options:
      show_root_heading: true
      show_source: false
      members:
        - SciPyAllocationPlanner

::: ewm_engine.integrations.ortools
    options:
      show_root_heading: true
      show_source: false
      members:
        - ORToolsAllocationPlanner

::: ewm_engine.integrations.solvers
    options:
      show_root_heading: true
      show_source: false
      members:
        - Z3SMTConstraint

::: ewm_engine.integrations.gym
    options:
      show_root_heading: true
      show_source: false
      members:
        - EWMGymEnv
