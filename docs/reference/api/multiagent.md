# Multi-Agent Coordination API Reference

This module defines actor observation views, action proposals, deterministic constraint mediation, and game-theoretic solvers (Track T6).

---

## Observation Views & Proposals

::: ewm_engine.multiagent.views
    options:
      show_root_heading: true
      show_source: false
      members:
        - ActorObservationView
        - ActorActionProposal
        - AdjudicationResult

---

## Deterministic Mediator

::: ewm_engine.multiagent.mediator
    options:
      show_root_heading: true
      show_source: false
      members:
        - Mediator
        - ConstraintMediator

---

## Game Theory & Solvers

::: ewm_engine.multiagent.game_theory
    options:
      show_root_heading: true
      show_source: false
      members:
        - NashpyGameSolver

---

## Multi-Agent Gym Adapters

::: ewm_engine.multiagent.adapters
    options:
      show_root_heading: true
      show_source: false
      members:
        - PettingZooParallelAdapter
