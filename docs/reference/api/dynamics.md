# Pluggable Dynamics API Reference

This module defines the pluggable dynamics protocol governing state forward transitions ($\mathcal{T}: S_t \times A_t \times E_t \to S_{t+1}$).

---

## Dynamics Protocol & Results

::: ewm_engine.dynamics.base
    options:
      show_root_heading: true
      show_source: false
      members:
        - DynamicsModel
        - TransitionResult

---

## Composition Pipeline

::: ewm_engine.dynamics.composite
    options:
      show_root_heading: true
      show_source: false
      members:
        - CompositeDynamics

---

## Standard Dynamics Implementations

::: ewm_engine.dynamics.deterministic
    options:
      show_root_heading: true
      show_source: false
      members:
        - DeterministicTransferDynamics
        - DeterministicDemandDynamics

::: ewm_engine.dynamics.stochastic
    options:
      show_root_heading: true
      show_source: false
      members:
        - StochasticDemandDynamics

---

## Learned Dynamics & Datasets

::: ewm_engine.dynamics.learned
    options:
      show_root_heading: true
      show_source: false
      members:
        - LearnedDynamics
        - LinearResidualDynamics
        - TransitionDataset
        - TransitionSample
