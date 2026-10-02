# ADR-002: Pluggable Dynamics Model Protocol

## Status
Accepted

## Context
Real-world systems cannot be represented by a single monolithic ML model or rule set. Different components require deterministic flows, empirical regressions, or stochastic models.

## Decision
Define `DynamicsModel` as a lightweight structural Python `Protocol` returning a typed `TransitionResult`. Enable modular composition through `CompositeDynamics`.

## Consequences
- **Positive**: Dynamics models can be swapped, chained, or extended without modifying the core simulation loop.
- **Negative**: Model composition order must be carefully designed to preserve physical conservation.
