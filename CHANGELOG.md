# Changelog

All notable changes to the Enterprise World Model Engine (EWM Engine) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-02

### Added
- **Core Domain Primitives**: `World`, `WorldState`, `Entity`, `Relationship`, `Resource`, `Action`, `Intervention`, `ExogenousEvent`.
- **Pluggable Dynamics Engine**: `DynamicsModel` protocol, `DeterministicDynamics`, `StochasticDynamics`, `CompositeDynamics`, and `LearnedDynamics` experimental interface.
- **First-Class Constraints Engine**: `Constraint` protocol, `ConstraintRegistry`, hard and soft constraint validation, and constraint violation provenance tracking.
- **Actor & Agent Interface**: `Actor` protocol, `RuleBasedActor`, and `StochasticActor`.
- **Deterministic Monte Carlo Simulation**: `SimulationEngine`, `Scenario`, `Trajectory`, reproducible random seed spawning.
- **Scenario Branching & Comparison**: Snapshot branching from identical initial states, counterfactual evaluation, distribution quantiles, and automated metric deltas.
- **Systemic Traces & Epistemic Honesty**: Causal/dependency graph capture with explicit `EvidenceLevel` categorization (`STRUCTURAL`, `INTERVENTIONAL`, `QUASI_CAUSAL`, `PREDICTIVE`, `ASSUMED`).
- **Reference Examples**:
  - `minimal_world`: Two-warehouse inventory balancing with capacity constraints.
  - `civicflow`: Multi-region flood-response disaster logistics research simulation.
- **CLI & Governance**: Minimal `ewm` CLI entrypoint, complete test suite, strict static typing, and open-source documentation.
