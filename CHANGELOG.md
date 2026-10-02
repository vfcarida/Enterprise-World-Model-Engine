# Changelog

All notable changes to the Enterprise World Model Engine (EWM Engine) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Governance & Verification**: Formally vendored v1 specification contract in `docs/specs/spec-driven-development.md`, governance templates in `.github/`, and static import boundary tests in `tests/architecture/`.
- **API Surface Refinements**: Added `ConstraintPhase` enum (`PRE_ACTION`, `POST_TRANSITION`), `TrajectoryStatus` enum (`COMPLETED`, `INVALID`, `FAILED`), and `Provenance` metadata alias.
- **Experimental Namespace**: Created `ewm_engine.experimental` for research models (`RecedingHorizonSimulator`, `LinearResidualDynamics`, `Intervention`).

### Changed
- Reconciled root `ewm_engine.__all__` to the 22 canonical Stable symbols (ADR-007). Concrete dynamics, constraints, and actor classes remain importable from their respective submodules.

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
