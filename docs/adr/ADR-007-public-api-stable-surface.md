# ADR-007: Public API Stable Surface and Experimental Namespace Split

## Status
Accepted

## Context
As the engine evolves towards v1.0.0, the root package `ewm_engine` exported concrete implementations alongside core domain abstractions. To provide long-term SemVer stability, the public API surface must be strictly delineated between frozen **Stable** primitives and **Experimental** research extensions.

## Decision
1. Reconcile the root `ewm_engine.__all__` surface to the 22 canonical Stable symbols:
   - Core primitives: `Action`, `Entity`, `ExogenousEvent`, `Relationship`, `Resource`, `World`, `WorldState`
   - Constraints: `Constraint`, `ConstraintPhase`, `ConstraintResult`, `ConstraintSeverity`
   - Dynamics: `DynamicsModel`, `TransitionResult`
   - Simulation & Trajectory: `Scenario`, `SimulationEngine`, `SimulationResult`, `Trajectory`, `TrajectoryStatus`
   - Provenance: `EvidenceLevel`, `Provenance`, `TraceEdge`
   - Evaluation: `compare_scenarios`
   - Package version: `__version__`
2. Concrete dynamics implementations (`CompositeDynamics`, `DeterministicDynamics`, `StochasticDemandDynamics`), actors, and registries remain public from their submodules (`ewm_engine.dynamics`, `ewm_engine.actors`, `ewm_engine.constraints`).
3. Relocate online re-grounding (`RecedingHorizonSimulator`), learned dynamics models (`LinearResidualDynamics`, `NeuralResidualDynamics`), and candidate policy interventions (`Intervention`) into the `ewm_engine.experimental` namespace until their evaluation semantics stabilize.

## Consequences
- **Positive**: Strict API stability guarantees for core simulation users; zero risk of breaking core users when experimental models evolve.
- **Negative**: Users of advanced experimental features must import from `ewm_engine.experimental`.
