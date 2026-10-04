# Enterprise World Model Engine Roadmap

This document outlines the phased development roadmap for the **Enterprise World Model Engine (EWM Engine)**.

> **Canonical Contract:** See [docs/specs/spec-driven-development.md](docs/specs/spec-driven-development.md) for the authoritative v1 specification and authority chain governing this roadmap. See [docs/stability-policy.md](docs/stability-policy.md) for SemVer 2.0.0 stability guarantees.

---

## Maturity Framework

Capabilities in EWM Engine are classified under explicit maturity levels:

- **Stable**: Frozen public contract (`1.x`), covered by backwards-compatibility guarantees, automated contract gates (`AC-024`), and deprecation lifecycles.
- **Experimental**: Active research and development in `ewm_engine.experimental.*`; subject to API evolution across minor releases without deprecation cycles.
- **Planned Adapter**: Planned external integration orbiting the core simulation kernel.
- **Research / Exploration**: Theoretical formulation or empirical investigation.

---

## Current Status (v1.0.0 — Stable Contract)

| Subsystem | Maturity Level | Status / Notes |
| :--- | :---: | :--- |
| **Core Simulation Kernel** | **Stable** | Deterministic Monte Carlo rollout, seed spawning, bitwise reproducibility (AC-004, AC-009) |
| **World & State Model** | **Stable** | Deeply immutable snapshot states, canonical JSON serializer, SHA-256 fingerprinting (AC-005, AC-010) |
| **Constraint Engine** | **Stable** | Phase-aware (`PRE_ACTION`, `POST_TRANSITION`), normative action rejection and rollout invalidation (AC-006–AC-008) |
| **Pluggable Dynamics** | **Stable** | Structural `DynamicsModel` protocol, composite and deterministic transfer implementations |
| **Scenario Branching** | **Stable** | Safe branch isolation without mutable state cross-contamination (AC-005) |
| **Systemic Traces** | **Stable** | Epistemic honesty with explicit `EvidenceLevel` tiers; directed dependency DAG export to Mermaid/NetworkX (AC-011) |
| **Declarative Serialization** | **Stable** | Draft 2020-12 versioned JSON Schemas, safe YAML loader, and trusted closed `WorldFactory` registry (AC-002, AC-003, AC-018) |
| **Scenario Evaluation** | **Stable** | Multi-quantile counterfactual comparisons (`compare_scenarios`) |
| **Normative Acceptance** | **Stable** | Minimal Warehouse (AC-012) and CivicFlow regional flood response (AC-013) fixtures |
| **Observability Layer** | **Stable** | Typed lifecycle hooks (`HookRegistry`, `HookEvent`) and programmatic `RunMetrics` |
| **Command-Line Interface** | **Stable** | `ewm [example|validate|run|schema]` subcommands for execution and validation |
| **Ecosystem Adapters** | **Alpha** | `CallableActorAdapter` (LangGraph/AutoGen), `Z3ConstraintAdapter` (SMT), `EnterpriseGymEnv` (RL), and `ORToolsAllocationAdapter` (OR) |
| **Online Re-Grounding (MPC)** | **Experimental** | `RecedingHorizonSimulator` and `MPCDecisionRecord` in `ewm_engine.experimental` |
| **Learned Dynamics** | **Experimental** | `LinearResidualDynamics` and `TransitionDataset` in `ewm_engine.experimental` |

---

## Near-Term Roadmap (v1.1.0 — Ecosystem & Scale)

### 1. Operations Research (OR) Adapters
- **Google OR-Tools Integration**: Adapter connecting linear and mixed-integer programming (MILP) dispatch policies as external interventional actors.
- **SciPy Optimization Adapters**: Continuous resource optimization for multi-warehouse inventory allocation.

### 2. Distributed Monte Carlo Execution
- **Ray / Multiprocessing Parallelism**: High-throughput parallel rollouts across multi-core nodes while strictly preserving per-rollout `SeedSequence` determinism.
- **Batch State Evaluations**: Vectorized constraint checking for massive topology simulations.

### 3. OpenTelemetry & Telemetry Exporters
- Zero-overhead OpenTelemetry span export from lifecycle hooks for production simulation observability.

---

## Long-Term Research Agenda (v2.0.0+)

1. **Latent Dynamics & World Representations**:
   - Recurrent State-Space Models (RSSM / Dreamer-like latent rollouts) integrated as pluggable `DynamicsModel` adapters.
   - Joint Embedding Predictive Architectures (JEPA) for multi-scale organizational dynamics.
   - Graph Neural Networks (GNNs) for heterogeneous organizational relational graphs.
2. **Causal Epistemics & Off-Policy Evaluation**:
   - Automated identification of unobserved confounders in observational enterprise event logs.
   - Off-policy counterfactual evaluation with overlap and positivity diagnostics.
3. **Out-of-Distribution (OOD) & Regime Shift Detection**:
   - Epistemic uncertainty quantification detecting when simulation trajectories enter ungrounded state regimes.
4. **Differentiable Constraints & Projectors**:
   - Differentiable manifold projection for continuous control optimization.
