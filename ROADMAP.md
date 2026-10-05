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
| **Scenario Evaluation** | **Stable** | Bootstrap CIs on deltas, empirical significance flags, and multi-objective Pareto analysis (FEAT-001, ADR-016) |
| **Normative Acceptance** | **Stable** | Minimal Warehouse (AC-012) and CivicFlow regional flood response (AC-013) fixtures |
| **Observability Layer** | **Stable** | Typed lifecycle hooks (`HookRegistry`, `HookEvent`) and programmatic `RunMetrics` |
| **Command-Line Interface** | **Stable** | `ewm [example|validate|run|schema]` subcommands for execution and validation |
| **Ecosystem Adapters** | **Alpha** | `CallableActorAdapter` (LangGraph/AutoGen), `Z3ConstraintAdapter` (SMT), `EnterpriseGymEnv` (RL), and `ORToolsAllocationAdapter` (OR) |
| **Online Re-Grounding (MPC)** | **Experimental** | `RecedingHorizonSimulator` and `MPCDecisionRecord` in `ewm_engine.experimental` |
| **Learned Dynamics** | **Experimental** | `LinearResidualDynamics` and `TransitionDataset` in `ewm_engine.experimental` |

---

## Near-Term Roadmap (v1.1.0 — Ecosystem & Scale)

### 1. Evaluation & Uncertainty Upgrade (Completed)
- **Bootstrap Confidence Intervals**: Non-parametric percentile bootstrap intervals on counterfactual deltas ($\Delta = \mu_{cand} - \mu_{base}$).
- **Empirical Significance Testing**: Simulation stochastic significance testing under $P_{model}$ with explicit epistemic guardrails.
- **Multi-Objective Pareto Analysis**: Non-dominated Pareto frontier extraction, dominance relationship tracking, and anti-winner guardrails.

### 2. Operations Research (OR) Adapters
- **Google OR-Tools Integration**: Adapter connecting linear and mixed-integer programming (MILP) dispatch policies as external interventional actors.
- **SciPy Optimization Adapters**: Continuous resource optimization for multi-warehouse inventory allocation.

### 3. Distributed Monte Carlo Execution
- **Ray / Multiprocessing Parallelism**: High-throughput parallel rollouts across multi-core nodes while strictly preserving per-rollout `SeedSequence` determinism.
- **Batch State Evaluations**: Vectorized constraint checking for massive topology simulations.

### 4. OpenTelemetry & Telemetry Exporters
- Zero-overhead OpenTelemetry span export from lifecycle hooks for production simulation observability.

---

## Platform Completeness Roadmap (v1.2 — v1.6)

See [01_EXPANDED_ROADMAP.md](01_EXPANDED_ROADMAP.md) for detailed research grounding, dependency budgets, and prompt governance.

### v1.2.0 — Durability & Reproducibility (Tracks T1, T2, T9)
- **T1: Event-Sourced `TraceLog` & Replay (CORE)**: Append-only transition log with versioned canonical event schemas; `EventStore` protocol (`append/read/fold`) with stdlib SQLite and JSON/YAML backends. Columnar sink to Parquet/DuckDB (`[analytics]` extra).
- **T2: Fingerprint-Keyed `ResultStore` & Memoization (CORE)**: Deterministic rollout memoization keyed by cryptographic scenario/state fingerprints. Filesystem backend with `.npy` sidecars; Redis/S3 adapters (`[storage]` extra).
- **T9: Native Cards & Experiment Tracking (CORE)**: Pydantic-based Scenario, Model, and Dataset Cards with embedded SHA-256 fingerprints; `TrackerBackend` protocol with local JSON logging; MLflow/W&B/DVC adapters.

### v1.3.0 — Trajectory Verification & Temporal Logic (Track T3)
- **T3: Trajectory Verification (CORE / EXTRA)**:
  - **Oracle-Graph Verifier (CORE)**: DAG-based consistency, causality, and timing verification over systemic traces.
  - **STL Robustness Monitoring (`[stl]` extra)**: Signal Temporal Logic (STL/MTL) verification via RTAMT, computing quantitative robustness margins and rollout distributions.

### v1.4.0 — Scientific Experimentation Suite (Track T4)
- **T4: Design of Experiments, Calibration & Optimization (CORE / EXTRA)**:
  - **DoE & Backtesting (CORE)**: Factorial/LHS/OAT parameter sweep harness keyed by `SeedSequence`; walk-forward validation with moment-matching, CRPS, and coverage scoring.
  - **Global Sensitivity (`[sensitivity]` extra)**: Sobol, Morris, and FAST indices via SALib.
  - **Calibration & Emulation (`[calibrate]` extra)**: Distance-based calibration via SciPy/scikit-learn and Gaussian Process emulation.
  - **Intervention Optimization (`[opt-evolutionary]`, `[opt-pareto]` extras)**: CMA-ES via pycma, multi-objective Pareto optimization via pymoo.

### v1.5.0 — Interoperability, Multi-Agent & Serving (Tracks T5, T6, T7, T8)
- **T5: Co-Simulation & Standards (CORE / ADAPTER)**: Master co-simulation stepping loop (CORE); FMI 3.0 / FMU adapter via FMPy (`[fmi]`), discrete-event via SimPy (`[simpy]`), agent-based via Mesa (`[mesa]`), system dynamics via PySD (`[sd]`).
- **T6: Multi-Agent Coordination & Game Theory (CORE / EXTRA)**: Multi-actor observation/action views and deterministic mediator (CORE); 2-player Nash and replicator dynamics via Nashpy (`[game]` extra); PettingZoo and Concordia adapters.
- **T7: Visualization & Reporting (CORE / EXTRA)**: Pydantic `ReportModel` (CORE); interactive Plotly fan charts and comparison views (`[viz]` extra).
- **T8: Simulation-as-a-Service & Orchestration (CORE / EXTRA)**: Local `JobRunner` with multiprocessing (CORE); FastAPI REST serving with fingerprint caching (`[serve]` extra); Prefect/Dagster pipe integration.

### v1.6.0+ — Learned Depth & Causal Epistemics
- **Learned-Dynamics Evaluation Harness**: Multi-step rollout divergence, calibration, constraint-violation rates, and invariant consistency gates.
- **Neural `LearnedDynamics` Baseline (`[ml]` extra)**: Latent transition models trained on `TransitionDataset`.
- **Benchmark Families**: `InterventionShift`, `RuleShift`, `ConstraintStress`, `LongHorizon`, `MultiAgentCascade`.
- **Planning & Controller Layer**: Uncertainty-aware rollout scoring and receding-horizon selection.
- **OOD & Regime-Shift Detection**: Epistemic uncertainty monitoring flagging ungrounded state trajectories.
- **Causal Discovery & Off-Policy Diagnostics**: Overlap, positivity, and confounding diagnostics.

---

## Long-Term Research Agenda (v2.0.0+)

1. **Heterogeneous Relational Graph State & GNNs**:
   - Relational graph state models and Graph Neural Network dynamics adapters for complex organizational networks.
2. **Declarative World Specification Language (WSL)**:
   - Versioned, safe, domain-specific language for organizational world specifications, compiling down to Python kernels.
3. **Latent World Foundation Architectures**:
   - RSSM / Dreamer-style latent world models and Joint Embedding Predictive Architectures (JEPA) for multi-scale organizational dynamics.
4. **Differentiable Constraints & Manifold Projectors**:
   - Differentiable manifold projection for continuous control optimization.
