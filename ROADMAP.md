# Enterprise World Model Engine Roadmap

This document outlines the phased development roadmap for the **Enterprise World Model Engine (EWM Engine)**.

> **Canonical Contract:** See [docs/specs/spec-driven-development.md](docs/specs/spec-driven-development.md) for the authoritative v1 specification and authority chain governing this roadmap. See [docs/stability-policy.md](docs/stability-policy.md) for SemVer 2.0.0 stability guarantees.
>
> **Interactive Project Tracker:** Follow live development progress, sprint backlogs, and milestone deliverables on the pinned GitHub Project board:
> - **Pinned GitHub Project:** [https://github.com/users/vfcarida/projects/1](https://github.com/users/vfcarida/projects/1)
> - **Milestone v1.5.0 (Shipped):** [v1.5.0 Interop & Ecosystem](https://github.com/vfcarida/Enterprise-World-Model-Engine/milestone/1)
> - **Milestone v1.5.x (Shipped):** Positioning & Narrative
> - **Milestone v1.6.0 (Active):** [v1.6.0 Learned Dynamics & Causal Epistemics](https://github.com/vfcarida/Enterprise-World-Model-Engine/milestone/2)
> - **Milestone v1.7.0 (Planned):** Decision Intelligence (`ewm decide`)
> - **Milestone v1.8.0 (Planned):** Calibration to Reality (`ewm calibrate`)
> - **Milestone v1.9.0 (Planned):** Flywheel & Neuro-Symbolic Maturation
> - **Milestone v2.0.0 (Horizon):** Reference Release (Stable Kernel + Relational GNNs & WSL)

---

## Maturity Framework

Capabilities in EWM Engine are classified under explicit, strictly enforced maturity levels:

- **Stable**: Frozen public contract (`1.x`), covered by backwards-compatibility guarantees, automated contract gates (`AC-024`), and formal deprecation lifecycles.
- **Beta Adapter / Beta**: Production-grade, tested, time/resource-bounded; may evolve with deprecation.
- **Experimental**: Active research and prototyping under `ewm_engine.experimental.*`; subject to API evolution across minor releases without deprecation cycles.
- **Research**: Scientific measurement instruments, benchmark families, and theoretical formulations without production claims.

---

## Current Status (v1.5.0 Released / v2.0-alpha Horizon)

| Subsystem | Maturity Level | Status / Notes |
| :--- | :---: | :--- |
| **Core Simulation Kernel** | **Stable (v1.0.0)** | Deterministic Monte Carlo rollout, seed spawning, bitwise reproducibility (AC-004, AC-009) |
| **World & State Model** | **Stable (v1.0.0)** | Deeply immutable snapshot states, canonical JSON serializer, SHA-256 fingerprinting (AC-005, AC-010) |
| **Constraint Engine** | **Stable (v1.0.0)** | Phase-aware (`PRE_ACTION`, `POST_TRANSITION`), normative action rejection and rollout invalidation (AC-006–AC-008) |
| **Pluggable Dynamics Protocol** | **Stable (v1.0.0)** | Structural `DynamicsModel` protocol, composite and deterministic transfer implementations |
| **Scenario Branching** | **Stable (v1.0.0)** | Safe branch isolation without mutable state cross-contamination (AC-005) |
| **Systemic Traces** | **Stable (v1.0.0)** | Epistemic honesty with explicit `EvidenceLevel` tiers; directed dependency DAG export to Mermaid/NetworkX (AC-011) |
| **Declarative Serialization** | **Stable (v1.0.0)** | Draft 2020-12 versioned JSON Schemas, safe YAML loader, and trusted closed `WorldFactory` registry (AC-002, AC-003, AC-018) |
| **Scenario Evaluation** | **Stable (v1.1.0)** | Bootstrap CIs on deltas, empirical significance flags, and multi-objective Pareto analysis (FEAT-001, ADR-016) |
| **Distributed Monte Carlo** | **Beta Adapter (v1.1.0)** | High-throughput parallel execution preserving `SeedSequence` determinism across workers (FEAT-002, ADR-017) |
| **OpenTelemetry Observability** | **Beta Adapter (v1.1.0)** | Zero-overhead OpenTelemetry span and metric export from lifecycle hooks without core coupling (ADR-013) |
| **OR & Continuous Planners** | **Beta Adapter (v1.1.0)** | Google OR-Tools CP-SAT discrete and SciPy continuous allocation planners with timeout guards (ADR-018) |
| **SMT Formal Verification** | **Beta Adapter (v1.1.0)** | Z3 SMT constraint satisfaction adapter with timeout and resource limits (ADR-018) |
| **Gymnasium RL Adapter** | **Beta Adapter (v1.1.0)** | Standard Gym environment wrapper (`gymnasium.Env`) with step-bound resource safeguards |
| **Durability & Replay (T1, T2)** | **Beta (v1.2.0)** | Event-sourced `TraceLog`, `EventStore` protocol (SQLite/JSON), fingerprint-keyed `ResultStore` (ADR-026) |
| **Native Cards & Tracking (T9)** | **Beta (v1.2.0)** | Pydantic Scenario/Model/Dataset Cards with SHA-256 fingerprints; `TrackerBackend` protocol (ADR-029) |
| **Trajectory Verification (T3)** | **Beta (v1.3.0)** | Oracle-graph DAG verifier (CORE) and RTAMT Signal Temporal Logic (STL) robustness monitoring (`[stl]`, ADR-027) |
| **Scientific Experimentation (T4)** | **Beta (v1.4.0)** | DoE sweep harness (LHS/Sobol), walk-forward backtesting, global sensitivity (SALib), optimization (ADR-028) |
| **Platform Interoperability (T5)** | **Beta (v1.5.0)** | Master co-simulation stepping loop (CORE), FMI 3.0 / FMU adapter (FMPy), SimPy, Mesa, and PySD (ADR-029) |
| **Multi-Agent Coordination (T6)** | **Beta (v1.5.0)** | Multi-actor observation/action views, deterministic mediator, Nash equilibrium solving via Nashpy (ADR-029) |
| **Reporting & Serving (T7, T8)** | **Beta (v1.5.0)** | Pydantic `ReportModel`, Plotly interactive fan charts, local multiprocessing `JobRunner`, FastAPI REST service (ADR-029) |
| **Learned-Dynamics Eval Harness**| **Experimental (v1.2.0)** | Multi-step rollout divergence, invariant verification, and dataset collection (ADR-019) |
| **Torch Neural Residual Baseline**| **Experimental (v1.2.0)** | PyTorch MLP residual baseline with symlog scaling (`[ml]` extra, ADR-019) |
| **Scientific Benchmark Families**| **Research (v1.2.0)** | 5 scientific shift benchmark families probing structural dynamics under change |
| **Planning & Controller Layer** | **Experimental (v1.3.0)** | Pluggable rollout scorers (CVaR, constraint-penalized) & receding-horizon control (ADR-020) |
| **OOD & Regime-Shift Detection** | **Experimental (v1.4.0)** | Grounded-regime detection (support bounds, Mahalanobis covariance, ADR-021) |
| **Honest Causal Diagnostics** | **Experimental (v1.4.0)** | Backdoor identifiability, positivity checks, and Twin Rollout noise coupling (ADR-021) |
| **Counterfactual SCM Kernel** | **Active (v1.6.0)** | Native Pearlian SCM, Abduction-Action-Prediction with coupled noise, identifiability gate (FEAT-008, ADR-040) |
| **Grey-Box Dynamics & Conservation** | **Active (v1.6.0)** | Residual-only learning, hard nullspace projection for conservation laws (FEAT-010, ADR-042) |
| **Causal-Correctness & Metamorphic Suite** | **Active (v1.6.0)** | do(x)-descendant selectivity, label-free MR cards, epistemic abstention, Physics-IQ (FEAT-012, ADR-044) |
| **Decision Scenario Discovery** | **Planned (v1.7.0)** | PRIM bump-hunting vulnerability boxes, multi-robustness metric comparison, EFE (FEAT-009, ADR-041) |
| **Amortized Calibration & TARP Coverage** | **Planned (v1.8.0)** | Simulation-Based Inference, certified coverage via TARP, misspecification refusal (FEAT-011, ADR-043) |
| **Scientific Results Gallery & Benchmarks** | **Planned (v2.0.0)** | CI-reproducible gallery, NeurIPS D&B checklist, arXiv companion methods paper (FEAT-012, ADR-044) |
| **Heterogeneous Graph State & GNN**| **Planned (v2.0.0-alpha)**| Relational graph state representation, schema migration, and GNN dynamics (FEAT-003, ADR-022) |
| **World Specification Language**| **Planned (v2.0.0-alpha)**| Declarative safe YAML/JSON grammar, validator, compiler, and exporter (FEAT-003, ADR-023) |

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

## Platform Completeness & Transformative Roadmap (v1.2 — v2.0)

See [01_EXPANDED_ROADMAP.md](01_EXPANDED_ROADMAP.md) for platform tracks T1–T9.
See [02_ROUND5_EXPANDED_ROADMAP.md](02_ROUND5_EXPANDED_ROADMAP.md) for transformative architecture capabilities and milestone sequencing.

### v1.2.0 — Durability & Reproducibility (Tracks T1, T2, T9) [Shipped]
- **T1: Event-Sourced `TraceLog` & Replay (CORE)**: Append-only transition log with versioned canonical event schemas; `EventStore` protocol (`append/read/fold`) with stdlib SQLite and JSON/YAML backends. Columnar sink to Parquet/DuckDB (`[analytics]` extra).
- **T2: Fingerprint-Keyed `ResultStore` & Memoization (CORE)**: Deterministic rollout memoization keyed by cryptographic scenario/state fingerprints. Filesystem backend with `.npy` sidecars; Redis/S3 adapters (`[storage]` extra).
- **T9: Native Cards & Experiment Tracking (CORE)**: Pydantic-based Scenario, Model, and Dataset Cards with embedded SHA-256 fingerprints; `TrackerBackend` protocol with local JSON logging; MLflow/W&B/DVC adapters.

### v1.3.0 — Trajectory Verification & Temporal Logic (Track T3) [Shipped]
- **T3: Trajectory Verification (CORE / EXTRA)**:
  - **Oracle-Graph Verifier (CORE)**: DAG-based consistency, causality, and timing verification over systemic traces.
  - **STL Robustness Monitoring (`[stl]` extra)**: Signal Temporal Logic (STL/MTL) verification via RTAMT, computing quantitative robustness margins and rollout distributions.

### v1.4.0 — Scientific Experimentation Suite (Track T4) [Shipped]
- **T4: Design of Experiments, Calibration & Optimization (CORE / EXTRA)**:
  - **DoE & Backtesting (CORE)**: Factorial/LHS/OAT parameter sweep harness keyed by `SeedSequence`; walk-forward validation with moment-matching, CRPS, and coverage scoring.
  - **Global Sensitivity (`[sensitivity]` extra)**: Sobol, Morris, and FAST indices via SALib.
  - **Calibration & Emulation (`[calibrate]` extra)**: Distance-based calibration via SciPy/scikit-learn and Gaussian Process emulation.
  - **Intervention Optimization (`[opt-evolutionary]`, `[opt-pareto]` extras)**: CMA-ES via pycma, multi-objective Pareto optimization via pymoo.

### v1.5.0 — Interoperability, Multi-Agent & Serving (Tracks T5, T6, T7, T8) [Shipped]
- **T5: Co-Simulation & Standards (CORE / ADAPTER)**: Master co-simulation stepping loop (CORE); FMI 3.0 / FMU adapter via FMPy (`[fmi]`), discrete-event via SimPy (`[simpy]`), agent-based via Mesa (`[mesa]`), system dynamics via PySD (`[sd]`).
- **T6: Multi-Agent Coordination & Game Theory (CORE / EXTRA)**: Multi-actor observation/action views and deterministic mediator (CORE); 2-player Nash and replicator dynamics via Nashpy (`[game]` extra); PettingZoo and Concordia adapters.
- **T7: Visualization & Reporting (CORE / EXTRA)**: Pydantic `ReportModel` (CORE); interactive Plotly fan charts and comparison views (`[viz]` extra).
- **T8: Simulation-as-a-Service & Orchestration (CORE / EXTRA)**: Local `JobRunner` with multiprocessing (CORE); FastAPI REST serving with fingerprint caching (`[serve]` extra); Prefect/Dagster pipe integration.

### v1.5.x — Positioning & Narrative [Shipped]
- **Positioning & Identity**: Mechanism-first README, neural-world-model disambiguation, CI-generated proof hero, curated GitHub topics, `docs/positioning/landscape.md`.
- **Executive Narrative & Governance**: `docs/executive/` (intellectual lineage, decision-grade guarantees, C-suite use-case gallery, maturity model, reference architectures) + `docs/governance/compliance-mapping.md` (NIST AI RMF, EU AI Act).
- **Handbook Scaffold**: Initial `docs/handbook/` question-organized structure + "When to Refuse a Claim" flagship honesty chapter.

### v1.6.0 — Causal Epistemics & Learned Dynamics [Active]
- **Counterfactual SCM Kernel (Keystone)**: Native Structural Causal Model with addressable seeded exogenous noise; Pearl's Abduction-Action-Prediction Twin-Rollout with coupled noise; non-unique-abduction distributions; ID-gate integration. Theoretical license: Richens & Everitt (ICLR '24 oral, arXiv:2402.10877).
- **Grey-Box Dynamics & Conservation**: Residual-only learned dynamics (`DynamicsModel` with zero residual ≡ symbolic kernel, fingerprint-identical) + hard architectural nullspace projection ensuring zero mass/flow violations by construction.
- **Causal-Correctness & Metamorphic Suite**: `do(x)`-descendant selectivity gates, label-free metamorphic test cards (MR-cards), mutation adequacy testing, and first-class epistemic abstention.
- **Verification Metrics**: NumPy-only verification metrics (rollout error compounding, CRPS, PIT/ECE calibration) and identifiability gate + E-values.

### v1.7.0 — Decision Intelligence (`ewm decide`) [Planned]
- **Scenario Discovery & Robust Decision-Making**: PRIM bump-hunting vulnerability boxes and CART scenario discovery; multi-robustness metric comparison (minimax regret, CVaR regret, satisficing) with explicit refusal to over-rank when metrics conflict; optional Expected Free Energy (EFE) scoring; `ewm decide` CLI and decision reporting.
- **Scientific Benchmarks**: Scored shift families under reproducible protocol.

### v1.8.0 — Calibration to Reality (`ewm calibrate`) [Planned]
- **Amortized Calibration with Certified Coverage**: Simulation-Based Inference (SBI) wrapper over the engine; amortized posterior estimation over `WorldSpec` parameters; mandatory TARP and Posterior-SBC coverage certification; misspecification-robust scoring; `ewm calibrate` CLI.
- **Validation Packs**: Empirical validation targets (SEIR, Beer-Game, WORLD3) and batched/vectorized Monte Carlo engine for scalable simulation budgets.

### v1.9.0 — Flywheel & Neuro-Symbolic Maturation [Planned]
- **Symbolic Promotion Flywheel**: Equation discovery and residual extraction via sparse regression (SINDy/PySR) with human-in-the-loop spec-driven promotion workflow (LEARNED → KNOWN).
- **End-to-End Abstention**: Out-of-competence signaling integrated across `ewm decide` and `ewm calibrate`.
- **Handbook Completion**: Four-part Diátaxis documentation, client-side Distill-style interactive SVG explorables, and complete whole-game tutorial ladder.

### v2.0.0 — Reference Release & Standard [Horizon]
- **Stable Core Graduation**: Counterfactual SCM kernel and grey-box residual/conservation contract graduated to **Stable** under SemVer guarantees and contract gates.
- **Capstone Deliverables**: CI-reproducible `docs/results/` gallery with SHA-256 fingerprints, arXiv methods/benchmark companion paper (NeurIPS D&B checklist QA), reproducible competitive comparison harness, and launch assets.
- **Relational GNNs & World Specification Language**: Heterogeneous relational graph state & GNN dynamics adapters (`FEAT-003`, `ADR-022`), declarative safe WSL language (`ADR-023`), and WMIF interchange format with conformance test suite.
