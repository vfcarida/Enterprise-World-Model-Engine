# Platform Architecture & Capability Tracks (T1–T9)

> Architectural specification defining the platform-completeness tracks (T1–T9) extending EWM Engine with event-sourced persistence, trajectory verification, scientific experimentation, co-simulation, multi-agent coordination, and provenance.
> Every capability is classified as **CORE** (adhering strictly to the `pydantic`/`numpy`/`pyyaml` runtime dependency budget), **EXTRA** (optional package extra), or **ADAPTER** (isolated plugin with strict timeout/boundary safeguards).

---

## 1. Context & Architectural Scope

The foundational releases established the deterministic core kernel:
- **Core Stability:** Deterministic Monte Carlo simulation, immutable state branching, constraint engine, pluggable dynamics, systemic traces, and causal-epistemics evaluation.
- **Ecosystem Scale:** Distributed Monte Carlo execution (preserving `SeedSequence` determinism across workers), OpenTelemetry lifecycle metrics, and mathematical programming planners (CP-SAT, SciPy).

This document specifies the **platform and workflow layer (Tracks T1–T9)** required for enterprise and research-grade simulation: event-sourced persistence, calibration to empirical data, design of experiments, intervention optimization, trajectory verification, platform interoperability, visualization, serving, and provenance tracking.

**Architectural Cohesion:** The engine's **canonical fingerprints, `SeedSequence` determinism, immutable branching, and phase-aware constraints** serve as the structural backbone for these capabilities. They enable event-sourced replay, reproducible backtesting, result caching, orchestrator cache keys, and oracle-graph verification with zero overhead in core. The governance rule is: **establish abstract protocols in core (zero external dependencies), and land heavy ecosystem integrations as isolated adapters.**

---

## 2. Platform Capability Tracks (T1–T9)

### Track T1 — Persistence & Event-Sourced Replay
**Motivation:** High-assurance research and enterprise decision support require durable, replayable, and resumable simulation trajectories. An immutable, fingerprinted `WorldState` plus a transition event naturally forms an append-only event store `(parent_fingerprint → event → child_fingerprint, seed, metadata)`.
- **CORE:** Append-only `TraceLog` capturing state transitions under a versioned, canonical event schema ensuring bitwise replay across engine versions; deterministic replay and branching-as-forks; dependency-free SQLite and JSON/YAML stores.
- **CORE Protocol:** `EventStore` (`append`, `read`, `fold`) enabling pluggable backend storage.
- **EXTRA `[analytics]`:** Columnar trajectory sink to Parquet and DuckDB for downstream analytical querying.
- **ADAPTERS:** EventStore-compliant enterprise backends.
- **References:** Meta ARE (arXiv:2509.17158); DuckDB; Apache Arrow.

### Track T2 — Fingerprint-Keyed Result Store & Memoization
**Motivation:** Monte Carlo rollouts are computationally intensive and deterministic. A state/scenario cryptographic fingerprint provides an ideal, content-addressed cache key for instant memoization.
- **CORE:** `ResultStore` protocol and filesystem backend (`get`/`put` by scenario fingerprint, structured JSON/YAML with `.npy` binary arrays); automatic memoization of rollouts and counterfactual branches.
- **EXTRA `[storage]`:** Distributed Redis and Amazon S3 caching backends.
- **References:** Prefect cache policies; Dagster `DataVersion`.

### Track T3 — Trajectory Verification: Oracle-Graph & Temporal Logic
**Motivation:** While step-by-step constraint checks evaluate local state validity, evaluating whole-trajectory temporal requirements ("SLA recovers within $k$ steps of an intervention", "capacity floor is never breached consecutively") requires formal trajectory monitoring.
- **CORE:** **Oracle-Graph Verifier** checking consistency, causality, and timing invariants over systemic traces as a directed acyclic graph (DAG), paired with a restricted discrete Signal Temporal Logic (STL) fragment in pure NumPy.
- **EXTRA `[stl]`:** Full Signal Temporal Logic (STL/MTL) verification via RTAMT, computing Boolean verdicts and quantitative robustness margins, aggregated into empirical robustness distributions.
- **ADAPTER:** Spatio-temporal logic adapters (MoonLight) and host-provided async verification callables.
- **References:** RTAMT (STL robustness); Meta ARE oracle-graph verification.

### Track T4 — Scientific Experimentation Suite: DoE, Sensitivity, Calibration & Optimization
**Motivation:** Transitioning from "simulate once" to "systematic exploration." Parameter space exploration, sensitivity analysis, model calibration, and intervention optimization share a unified parameter-sweep substrate.
- **CORE:** Design of Experiments (DoE) sweep harness (Full Factorial, Latin Hypercube Sampling, One-at-a-Time sweeps) keyed by `SeedSequence`; walk-forward backtesting with scoring (moment-matching distance, CRPS, empirical coverage); abstract protocols for `Optimizer`, `Surrogate`, and `Sampler`.
- **EXTRA `[sensitivity]`:** Global sensitivity analysis via SALib (Sobol, Morris, FAST).
- **EXTRA `[calibrate]`:** Distance-based parameter calibration via SciPy/scikit-learn.
- **EXTRA `[opt-evolutionary]`, `[opt-pareto]`:** CMA-ES optimization (pycma) and multi-objective Pareto optimization (pymoo).
- **ADAPTERS:** Neural simulation-based inference (SBI) and Bayesian optimization (BoTorch).
- **References:** SALib; pycma; pymoo; Simulation-Based Inference (sbi).

### Track T5 — Co-Simulation & Interoperability Standards
**Motivation:** Organizations run specialized models across diverse software platforms (system dynamics, discrete-event, vendor FMUs). EWM Engine couples with external simulators rather than requiring total reimplementation.
- **CORE:** Co-simulation master stepping loop with typed variable exchange and interpolation envelopes.
- **ADAPTER `[fmi]`:** Functional Mock-up Interface (FMI 3.0 / FMU) co-simulation adapter via FMPy with sandboxed binaries.
- **ADAPTER `[simpy]`, `[mesa]`, `[sd]`:** Adapters for SimPy (discrete-event), Mesa (agent-based), and PySD (System Dynamics XMILE models).
- **References:** Functional Mock-up Interface 3.0; FMPy; PySD; SimPy; Mesa.

### Track T6 — Multi-Agent Coordination & Game-Theoretic Framework
**Motivation:** Complex organizational systems involve competing or collaborating decision-makers. EWM Engine supports structured multi-agent observation views, turn-taking, and rule adjudication.
- **CORE:** Multi-actor observation and action views over immutable `WorldState`; `Mediator` protocol with deterministic default rules leveraging the hard/soft constraint engine.
- **EXTRA `[game]`:** Game-theoretic analysis via Nashpy (2-player Nash equilibrium and replicator dynamics).
- **ADAPTERS:** PettingZoo standard interfaces and Concordia Game-Master environments.
- **References:** PettingZoo; Nashpy; Concordia (arXiv:2312.03664).

### Track T7 — Visualization & Reporting Framework
**Motivation:** Executive and technical decision-makers require clear scenario comparison dashboards without entangling visualization dependencies into the core mathematical engine.
- **CORE:** Declarative Pydantic `ReportModel` encapsulating trajectories, quantiles, branch diffs, and constraint violations as renderer-neutral data.
- **EXTRA `[viz]`:** Interactive fan charts and multi-scenario comparison views using Plotly.
- **EXTRA `[cli]`:** Terminal tables and tree visualization via Rich.
- **ADAPTER `[dashboard]`:** Interactive web applications via Streamlit or Dash calling the public API.
- **References:** Plotly; Vega-Lite; Rich.

### Track T8 — Simulation-as-a-Service & Orchestration
**Motivation:** Support scalable batch execution and remote execution over HTTP while retaining cryptographic fingerprint guarantees.
- **CORE:** Local multiprocessing `JobRunner` with seed-tree isolation and automatic result store caching.
- **EXTRA `[serve]`:** Production FastAPI REST service (`POST /simulate`, `GET /results/{fingerprint}`).
- **ADAPTERS `[prefect]`, `[dagster]`:** Workflow orchestration adapters passing state fingerprints directly as pipeline data versions.
- **References:** FastAPI; Prefect; Dagster Pipes.

### Track T9 — Reproducibility Artifacts & Experiment Tracking
**Motivation:** Scientific transparency and auditability demand self-contained, machine-readable documentation of model assumptions and execution environments.
- **CORE:** Native Scenario, Model, and Dataset **Cards** (Pydantic models exporting to YAML, JSON, and Markdown) tracking intended uses, boundaries, metrics, and state fingerprints; local file-based tracker backend.
- **EXTRA `[config]`:** Dynamic configuration interpolation via OmegaConf.
- **ADAPTERS:** MLflow, Weights & Biases, and DVC experiment tracking integrations.
- **References:** Model Cards for Model Reporting (arXiv:1810.03993); Datasheets for Datasets (arXiv:1803.09010).

---

## 3. Release Sequencing & Architectural Order

Platform tracks are scheduled to establish durable storage and verification primitives before running large-scale experimentation:

| Milestone | Subsystem Focus | Shipped / Planned Tracks | Architecture References |
| :--- | :--- | :--- | :--- |
| **v1.1.0** | Scale & Ecosystem | Distributed Monte Carlo, OpenTelemetry, OR Planners | FEAT-001, FEAT-002, ADR-016–ADR-018 |
| **v1.2.0** | Durability & Provenance | T1 TraceLog, T2 ResultStore, T9 Cards & Protocols | FEAT-004, ADR-026, ADR-029 |
| **v1.3.0** | Trajectory Verification | T3 Oracle-Graph Verifier, RTAMT STL Monitoring | FEAT-005, ADR-027 |
| **v1.4.0** | Scientific Experimentation | T4 DoE Sweeps, Sensitivity, Calibration, Optimizers | FEAT-006, ADR-028 |
| **v1.5.0** | Interoperability & Serving | T5 Co-Simulation, T6 Multi-Agent, T7 Reporting, T8 Serving | FEAT-007, ADR-029 |
| **v1.6.0** | Causal Epistemics | Counterfactual SCM Kernel, Grey-Box Conservation, Metamorphic Testing | FEAT-008, FEAT-010, FEAT-012 |
| **v2.0.0** | Reference Architecture | Relational Graph States, World Specification Language, Stable Kernel | FEAT-003, ADR-022, ADR-023 |

---

## 4. Architectural Boundaries & Dependency Isolation

To maintain high runtime reliability, heavy third-party packages must remain quarantined in optional extras:
1. **Zero-Dependency Core:** Core modules (`core`, `simulation`, `constraints`, `provenance`, `evaluation`) must never import PyTorch, DuckDB, FastAPI, or solver packages.
2. **Deterministic Fallbacks:** When optional extras are not installed, the engine gracefully falls back to deterministic pure-Python or NumPy primitives without raising unexpected errors.
3. **Execution Resource Bounds:** All external planners, solvers, and simulation adapters must enforce explicit execution timeouts and memory bounds to protect host processes.

---

## 5. Maturity Model & Release Governance

Every capability in EWM Engine advances through an objective, automated maturity lifecycle:

- **Stable**: Frozen public API contract covered by backward-compatibility guarantees, automated contract gates (`AC-024`), and formal deprecation lifecycles.
- **Beta Adapter / Beta**: Tested, production-grade ecosystem components with formal resource limits and comprehensive test coverage.
- **Experimental**: Prototyping and scientific research modules located under `ewm_engine.experimental.*`; subject to design evolution across minor releases.
- **Research**: Methodological benchmarks, verification suites, and exploratory validation packs.

### Anti-Overclaiming Policy
All documentation, benchmark reports, and capability descriptions must reflect verified empirical evidence. Promotional buzzwords ("state-of-the-art", "world's first", "guarantees optimal decisions") are strictly prohibited. Every claim of causal identifiability, coverage calibration, or invariant preservation must be accompanied by a reproducible test script and committed seed configuration.
