# ADR-029: Interop, Multi-Agent, Reporting, Serving, and Experiment Trackers

## Status
Accepted

## Date
2026-10-06

## Context
Up to v1.4, the Enterprise World Model Engine operated primarily as an in-process, single-agent simulation and optimization engine. As the engine moves to v1.5, it must interface with complex enterprise ecosystems:
1. **Co-Simulation (Track T5):** Coupling with external domain simulations (SimPy discrete event, Mesa agent-based, PySD system dynamics, and FMI vendor FMUs) requires standard master loops, typed variable exchanges, and zero-order-hold interpolation.
2. **Multi-Agent Interaction & Adjudication (Track T6):** Organizations feature competing departments and decentralized actors. The engine must support partial observability views, simultaneous action proposals, and deterministic constraint adjudication without relying on unpredictable LLM arbiters.
3. **Visualization & Reporting (Track T7):** Domain experts need renderer-neutral data structures representing trajectory quantiles, fan charts, branch trees, and constraint violations, rendered across Rich terminal CLI, interactive Plotly HTML, Vega-Lite JSON specifications, and static images.
4. **Simulation-as-a-Service (Track T8):** High-scale execution requires thread/process pool concurrency, child seeding with `SeedSequence.spawn()`, exact ResultStore memoization, and a standard FastAPI REST API compatible with orchestrators like Prefect and Dagster.
5. **Experiment Trackers & Hierarchical Configs (Track T9):** Standard experiment run logging (`.ewm_runs/{fingerprint}/run.json`), hierarchical YAML configs with variable interpolation (`OmegaConf`), and enterprise tracking integrations (MLflow, Weights & Biases).

## Decision

### 1. Dedicated Packages & Zero-Dependency Core Substrates
We introduce five focused packages:
- `ewm_engine.cosim` (Co-simulation master and exchange envelopes)
- `ewm_engine.multiagent` (Role-specific views and deterministic mediator)
- `ewm_engine.reporting` (Renderer-neutral `ReportModel` and Vega-Lite specs)
- `ewm_engine.service` (LocalJobRunner and FastAPI app factory)
- `ewm_engine.trackers` (TrackerBackend protocol, LocalJsonTracker, and OmegaConf loader)

The core implementations of all five packages rely exclusively on Python standard library, Pydantic, and NumPy.

### 2. External Tools Quarantined Behind Optional Extras
All third-party tools are strictly quarantined behind optional extras in `pyproject.toml`:
- `[fmi]` (`fmpy`)
- `[simpy]` (`simpy`)
- `[mesa]` (`mesa`)
- `[sd]` (`pysd`)
- `[game]` (`nashpy`)
- `[cli]` (`rich`)
- `[viz]` (`plotly`)
- `[viz-export]` (`kaleido`)
- `[serve]` (`fastapi`, `httpx`)
- `[config]` (`omegaconf`)
- `[trackers]` (`mlflow`)
- `[interop]` (bundle of key ecosystem packages)

### 3. Deterministic Adjudication without LLM Dependencies
In line with ADR-004, `ConstraintMediator` adjudicates simultaneous multi-agent actions strictly using pre-defined priorities, arrival timestamps, and resource capacity limits, enforcing hard/soft constraints deterministically without invoking non-reproducible language models.

### 4. Canonical Fingerprinting across Ecosystem Boundaries
- Vega-Lite specifications are cryptographically fingerprinted via canonical SHA-256.
- Prefect and Dagster cache keys map 1:1 to EWM Engine canonical simulation fingerprints.
- Experiment runs in `LocalJsonTracker` are keyed by canonical fingerprints (`.ewm_runs/{fingerprint}/run.json`).

## Consequences
- **Positive:** Enables comprehensive end-to-end enterprise deployment from external simulation coupling to REST API serving and interactive reporting.
- **Positive:** Zero new required runtime dependencies added to core engine.
- **Positive:** Preserves 100% deterministic reproducibility and cryptographic auditability across all integration points.
- **Negative:** Users requiring specialized engine features (e.g. FMI, SimPy, Plotly, FastAPI) must install the appropriate optional extras.
