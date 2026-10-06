# FEAT-007: Interop, Multi-Agent, Visualization, Serving, and Trackers

- **Status:** Approved / Implemented
- **Horizon:** v1.5.0 "Interop, Multi-Agent, Viz, Serving & Trackers"
- **Authors:** Platform, Interoperability & Ecosystem Engineering Teams
- **Governance:** `01_EXPANDED_ROADMAP.md` (Tracks T5, T6, T7, T8, T9), ADR-029, ACP-008
- **Dependencies:** Core engine (Zero new dependencies: stdlib, NumPy, Pydantic); Optional extras: `[fmi]` (fmpy), `[simpy]` (simpy), `[mesa]` (mesa), `[sd]` (pysd), `[game]` (nashpy), `[cli]` (rich), `[viz]` (plotly), `[viz-export]` (kaleido), `[serve]` (fastapi, httpx), `[config]` (omegaconf), `[trackers]` (mlflow, wandb)

---

## 1. Motivation & Context

As enterprise world models scale into operational systems, they cannot exist in isolation. Modern decision infrastructure requires:
1. **Co-Simulation & Interoperability (Track T5):** Coupling with external domain models (discrete-event factories, system-dynamics supply chains, vendor Functional Mock-up Units [FMUs]) via typed variable exchange and zero-order-hold interpolation.
2. **Multi-Agent Simulation & Adjudication (Track T6):** Modeling decentralized actors with restricted observation views, simultaneous action proposals, and deterministic constraint-enforced arbitration without reliance on stochastic LLM judges.
3. **Visualization & Reporting (Track T7):** Decoupled renderer-neutral reports (`ReportModel`) capturing quantiles, fan charts, branch trees, and invariant violations across CLI (Rich), interactive HTML (Plotly), and deterministic specifications (Vega-Lite).
4. **Simulation-as-a-Service (Track T8):** High-throughput execution runners (`LocalJobRunner`), background concurrency with `SeedSequence.spawn()`, exact memoization (`ResultStore`), and a production REST API (FastAPI) compatible with orchestrators like Prefect and Dagster.
5. **Experiment Trackers & Structured Configs (Track T9):** Standardized experiment logging (`TrackerBackend`, `LocalJsonTracker`), hierarchical configs with variable interpolation (`OmegaConf`), and enterprise tracker adapters (MLflow, Weights & Biases).

---

## 2. Requirements & Acceptance Criteria

### Sub-track 1: Co-Simulation & Model Exchange (Track T5, Core & Adapters)
- **AC-053 (Co-Simulation Master & Envelopes):**
  - Define `CoSimExchangeEnvelope` with timestamp, time step, inputs, outputs, and status.
  - Define `SubModel` protocol (`initialize`, `step`, `terminate`, `get_state`, `set_state`).
  - Implement deterministic `CoSimMaster` executing fixed-step master simulation loops with zero-order-hold interpolation across registered sub-models.
  - Provide `CoSimDynamicsModel` wrapping co-simulation ensembles into standard EWM `DynamicsModel`.
  - Provide adapters for FMI (FMPy), SimPy, Mesa, and PySD behind optional extras (`[fmi]`, `[simpy]`, `[mesa]`, `[sd]`).
  - Implement CLI command `ewm import-sd <model_file>` extracting variables into EWM schemas.

### Sub-track 2: Multi-Agent Simulation & Mediation (Track T6, Core & Adapters)
- **AC-054 (Multi-Agent Views & Deterministic Adjudication):**
  - Provide `ActorObservationView` defining partial-observability masks (visible entities, masked fields).
  - Provide `ActorActionProposal` capturing actor identity, intent, and resource claims.
  - Implement deterministic `ConstraintMediator` resolving simultaneous conflicting proposals via priority, arrival order, and resource capacity without non-deterministic LLM arbitration.
  - Provide `NashpyGameSolver` behind `[game]` extra computing Nash equilibria and replicator dynamics.
  - Provide standard environment adapters (`PettingZooParallelAdapter`, OpenSpiel, and Concordia stubs).

### Sub-track 3: Visualization & Reporting (Track T7, Core & Renderers)
- **AC-055 (Renderer-Neutral Reporting & Multi-Surface Renderers):**
  - Implement `ReportModel` capturing simulation summary, empirical signal quantiles ($p_{10}, p_{25}, p_{50}, p_{75}, p_{90}, \mu$), branch DAG nodes, and invariant violations.
  - Emit JSON schema `schemas/report-model.schema.json`.
  - Provide Rich table formatter (`render_rich_report`, `format_rich_summary_str`) behind `[cli]`.
  - Provide Plotly fan chart generator with standalone HTML export behind `[viz]`.
  - Provide Vega-Lite JSON specification generator (`generate_vega_fan_chart_spec`) with SHA-256 fingerprinting.
  - Support static PNG/SVG export via Kaleido behind `[viz-export]`.

### Sub-track 4: Simulation-as-a-Service & Serving (Track T8, Core & Service)
- **AC-056 (Concurrent Job Runner & FastAPI REST Endpoints):**
  - Implement `LocalJobRunner` with thread-pool and process-pool concurrency, managing background execution, `SeedSequence.spawn()` child seeding, and exact `ResultStore` caching.
  - Implement FastAPI factory `create_simulation_app` exposing `POST /simulate`, `GET /jobs/{id}`, and `GET /results/{fingerprint}`.
  - Implement Prefect and Dagster cache-key adapters leveraging EWM canonical simulation fingerprints.

### Sub-track 5: Experiment Trackers & Hierarchical Configs (Track T9, Core & Trackers)
- **AC-057 (Tracker Protocol, Local JSON Tracker & Adapters):**
  - Define `TrackerBackend` protocol (`log_fingerprint`, `log_params`, `log_metrics`, `log_card`, `log_artifact`).
  - Implement zero-dependency `LocalJsonTracker` persisting structured runs atomically to `.ewm_runs/{fingerprint}/run.json`.
  - Implement `OmegaConfConfigLoader` supporting YAML configuration loading, `${...}` variable interpolation, and canonical fingerprinting behind `[config]`.
  - Provide adapters for MLflow and Weights & Biases behind `[trackers]`.
