# EWM Engine — Expanded Roadmap (new feature tracks)

> Prepared: 2026-10-05. Companion to `00_ANALYSIS_AND_EVOLUTION_PLAN.md`.
> Purpose: analyze the **current** roadmap, then amplify it with researched, well-justified feature tracks that make the engine a more *complete* research-grade simulation & decision platform.
> Every item is tagged **CORE** (fits the `pydantic`/`numpy`/`pyyaml` budget), **EXTRA** (optional pip extra), or **ADAPTER** (isolated plugin, never a default install), per the project's dependency discipline.

---

## 1. The current roadmap (verified) and what it already covers

From `README.md` (Project Roadmap & Maturity) and `ROADMAP.md`:

- **Shipped / Stable (v1.0.0):** core simulation & branching, constraint engine, pluggable dynamics, systemic traces, causal-epistemics evaluation. **Beta:** learned-dynamics protocol + linear baseline. **Alpha:** Gym/RL, Z3/OR-Tools adapters.
- **Near-term (v1.1):** OR adapters (OR-Tools, SciPy), distributed Monte Carlo (Ray/multiprocessing), OpenTelemetry.
- **Long-term (v2.0+):** latent dynamics (RSSM/Dreamer, JEPA, GNN), causal epistemics & off-policy evaluation, OOD/regime-shift detection, differentiable constraints.

The prompt package `P01`–`P12` operationalizes that core path. **The gap this document fills:** the current roadmap is strong on *scale* and *learned/causal depth* but largely silent on the **platform/workflow layer** that turns a good kernel into a complete research instrument — persistence, calibration-to-data, experiment design, intervention optimization, trajectory verification, interoperability, visualization, serving, and reproducibility artifacts.

**Cross-cutting thesis (why these fit the engine so well):** the existing **canonical fingerprints + `SeedSequence` determinism + immutable branching + hard/soft constraints** are the connective tissue for nearly every feature below. They make event-sourced replay, reproducible backtesting, result caching, orchestrator cache keys, and oracle-graph verification *almost free in core*. The strategy is: **establish the protocols now (CORE, zero-dep), land every heavy integration later as an isolated adapter.**

---

## 2. New feature tracks

Each track: motivation, concrete deliverables with tags, key references (verified 2026-10-05), and the governing prompt.

### Track T1 — Persistence & event-sourced replay  → `P13`
**Why:** today a run lives in memory. A research platform needs durable, replayable, resumable trajectories. The engine is already ~90% an event store: an immutable, fingerprinted `WorldState` + a transition is exactly `(parent_fingerprint → event → child_fingerprint, seed, metadata)`. Meta ARE's "everything is an event" (DAG + append-only EventLog, state = fold over events) is the validated pattern.
- **CORE:** append-only `TraceLog` of transitions with a *versioned canonical event schema* so logs replay across engine versions; deterministic replay; branching-as-DAG-forks; a `JSON/YAML` store + a stdlib `sqlite` store (still dep-free).
- **CORE protocol:** `EventStore` (`append/read/fold`) so external stores plug in.
- **EXTRA `[analytics]`:** columnar trajectory sink to Parquet/DuckDB (PyArrow/DuckDB — ~80 MB, lazy-import, quarantined).
- **ADAPTERS:** `eventsourcing`, KurrentDB.
- Refs: Meta ARE (arXiv:2509.17158, MIT repo); DuckDB 1.5.x; PyArrow 25.x; `eventsourcing` 9.5.x.

### Track T2 — Fingerprint-keyed result store & memoization  → `P13`
**Why:** rollouts are expensive and deterministic — so they are perfectly cacheable, and the engine already computes the ideal cache key (the scenario/state fingerprint). Prefect/Dagster approximate this with input hashes; the engine *has* it natively.
- **CORE:** `ResultStore` protocol + filesystem backend (`get/put` by fingerprint, JSON/YAML + numpy `.npy` sidecars); memoize rollouts & branches.
- **EXTRA:** Redis/S3 backends.
- Refs: Prefect 3.x cache policies; Dagster `DataVersion`.

### Track T3 — Trajectory verification: oracle-graph + temporal logic  → `P14`
**Why:** the engine detects *constraint* violations step-by-step but cannot yet express *temporal* requirements over a whole trajectory ("whenever demand spikes, SLA recovers within k steps", "a shelter is never over capacity for >2 consecutive steps"). This is a genuine differentiator and reuses hard/soft constraints + `EvidenceLevel`.
- **CORE:** an **oracle-graph verifier** (ARE pattern: oracle events as a DAG; Consistency/Causality/Timing checks over systemic traces — pure pydantic/stdlib) + a *restricted bounded-future discrete STL fragment* monitor in numpy, plus a backend-agnostic property-spec whose hash folds into the fingerprint.
- **EXTRA `[stl]`:** full STL/MTL via RTAMT, reporting Boolean verdict **and quantitative robustness margin**, aggregated into robustness *distributions* across Monte Carlo rollouts.
- **ADAPTER:** MoonLight for spatio-temporal (STREL) on topological worlds (JVM — isolated); optional LLM soft-check via the existing async-callable pattern (no SDK in core).
- Refs: RTAMT (BSD-3; STL robustness; caps at Py≤3.12 → isolate); `py-metric-temporal-logic`; MoonLight 0.3; ARE oracle-graph (0.98 agreement over 450 trajectories).

### Track T4 — Experimentation suite: DoE, sensitivity, calibration, optimization  → `P15`
**Why:** "simulate once" → "systematically explore." This is the scientific-method layer: design experiments, measure what matters, fit to data, and search intervention space. All share one numpy-only sweep substrate keyed by `SeedSequence`.
- **CORE:** DoE/sweep harness (factorial/LHS/OAT over declared params → reproducible rollouts; tidy results + tornado-diagram data); a validation/backtest harness (walk-forward hindcasting; numpy scorers: method-of-moments distance, empirical coverage/SBC concept, RMSE/CRPS, spectral distance); `Optimizer`, `Surrogate`, `Sampler` **protocols** + one trivial built-in each (random+hill-climb; sklearn-free).
- **EXTRA `[sensitivity]`:** global SA via SALib (Sobol/Morris/FAST). `[calibrate]`: `DistanceCalibrator` (scipy/sklearn, **no torch**) + optional pyabc ABC-SMC. `[opt-evolutionary]`: pycma + Nevergrad (light, default). `[opt-pareto]`: pymoo multi-objective `ParetoFront` (fingerprinted policies). `[surrogate]`: sklearn-GP emulator.
- **ADAPTERS (heavy, quarantined):** `sbi` neural simulation-based inference (torch); Ax/BoTorch Bayesian opt (torch); GPyTorch/Emukit emulation.
- **License flag:** `black-it` is **AGPL-3.0** — reimplement its trivial MSM loss, do not vendor it.
- Refs: SALib 1.6; pycma 4.5; Nevergrad 1.0.x; pymoo 0.6.2; sbi 0.27 (arXiv:2508.12939); Ax 1.3/BoTorch 0.18.

### Track T5 — Co-simulation & interoperability standards  → `P16`
**Why:** real organizations already run models in other tools (system-dynamics, discrete-event, vendor FMUs). To be a *platform*, the engine must couple to them rather than demand reimplementation. The co-sim *master* contract (stepping multiple models, typed per-tick variable exchange, zero-order-hold vs. interpolated coupling) is pure scheduling + dicts — core-friendly.
- **CORE:** co-simulation master loop + model-exchange envelope (FMI "co-simulation master" pattern), pydantic-only.
- **ADAPTER `[fmi]`:** `FMUDynamicsModel` via FMPy → drop-in access to 280+ FMI tools (fold FMU GUID + modelDescription hash into fingerprint; sandbox binaries). `[simpy]` (zero-dep, default sub-engine), `[mesa]`, `[sd]` (PySD + a `ewm import-sd model.xmile` CLI for non-programmer domain experts).
- Refs: FMI 3.0 (Modelica Association; Reference-FMUs); FMPy 0.3.x (MIT, recommended runtime); PySD 3.14; SimPy 4.1; Mesa 3.5.

### Track T6 — Multi-agent & game-theoretic  → `P16`
**Why:** many socio-technical systems are inherently multi-actor; the engine has single `Actor`s but no first-class multi-agent turn/parallel model, equilibrium analysis, or a mediator that adjudicates competing proposals. The Concordia "Game Master" maps *exactly* onto the engine's constraint engine as a deterministic rules adjudicator.
- **CORE:** lightweight multi-actor model (per-actor action/observation views over immutable `WorldState`); a `Mediator`/`GameMaster` **protocol** with a **deterministic default that reuses the hard/soft constraint engine** (no LLM).
- **EXTRA `[game]`:** Nashpy (pure numpy/scipy) for 2-player Nash + replicator dynamics from actor outcome metrics. **ADAPTERS:** PettingZoo (AEC/Parallel) surface; OpenSpiel (n-player, C++ wheels); Concordia LLM Game-Master (Py≥3.12 + LLM — fully isolated).
- Refs: PettingZoo 1.27 (Farama; arXiv:2009.14471); Nashpy 0.0.43; OpenSpiel 2.0; Concordia 2.4 (arXiv:2312.03664, 2507.08892).

### Track T7 — Visualization & reporting  → `P16`
**Why:** the one static HTML trace is not enough for decision work; but the core must stay viz-free. Solve with a renderer-neutral data model.
- **CORE:** a pydantic `ReportModel` (trajectories, uncertainty quantiles, branch tree, violation events) — pure data, zero deps.
- **EXTRA `[cli]`:** Rich terminal tables/trees. `[viz]`: Plotly (tiny core; `write_html()`, no server) fan charts + interactive scenario-comparison; optional Altair backend emitting fingerprintable Vega-Lite JSON. `[viz-export]`: static images (kaleido/Chromium — heavy, separate).
- **ADAPTER `[dashboard]`:** Streamlit/Dash front-end that only calls the public API.
- Refs: Plotly 7.x; Vega-Altair 6.x; Rich 15.x.

### Track T8 — Simulation-as-a-service & orchestration  → `P16`
**Why:** teams want to submit scenarios and retrieve results without embedding the library; and large studies need batch orchestration. Fingerprint caching makes a service fast and idempotent.
- **CORE:** local `JobRunner` (multiprocessing + `SeedSequence.spawn()`), reusing the `ResultStore` from T2.
- **EXTRA `[serve]`:** FastAPI REST (`POST /simulate` → id+fingerprint; `GET /results/{fingerprint}` → instant cache hit). **Do not** use `BackgroundTasks` for rollouts.
- **ADAPTERS `[prefect]`/`[dagster]`:** wire the engine fingerprint directly into their cache-key / `DataVersion` (their keys *are* input hashes — zero translation).
- Refs: FastAPI 0.14x; Prefect 3.x; Dagster 1.13.x (`dagster-pipes`).

### Track T9 — Reproducibility artifacts & experiment tracking  → `P13` (cards) + `P16` (trackers)
**Why:** research credibility requires portable, standardized provenance. The Google Model Card Toolkit is **archived (2024)** — so native, pydantic-based cards are both the right call and a real gap-filler.
- **CORE:** native Scenario/Model/Dataset **Cards** (pydantic → YAML/JSON/Markdown): intended use, assumptions, out-of-scope conditions, constraints exercised, metrics, and the **fingerprint** of the artifact described; optional hand-emitted Croissant 1.1 JSON-LD (stdlib). `TrackerBackend` protocol + dependency-free local JSON logger keyed by fingerprint.
- **EXTRA `[config]`:** OmegaConf (`${...}` interpolation; one light dep). **ADAPTERS:** MLflow / W&B (log fingerprint as a run tag); DVC via CLI shell-out (don't import — ~40 deps).
- Refs: Model Cards (arXiv:1810.03993); Datasheets (arXiv:1803.09010); Croissant 1.1; OmegaConf 2.3; MLflow 3.16.

---

## 3. Proposed release sequencing

The current roadmap keeps v1.1 (scale) and v2.0 (learned/causal). These new tracks slot in as **v1.2–v1.6 "platform completeness"**, mostly *additive* and low-risk because the heavy pieces are adapters.

| Release | Theme | Headline tracks | Prompt |
|---|---|---|---|
| **v1.1** | Ecosystem & scale *(existing)* | distributed MC, OTel, OR adapters | `P03`,`P04`,`P05` |
| **v1.2** | Durability & reproducibility | T1 event-sourced TraceLog, T2 ResultStore, T9 Cards + protocols | `P13` |
| **v1.3** | Trajectory verification | T3 oracle-graph (core) + STL/robustness (extra) | `P14` |
| **v1.4** | Experimentation suite | T4 DoE + backtest + sensitivity + intervention optimization | `P15` |
| **v1.5** | Interop, multi-agent, viz, serving | T5 co-sim/FMI, T6 multi-agent/mediator, T7 ReportModel/Plotly, T8 serve/orchestrate, T9 trackers | `P16` |
| **v1.6+** | Learned depth *(existing v2 agenda)* | learned-dynamics harness, planning layer, OOD, causal, graph/WSL | `P06`–`P10` |
| **v2.0** | Major evolution *(existing)* | graph state, DSL, latent dynamics families | `P10` |

The ordering is deliberate: **durability (v1.2) and verification (v1.3) before experimentation (v1.4)**, because you want every large study to be replayable and checkable before you run thousands of them; and the *protocols* (EventStore, ResultStore, Optimizer, Surrogate, Sampler, JobRunner, TrackerBackend, Mediator) all land early in v1.2 so later heavy integrations are pure adapters.

---

## 4. Highest-value, zero-dependency CORE additions (do first)

If only a handful of items ship, these give the most completeness per unit of dependency risk — all pure `pydantic`/`numpy`/stdlib:

1. **Append-only `TraceLog` / event-sourced replay (T1)** — you're already 90% there via fingerprints + immutability.
2. **Oracle-graph trajectory verifier (T3)** — reuses hard/soft constraints + `EvidenceLevel`; strong differentiator vs. ARE.
3. **Fingerprint-keyed `ResultStore` (T2)** — immediate performance win; you're *ahead* of Prefect/Dagster here.
4. **DoE/sweep + validation/backtest harness (T4)** — the numpy-only substrate for calibration, SA, and optimization.
5. **The protocol set** (`EventStore`, `ResultStore`, `Optimizer`, `Surrogate`, `Sampler`, `JobRunner`, `TrackerBackend`, `Mediator`) with trivial built-ins — establishes contracts now so every heavy integration lands isolated later.
6. **Native Scenario/Model/Dataset Cards (T9)** — pydantic-only; the archived MCT leaves a real gap.

---

## 5. Dependency landmines to quarantine (core tests must pass without them)

- **PyTorch stack** — sbi, blackbirds, GPyTorch, Ax/BoTorch, neural dynamics.
- **PyArrow/DuckDB binaries (~80 MB)** — analytics sink.
- **Web-server stacks** — FastAPI serving, Prefect, Dagster, MLflow, Streamlit/Dash.
- **JVM** — MoonLight (STREL), HLA/RTI.
- **License:** `black-it` is **AGPL-3.0** (reimplement, don't vendor); use **DVC via CLI**, not import.
- **Python-version pins:** RTAMT caps at Py≤3.12 (ANTLR), Concordia needs Py≥3.12 — both stay in isolated adapters so neither constrains the core's supported range.

---

## 6. Updated README roadmap table (drop-in replacement)

The agent implementing these should replace the README "Project Roadmap & Maturity" table with one that reflects the expanded plan — see `P13`–`P16` and `P11` for the exact wording and maturity labels. Keep the forbidden-claims discipline: label every new track honestly (CORE Stable only after its ACs pass; EXTRA/ADAPTER as Beta/Alpha), and never describe a planned track as shipped.
