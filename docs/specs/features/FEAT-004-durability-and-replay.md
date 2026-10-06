# FEAT-004: Durability, Event-Sourced TraceLog, ResultStore, and Native Cards

- **Status:** Approved / Implemented
- **Horizon:** v1.2.0 "Durability & Reproducibility"
- **Authors:** Platform Engineering & Scientific Reproducibility Team
- **Governance:** `01_EXPANDED_ROADMAP.md` (Tracks T1, T2, T9), ADR-026, ACP-005
- **Dependencies:** Core engine (Zero new dependencies: stdlib `sqlite3`, `numpy`, `pydantic`, `pyyaml` only)

---

## 1. Motivation & Context

In production enterprise simulation and scientific AI research, simulations must not exist merely as ephemeral in-memory objects. Trajectories must be **durable, replayable, resumable, cacheable, and self-documenting**:
1. **Event-Sourced Replay:** In EWM Engine, `WorldState` is already immutable and canonically fingerprinted; a state transition is formally `(parent_fingerprint → event → child_fingerprint, seed, metadata)`. By formalizing this into an append-only event stream, state reconstruction becomes a deterministic fold over events ($S_t = \text{fold}(S_0, [e_1, \dots, e_t])$). This aligns directly with Meta ARE's "everything is an event" architecture (arXiv:2509.17158).
2. **Branching as a DAG:** Complex decision explorations require branching counterfactuals. By representing branches as forks in the event DAG, branches share common parent history while maintaining strict branch isolation.
3. **Fingerprint-Keyed Memoization:** Simulation rollouts over complex organizational models are computationally expensive. Because EWM Engine computes canonical cryptographic fingerprints over initial states, dynamics versions, and scenario configs, the fingerprint serves as the ideal cache key. This provides exact memoization ahead of workflow orchestrators (Prefect, Dagster) whose cache keys only approximate input state.
4. **Native Reproducibility Cards:** High-stakes decisions demand standardized, auditable documentation. The Google Model Card Toolkit was archived in 2024; native Pydantic cards grounded in Model Cards (arXiv:1810.03993), Datasheets for Datasets (arXiv:1803.09010), and population calibration provenance (arXiv:2411.10109) fill this critical gap without adding external dependencies.

---

## 2. Requirements & Acceptance Criteria

### Part A: Event-Sourced TraceLog & EventStore (Track T1)
- **AC-030 (Event Schema v1):** Define a versioned canonical transition event schema (`TransitionEvent`) locked at `schema_version = "1.0.0"`. Fields must capture `stream_id`, `step`, `parent_fingerprint`, `child_fingerprint`, `timestamp`, `seed`, `actions_proposed`, `actions_accepted`, `exogenous_events`, `transition_result`, `applied_changes`, `constraint_violations`, `state_snapshot`, and `component_versions`.
- **AC-031 (Deterministic Replay):** Implement `fold_events(initial_state, events)` and `replay_trajectory(events)` such that replaying an event stream produces a trajectory and final state that is bitwise and logically identical to the original run (`verify_trajectory_replay == True`).
- **AC-032 (Branch Isolation):** Represent branches as forks in the event DAG (`TraceLog.branch`). Appending events to a branched stream must not alter the trunk or sibling branches.
- **AC-033 (Pluggable EventStore Protocol):** Establish the `EventStore` protocol (`append`, `read_stream`, `get_event`, `list_streams`, `fold`). Implement three zero-dependency backends: `InMemoryEventStore`, `JsonFileEventStore` (with JSON and YAML support), and `SqliteEventStore` (stdlib `sqlite3` with WAL mode).

### Part B: Fingerprint-Keyed ResultStore (Track T2)
- **AC-034 (ResultStore Protocol):** Establish the `ResultStore` protocol (`get`, `put`, `contains`, `delete`, `list_fingerprints`).
- **AC-035 (Filesystem ResultStore Backend):** Implement `FilesystemResultStore` using atomic file replacements (`.tmp` write + rename), storing structured metadata in `result.json` and numerical timeseries in compressed NumPy `.npz` sidecars.
- **AC-036 (Exact Memoization):** Implement `compute_simulation_fingerprint` and `MemoizedSimulationRunner`. Prove that a cache hit produces an outcome identical to a fresh run without re-executing rollouts.
- **AC-037 (Remote Cache Stubs):** Provide documented stubs (`RedisResultStore`, `S3ResultStore`) raising informative exceptions directing users to the `[cache]` extra.

### Part C: Native Cards & Croissant 1.1 (Track T9)
- **AC-038 (Native Pydantic Cards):** Implement `ModelCard`, `ScenarioCard`, and `DatasetCard` models capturing intended use, assumptions, out-of-scope conditions, constraints exercised, metrics, limitations, maturity banner, and artifact SHA-256 fingerprint.
- **AC-039 (Card Renderers):** Provide renderers to serialize cards to formatted JSON, canonical YAML, and styled GitHub-flavored Markdown.
- **AC-040 (Croissant 1.1 JSON-LD):** Emit MLCommons Croissant 1.1 JSON-LD for simulation datasets using standard-library JSON without requiring heavy dependencies.

---

## 3. Architecture & Public Contract

### 3.1 Namespace Structure
All durability and card facilities are partitioned into dedicated core subpackages:
- `ewm_engine.durability`:
  - Protocols: `EventStore`, `ResultStore`
  - Core models: `TransitionEvent`, `TraceLog`
  - Replay: `fold_events`, `replay_trajectory`, `verify_trajectory_replay`
  - Memoization: `compute_simulation_fingerprint`, `MemoizedSimulationRunner`, `run_memoized`
  - Backends: `InMemoryEventStore`, `InMemoryResultStore`, `JsonFileEventStore`, `SqliteEventStore`, `FilesystemResultStore`, `RedisResultStore`, `S3ResultStore`
- `ewm_engine.cards`:
  - Models: `ModelCard`, `ScenarioCard`, `DatasetCard`
  - Renderers: `render_card_markdown`, `render_card_yaml`, `render_card_json`
  - Croissant: `to_croissant_dataset`, `to_croissant_json`

### 3.2 Schema Snapshot Compatibility
All models declare `schema_version = "1.0.0"` and are registered in `scripts/generate_schemas.py`. Draft 2020-12 JSON schemas are committed in `schemas/`:
- `schemas/transition-event.schema.json`
- `schemas/model-card.schema.json`
- `schemas/scenario-card.schema.json`
- `schemas/dataset-card.schema.json`

---

## 4. Verification & Testing

1. **Unit Tests:**
   - `tests/unit/test_durability_events.py`: Verifies event creation, monotonic step checks, and DAG forks.
   - `tests/unit/test_event_stores.py`: Exercises `InMemoryEventStore`, `JsonFileEventStore`, and `SqliteEventStore`.
   - `tests/unit/test_result_store.py`: Tests `FilesystemResultStore` with NumPy sidecars and memoization.
   - `tests/unit/test_native_cards.py`: Tests card rendering and Croissant 1.1 JSON-LD structure.
2. **Property Tests:**
   - `tests/property/test_replay_determinism.py`: Validates across parameter sweeps that `replay_trajectory(events) == original_trajectory` and branch isolation holds.
3. **Contract & Architectural Boundaries:**
   - `tests/contract/test_api_compatibility.py`: Verifies schema locks and root API surface immutability.
   - `tests/contract/test_core_dependencies.py`: Asserts zero optional dependencies in `durability` and `cards`.
