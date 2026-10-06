# ADR-026: Event-Sourced TraceLog, ResultStore, and Native Cards

## Status
Accepted

## Date
2026-10-05

## Context
In version 1.0.0, simulation rollouts were primarily ephemeral in-memory objects. Trajectories could be inspected after execution, but could not be reliably persisted, replayed across engine versions, resumed, or cached without custom ad-hoc scripts.

Furthermore:
1. `WorldState` is already immutable and fingerprinted with canonical SHA-256 hashes. A simulation step is fundamentally a state transition tuple `(parent_fingerprint → event → child_fingerprint, seed, metadata)`. This matches the formal definition of an event store (state = fold over events), as validated in Meta ARE (arXiv:2509.17158).
2. Monte Carlo rollouts are computationally intensive and deterministic. Since the engine computes canonical fingerprints over the initial state, dynamics, seed entropy, and scenario configurations, this fingerprint serves as the ideal cache key. Existing orchestrators (Prefect, Dagster) approximate cache keys with input hashes; EWM Engine computes it natively.
3. Decision makers and regulatory auditors require auditable documentation. The Google Model Card Toolkit was archived in 2024. A native, dependency-free Pydantic implementation grounded in Model Cards (arXiv:1810.03993), Datasheets for Datasets (arXiv:1803.09010), and population calibration provenance (arXiv:2411.10109) provides portable, verifiable documentation.

## Decision

### 1. Versioned Event Schema & Append-Only TraceLog
We define `TransitionEvent` locked at `schema_version = "1.0.0"` in `src/ewm_engine/durability/events.py`.
- Encapsulates: `stream_id`, `step`, `parent_fingerprint`, `child_fingerprint`, `timestamp`, `seed`, `actions_proposed`, `actions_accepted`, `exogenous_events`, `transition_result`, `applied_changes`, `constraint_violations`, `state_snapshot`, and `component_versions`.
- `TraceLog`: Represents an append-only event stream preserving DAG structure. Branching is implemented as forks in the event DAG (`branch()`), guaranteeing branch isolation.

### 2. EventStore Protocol & Built-in Backends
We define the `EventStore` protocol (`append`, `read_stream`, `get_event`, `list_streams`, `fold`) in `src/ewm_engine/durability/protocol.py`.
We provide three standard-library backends (zero external dependencies):
1. `InMemoryEventStore`: Fast, thread-safe in-memory store.
2. `JsonFileEventStore`: File-based store using structured JSON or YAML files.
3. `SqliteEventStore`: Embedded relational store using Python's standard `sqlite3` module with WAL-mode concurrency and foreign-key indexing.

External stores (e.g. `eventsourcing`, KurrentDB) will land as isolated adapters in later milestones without modifying core.

### 3. ResultStore Protocol & Memoization
We define the `ResultStore` protocol (`get`, `put`, `contains`, `delete`, `list_fingerprints`).
- `FilesystemResultStore`: Persists `result.json` (metadata, scenario, metrics, trajectories) alongside `arrays.npz` (NumPy compressed sidecars for numerical metrics and resource timeseries). Uses atomic file writes (`.tmp` write + rename).
- `compute_simulation_fingerprint`: Computes canonical SHA-256 fingerprint over initial state, dynamics, seed entropy, scenario configuration, interventions, and scheduled actions.
- `MemoizedSimulationRunner` / `run_memoized`: Caches simulation runs. Cache hits are verified to be byte-identical and logically identical to fresh runs.
- `RedisResultStore` and `S3ResultStore`: Documented stubs directing users to install `ewm-engine[cache]`.

### 4. Native Cards & Croissant 1.1 JSON-LD
We implement native Pydantic card models in `src/ewm_engine/cards/`:
- `ModelCard`: Documents dynamics models and policies (intended use, assumptions, out-of-scope conditions, constraints exercised, metrics, limitations, calibration score).
- `ScenarioCard`: Documents experiment setups and stress tests (horizon, samples, interventions, assumptions, context, metrics).
- `DatasetCard`: Documents trajectory datasets (collection process, features, intended use, license, upstream provenance fingerprints).
- Serialization: `.to_json()`, `.to_yaml()`, and `.to_markdown()` rendering GitHub-flavored Markdown with alert blocks and metric tables.
- Croissant 1.1: `to_croissant_dataset()` hand-emits standard MLCommons Croissant 1.1 JSON-LD using Python's standard `json` module.

### 5. Architectural Isolation & SemVer 1.x Compatibility
To preserve the committed `LOCKED_V1_STABLE_SURFACE` in `ewm_engine.__all__`, new symbols are provided under top-level subpackages:
- `ewm_engine.durability`
- `ewm_engine.cards`

## Consequences
- **Positive:** Trajectories can now be persisted, replayed, branched, and resumed with bitwise determinism.
- **Positive:** Rollouts can be memoized using exact cryptographic fingerprints, eliminating redundant compute.
- **Positive:** Native cards provide auditable, publication-ready documentation without depending on deprecated third-party toolkits.
- **Positive:** Zero new external core dependencies introduced (stdlib `sqlite3`, `numpy`, `pydantic`, `pyyaml` only).
- **Positive:** Establishes stable protocols (`EventStore`, `ResultStore`) allowing future heavy backends (Redis, S3, DuckDB, Parquet) to land cleanly as optional adapters.
