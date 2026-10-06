# ADR-022: Heterogeneous Graph World State and GNN Dynamics

**Status:** Accepted  
**Date:** 2026-10-05  
**Context:** EWM Engine v2.0 Candidate (Prompt P10 / Horizon C)  
**Related Specs:** `FEAT-003`, `ADR-002`, `ADR-007`, `ADR-009`, `ADR-019`

---

## Context

Enterprise world models capture organizational systems characterized by complex network topologies: multi-echelon supply chains, multimodal transportation corridors, financial credit webs, and administrative reporting structures.

In `v1.x`, `WorldState` formalized state as $S_t = (G_t, R_t, M_t, \Gamma_t, C_t)$ with `entities`, `relationships`, and `resources` stored as flat mappings and sequences. While this model provided canonical hashing, immutability, and state-machine transitions, extracting adjacency structures, computing relational graph neural network (GNN) dynamics, and handling temporal relation lifetimes required ad-hoc manual transformations.

As the engine prepares for `v2.0.0`, we must define how heterogeneous temporal relational graphs relate to `WorldState`, and how Graph Neural Network dynamics can operate without pulling heavyweight graph libraries or PyTorch into core.

---

## Decision

### 1. Additive Graph View Pattern (Zero Breaking Rewrite)
Rather than breaking `WorldState` or forcing an incompatible storage layer, we introduce an **Additive Graph View Pattern**:
- `WorldState` remains the canonical, immutable, auditable state representation.
- `HeterogeneousGraphView` (`ewm_engine.core.graph`) provides an additive, strongly-typed multi-relational graph projection over `WorldState`.
- `state.as_graph()` derives the graph view on demand without mutating state or duplicating data structures.
- A `HeterogeneousGraphView` can reconstruct or update a `WorldState` via `graph_view.to_world_state(base_state)`.

### 2. Heterogeneous Temporal Relational Graph Semantics
The graph view formalizes:
- **Heterogeneous Node Types**: Partitioned by `Entity.type` (e.g. `warehouse`, `retail_store`, `vehicle`).
- **Heterogeneous Edge Types**: Partitioned by `(src_type, relation_type, dst_type)` triples.
- **Node Feature Matrices**: Extracted from `Entity.attributes` and associated `Resource` currents.
- **Edge Feature Matrices**: Extracted from `Relationship.attributes` (e.g. distance, capacity, cost, latency).
- **Temporal Relations**: Incorporates simulation step, continuous timestamp, and temporal validity intervals $[t_{\text{valid\_from}}, t_{\text{valid\_until}})$.

### 3. Optional GNN Dynamics Behind `ml` Extra
Under `ewm_engine.experimental.graph_dynamics`:
- Implements `GraphNeuralDynamics` conforming to `DynamicsModel`.
- Executes Relational Graph Convolutional message passing over `HeterogeneousGraphView`.
- Strictly maintains zero PyTorch dependencies in `core`: `torch` is lazy-loaded at runtime via `_load_torch()`.
- Validates against the `P06` dynamics evaluation harness and `P07` scientific benchmarks.

### 4. Schema Evolution & Bidirectional Migration Tooling
- `WorldState.schema_version` supports both `"1.0.0"` and `"2.0.0"`.
- `ewm_engine.core.migration` provides `migrate_v1_to_v2()` and `migrate_v2_to_v1()` with full round-trip fidelity guarantees.

---

## Consequences

### Positive
- **100% Backward Compatibility**: Existing v1.x worlds, tests, and scenarios continue to function identically.
- **First-Class GNN Dynamics**: Enables relational graph representation learning and multi-echelon message passing.
- **Lightweight Core Preserved**: Zero forced external dependencies (no PyTorch, PyG, or NetworkX in core).

### Trade-offs & Mitigations
- Generating graph views incurs transient object creation overhead; mitigated by efficient vectorized feature extraction.
- Graph neural models require the `ml` extra; mitigated by clean error messages when optional dependencies are absent.
