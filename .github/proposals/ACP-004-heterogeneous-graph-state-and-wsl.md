# API Change Proposal (ACP-004): Heterogeneous Graph World State & Safe Declarative WSL

- **Target Release:** v2.0.0-alpha (Experimental candidate in v1.x series)
- **Author:** Principal Architect
- **Status:** Implemented (Experimental)
- **Impact Level:** Additive graph projection + new declarative serialization grammar (`wsl/2.0.0`)
- **Related Spec:** `docs/specs/features/FEAT-003-heterogeneous-graph-state-and-wsl.md`, `docs/specs/spec-driven-development.md`
- **Related ADRs:** `docs/adr/ADR-022-heterogeneous-graph-world-state-and-gnn-dynamics.md`, `docs/adr/ADR-023-declarative-world-specification-language-wsl.md`
- **Acceptance Criteria:** AC-032 (Safe Declarative WSL), AC-002 (Lossless Round-Trip), AC-024 (Contract Compatibility)

---

## 1. Summary

This proposal establishes the architecture for representing complex enterprise networks as **heterogeneous temporal relational graphs** and introduces the **World Specification Language (WSL)** as a declarative, human-readable grammar (YAML/JSON) for authoring and sharing simulation worlds without code execution vulnerabilities.

Key architectural deliverables:
1. **Additive Graph Projection (`HeterogeneousGraphView`)**: Exposes multi-relational topologies, node/edge type partitions, 2D NumPy feature matrices, and temporal validity filtering without altering the underlying immutable `WorldState` model.
2. **Schema Migration Contract (`ewm_engine.core.migration`)**: Formal bidirectional migration between `schema_version = "1.0.0"` and `"2.0.0"`.
3. **Safe Declarative World Specification Language (`wsl/2.0.0`)**: High-level YAML/JSON grammar validated against Pydantic models and a committed JSON Schema (`docs/schemas/wsl-v2.schema.json`), compiled into runnable `World` instances exclusively through a closed programmatic `ComponentRegistry`.

---

## 2. Motivation & Use Case

Real-world organizations operate as interconnected multi-echelon networks (suppliers, warehouses, regional clinics, transportation fleets). Representing these purely as flat lists of entities limits relational message passing (GNN dynamics) and topological query analysis.

Concurrently, sharing world models across enterprise teams, multi-tenant platforms, and autonomous agent systems requires a declarative serialization format. Imperative Python scripts carry critical arbitrary-code-execution risks (`eval`, `exec`, unsafe pickle deserialization). WSL solves this by enforcing a strictly declarative, validated grammar.

---

## 3. Proposed API Surface Diff

```python
# 1. Heterogeneous Graph View (Additive projection in ewm_engine.core.graph)
class HeterogeneousGraphView(BaseModel):
    @classmethod
    def from_world_state(cls, state: WorldState) -> HeterogeneousGraphView: ...
    def to_world_state(self, base_state: WorldState | None = None) -> WorldState: ...
    def get_node_features(self, node_type: str, ...) -> np.ndarray: ...
    def get_edge_index(self, edge_type: str) -> tuple[np.ndarray, np.ndarray]: ...
    def filter_temporal(self, current_time: float) -> HeterogeneousGraphView: ...

# 2. Schema Migration (ewm_engine.core.migration)
def migrate_v1_to_v2(state: WorldState) -> WorldState: ...
def migrate_v2_to_v1(state: WorldState) -> WorldState: ...

# 3. World Specification Language (ewm_engine.serialization)
def parse_wsl_file(path: str | Path) -> WSLDocument: ...
def parse_wsl_yaml(yaml_content: str) -> WSLDocument: ...
def compile_wsl(doc: WSLDocument, registry: ComponentRegistry | None = None) -> World: ...
def export_wsl(world: World, metadata: WSLMetadata | None = None) -> WSLDocument: ...
def dump_wsl_yaml(doc: WSLDocument) -> str: ...
```

### Affected Symbols in `ewm_engine.__all__`
- [x] **No breaking change to root `ewm_engine.__all__`**: Public surface remains strictly the 22 canonical Stable symbols + `__version__`.
- Graph views and migration are available via `ewm_engine.core.graph`, `ewm_engine.core.migration`, and `ewm_engine.experimental`.
- WSL tools are exported safely from `ewm_engine.serialization`.

---

## 4. Security & Safety Invariants

1. **Zero Dynamic Code Execution**: WSL parsers reject all embedded Python code, mathematical string evaluation (`eval`), and custom YAML tags (`!python/object`, `!cmd`).
2. **Safe Deserialization**: Parsed exclusively through `StrictSafeLoader` to ensure that untrusted world files cannot trigger remote code execution.
3. **Registry-Gated Component Instantiation**: Custom dynamics models and constraints can only be referenced by registered `type_id` strings managed by a programmatic `ComponentRegistry`. Arbitrary module import-by-string is strictly forbidden.

---

## 5. Verification & Acceptance Gate

- `tests/unit/test_wsl.py` verifies parsing, schema validation, safe compilation, and YAML round-tripping.
- `tests/unit/test_migration.py` verifies lossless bidirectional migration between v1.0 and v2.0 schemas.
- `tests/unit/test_graph_view.py` verifies node/edge feature extraction and temporal filtering.
- `tests/contract/test_api_compatibility.py` passes with zero breaking changes to the Stable contract.
