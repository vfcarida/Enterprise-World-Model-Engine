# ADR-011: Safe Serialization Contract, JSON Schemas & Trusted Component Registry

## Status
Accepted

## Context
Enterprise World Model Engine (EWM Engine) requires a safe, deterministic, and versioned serialization contract for declarative models, specifications, scenarios, trajectories, and provenance.
Previous prototypes or standard Python simulations often rely on unsafe deserialization mechanisms (such as Python `pickle`, `eval`, or arbitrary YAML tags like `!!python/object` or `implementation: "module.Class"`), creating critical Remote Code Execution (RCE) vulnerabilities when loading external or user-provided world models. Furthermore, floating-point ambiguity (NaN, Infinity) can compromise deterministic replay, branch isolation, and cross-platform verification.

## Decision
1. **Mandatory Safe Pipeline**:
   The engine implements the strict pipeline:
   `YAML/JSON text -> StrictSafeLoader / canonical_loads -> plain JSON data -> Pydantic validation -> WorldSpec -> trusted ComponentRegistry -> runtime objects`.
2. **Strict Security Boundaries**:
   - `SerializationSecurityError` is raised whenever unsafe YAML tags (`!!python/*`, `!custom`), pickle byte sequences, non-finite floats (`NaN`, `Infinity`, `-Infinity`), or arbitrary code execution tags are encountered.
   - Pickle deserialization (`load_pickle`) is explicitly prohibited and fails closed.
   - Dynamic import from data (e.g. `implementation: "some.module.Class"`) is strictly forbidden; `ComponentSpec` enforces `extra="forbid"`.
3. **Trusted Component Registry & WorldFactory**:
   - Component instantiation (dynamics, constraints, event sources) occurs exclusively via programmatic registration in `ComponentRegistry`.
   - Default built-in components (`deterministic_transfer`, `capacity`, `non_negative`, `transfer_availability`) are pre-registered with validated factories.
   - Any unknown component type ID fails closed with a `SimulationConfigurationError`.
4. **Draft 2020-12 Versioned JSON Schemas**:
   - Official JSON Schemas are generated directly from Pydantic models for `Action`, `WorldState`, `WorldSpec`, `Scenario`, `Trajectory`, and `Provenance`.
   - Each schema specifies a stable `$id` (`https://ewm-engine.org/schemas/v1/<name>.schema.json`) and `const: "1.0.0"` for `schema_version`.
   - A verification script `scripts/generate_schemas.py --check` and contract test `tests/contract/test_schema_snapshots.py` enforce zero silent drift in CI.

## Consequences
- **Positive**:
  - Eliminates arbitrary code execution vectors from user-supplied simulation files.
  - Guarantees interoperability via standard Draft 2020-12 JSON Schemas.
  - Enables loss-free, deterministic round-trips for state, scenario, trajectory, and provenance.
  - Rejects malformed and poisoned inputs (NaN, Infinity, unsafe tags) at the serialization boundary.
- **Negative**:
  - Custom dynamics models or user constraints cannot be injected via inline class paths in YAML; they must be registered programmatically in Python before running simulations.

## Acceptance Criteria Satisfied
- **AC-002**: Public serializable models round-trip losslessly via canonical JSON.
- **AC-003**: JSON Schemas match current models with zero drift.
- **AC-017**: Distribution ships PEP 561 `py.typed` typing marker.
- **AC-018**: Unsafe YAML tags and pickle rejected with `SerializationSecurityError`.
- **AC-019**: Core dependencies contain no forced ML/solver/LLM packages.
