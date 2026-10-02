# ADR-009: State Immutability, Defensive Copying, and Canonical Fingerprinting

## Status
Accepted

## Related Spec & Issues
- Spec: `docs/specs/spec-driven-development.md` (State Management, Determinism, and Reproducibility)
- Acceptance Criteria: AC-004 (reproducibility foundation), AC-005 (branch isolation), AC-010 (fingerprint stability)
- Gaps Closed: G6 (deep immutability + proof tests), partial G12 (canonical fingerprint)
- Milestone: M3

## Context
In previous implementations, `WorldState` was configured with Pydantic's `frozen=True`. While this prevented attribute reassignment on the model (e.g. `state.timestamp = 1.0` raised an error), it did not prevent in-place mutation of the mutable mapping and collection fields it held (`memory`, `active_rules`, `context`, `entities`, `resources`, `attributes`).

External callers retaining references to dictionaries passed to `WorldState.__init__` could mutate them after construction, silently contaminating the simulation state. Similarly, accessing `state.memory["key"] = new_val` allowed callers to mutate internal state in place, violating branch isolation (AC-005) and non-deterministic divergence across rollouts.

Furthermore, state fingerprinting (`state_hash` / `fingerprint`) previously relied on non-standardized float rounding and Python `json.dumps`, which permitted non-finite floats (`NaN`, `Infinity`) and was vulnerable to dictionary key ordering variations.

## Decision
1. **Defensive Copying on Construction**:
   - `WorldState.__init__` performs deep defensive copies (`copy.deepcopy`) of all collection and mapping inputs (`entities`, `relationships`, `resources`, `memory`, `active_rules`, `context`, and any auxiliary keyword arguments).
   - Domain models (`Entity`, `Relationship`, `Action`, `ExogenousEvent`, `Intervention`) perform deep defensive copies of their internal `attributes` and `parameters` dictionaries upon initialization.

2. **Defensive Copying on Accessors**:
   - `WorldState.__getattribute__` intercepts accesses to `memory`, `active_rules`, `context`, `entities`, and `resources`, returning deep copies.
   - Any in-place mutation of a returned dictionary (e.g. `state.memory["k"] = v` or `state.entities["e"] = ...`) affects only the returned copy and strictly leaves internal state and its canonical fingerprint unchanged.
   - Domain primitives (`Entity`, `Relationship`, `Action`, `ExogenousEvent`, `Intervention`) return deep copies of their `attributes` and `parameters` dictionaries upon property access.

3. **Centralized Canonical Serialization & Hashing (`_canonical.py`)**:
   - All state and provenance fingerprinting is centralized in `ewm_engine.core._canonical`.
   - **Non-finite float rejection**: Any `NaN`, `+Infinity`, or `-Infinity` float value in simulation state or parameters raises `InvalidWorldStateError`.
   - **Float precision determinism**: Finite floats are rounded to 6 decimal places (`round(val, 6)`), normalizing `-0.0` to `0.0`.
   - **Key ordering**: All dictionaries and mappings are recursively sorted by key. Sets and frozensets are canonicalized into sorted lists.
   - **Timestamp normalization**: All timestamps and `datetime` objects are normalized to UTC ISO-8601 strings.
   - **Unbounded resource representation**: Infinite resource bounds (`max_value = float("inf")`) are canonicalized to `None` in the resource dictionary to ensure standard RFC 8259 JSON compliance.
   - **SHA-256 Digest**: Digests are computed over UTF-8 encoded compact JSON (`separators=(",", ":")`).

4. **Public API Contract**:
   - `WorldState.fingerprint` is added as the authoritative canonical property alias for `WorldState.state_hash`.
   - `SimulationMetadata.fingerprint` and `WorldState.fingerprint` share the canonical serialization helper.

## Consequences
- **Positive**:
  - Full semantic, deep immutability across all state layers and domain models.
  - Complete branch isolation: branch simulations cannot contaminate each other or baseline snapshots.
  - Deterministic, order-independent, platform-independent cryptographic fingerprints.
  - Early detection of numerical divergence or invalid computations through strict NaN/Inf rejection.
- **Trade-offs / Performance**:
  - Accessing `state.memory` or `state.context` incurs a deep copy allocation. Callers that read state frequently should store the retrieved copy in a local variable rather than accessing `state.memory` repeatedly in tight loops.
