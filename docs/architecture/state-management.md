# State Management and Mathematical Formalism

## Formal State Representation

The world state at discrete simulation step $t$ is formalized as a 5-tuple:

$$S_t = (G_t, R_t, M_t, \Gamma_t, C_t)$$

where:
- **$G_t = (V_t, E_t)$**: Directed entity-relationship graph. $V_t$ contains domain entities (warehouses, hospitals, vehicles), and $E_t$ represents typed relationships (supplies, depends_on, connected_to).
- **$R_t \in \mathbb{R}^k$**: Map of finite, measurable quantities (inventory stock, fuel reserves, workforce hours, bed capacity) with defined lower bounds ($R_{\min}$) and upper bounds ($R_{\max}$).
- **$M_t$**: Temporal state memory, cumulative statistics, and historical lag variables (e.g. running unmet demand, rolling averages).
- **$\Gamma_t$**: Active operational rules, policy thresholds, and regulatory constraints.
- **$C_t$**: Exogenous environmental context (meteorological weather, macro-economic regime, pandemic phase).

---

## State Invariants and Properties

### 1. Deep Semantic Immutability (`frozen=True` + Defensive Copying)
`WorldState` instances cannot be mutated in place. State changes strictly produce a fresh `WorldState` instance via functional transitions:
```python
next_state = current_state.update_resource("stock", delta=-10.0)
```

To eliminate aliasing bugs and state contamination across simulation branches (AC-005):
- **Construction Defensive Copying**: Initializing a `WorldState` creates deep copies (`copy.deepcopy`) of all input sequences and mappings (`memory`, `context`, `active_rules`, `entities`, `resources`, `relationships`). External callers retaining references to input dictionaries cannot mutate internal state post-construction.
- **Accessor Defensive Copying**: Accessing dictionary fields on a state instance (`state.memory`, `state.context`, `state.active_rules`, `state.entities`, `state.resources`) returns an isolated deep copy. Any in-place mutation of the returned collection (e.g., `state.memory["flag"] = 1`) affects solely the retrieved copy and leaves the stored world state and its fingerprint strictly unchanged.
- **Domain Primitives**: `Entity`, `Relationship`, `Action`, `ExogenousEvent`, and `Intervention` similarly isolate their internal `attributes` and `parameters` dictionaries.

> **Performance Note**: Accessing `state.memory` incurs a deep copy allocation. In performance-critical simulation inner loops, store the retrieved mapping in a local variable rather than accessing `state.memory` repeatedly.

### 2. Canonical Cryptographic Fingerprinting (`fingerprint` / `state_hash`)
Every state exposes a deterministic, order-independent SHA-256 fingerprint:
- **Order Independence**: Dictionaries and nested mappings are recursively sorted by key. Entities, resources, and relationships are sorted deterministically by their identifiers. Different insertion orders of identical domain data produce identical fingerprints.
- **Deterministic Float Precision**: All finite floating-point numbers are rounded to 6 decimal places ($10^{-6}$ precision), normalizing `-0.0` to `0.0`.
- **Rejection of Non-Finite Floats**: Encountering `NaN`, `+Infinity`, or `-Infinity` anywhere in the state (memory, context, entity attributes, resource levels) immediately raises an `InvalidWorldStateError`. Numerical explosions or division-by-zero errors in dynamics models cannot produce corrupted fingerprints.
- **UTC Timestamp Normalization**: Continuous simulation timestamps and ISO-8601 datetimes are normalized to UTC representations.
- **Unbounded Resource Representation**: Infinite upper bounds on resources (`max_value = float("inf")`) are canonicalized to `None` to comply strictly with RFC 8259 JSON standards.

### 3. Safe Serialization Round-Trip (AC-002)
Every state supports lossless serialization to standard JSON dictionaries and deserialization back into validated domain instances:

$$\text{WorldState} \equiv \text{WorldState.from\_dict}(\text{state.to\_dict}())$$

Reconstituting a state from JSON identically preserves its canonical fingerprint bit-for-bit:
```python
state_dict = state.to_dict()
reconstituted = WorldState.from_dict(state_dict)
assert reconstituted.fingerprint == state.fingerprint
```
