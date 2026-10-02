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

1. **Immutability (`frozen=True`)**:
   `WorldState` instances cannot be mutated in place. State changes produce a fresh `WorldState` instance via structural sharing or validated copy:
   ```python
   next_state = current_state.update_resource("stock", delta=-10.0)
   ```
2. **Cryptographic Fingerprinting (`state_hash`)**:
   Every state maintains a deterministic SHA-256 hash computed over normalized canonical representations of its entities, relationships, resources, memory, and context.
3. **Safe Serialization Round-Trip**:
   Every state supports lossless serialization to standard JSON dictionaries and deserialization back into validated domain instances:
   $$\text{WorldState} \equiv \text{WorldState.from\_dict}(\text{state.to\_dict}())$$
