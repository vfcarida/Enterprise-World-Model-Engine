# API Reference Guide

This reference outlines the primary classes, protocols, and data models of the **Enterprise World Model Engine**.

---

## Core Primitives (`ewm_engine.core`)

### `WorldState`
Canonical immutable snapshot representing organizational state at step $t$:
$$S_t = (G_t, R_t, M_t, \Gamma_t, C_t)$$

- `entities`: Mapping of active entities keyed by `EntityId`.
- `relationships`: Sequence of directed relational edges.
- `resources`: Mapping of finite measurable quantities keyed by `ResourceId`.
- `memory`: Temporal lags, cumulative statistics, and state history.
- `active_rules`: Active policy configurations and thresholds.
- `context`: Environmental and exogenous variables.
- `state_hash`: Cryptographic SHA-256 fingerprint.

### `Entity`
Participant in the world (warehouse, hospital, region, vehicle).
- `id`: Unique identifier.
- `type`: Category string.
- `attributes`: Property map.
- `tags`: Categorical grouping tags.

### `Resource`
Bounded physical or measurable asset.
- `id`: Unique identifier.
- `current`: Current quantity.
- `min_value` / `max_value`: Lower and upper bounds.
- `utilization`: Fractional utilization in $[0.0, 1.0]$.
- `with_delta(delta, clamp, enforce_bounds)`: Returns updated resource.

### `Action` & `Intervention`
- `Action`: Operational micro-step issued by an actor.
- `Intervention`: Strategic macro-policy modification applied to initial state or rules.

---

## Pluggable Dynamics (`ewm_engine.dynamics`)

### `DynamicsModel` (Protocol)
```python
def transition(
    self,
    state: WorldState,
    actions: Sequence[Action],
    exogenous_events: Sequence[ExogenousEvent],
    rng: RandomGenerator,
) -> TransitionResult:
    ...
```

### `CompositeDynamics`
Sequentially pipes state through multiple modular dynamics models, aggregating applied changes and resolving epistemic evidence tiers.

---

## Constraints Engine (`ewm_engine.constraints`)

### `Constraint` (Protocol)
- `constraint_id`: Unique identifier.
- `severity`: `ConstraintSeverity.HARD` or `ConstraintSeverity.SOFT`.
- `evaluate(state, action)`: Returns detailed `ConstraintResult`.

### `ConstraintRegistry`
- `validate_actions(state, actions)`: Pre-transition validation of proposed actions.
- `validate_state(state, preceding_actions)`: Post-transition state invariant audit.

---

## Simulation & Evaluation (`ewm_engine.simulation` & `ewm_engine.evaluation`)

### `Scenario`
Specification for simulation experiments: horizon, samples, master random seed, and optional intervention.

### `SimulationResult`
Stores rollout trajectories, cryptographic metadata fingerprint, and distribution aggregators.

### `compare_scenarios(baseline, candidates, metrics)`
Produces structured machine-readable comparison dictionaries and formatted summary tables.
