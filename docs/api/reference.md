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
) -> TransitionResult: ...
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
Stores rollout trajectories, cryptographic metadata fingerprint, distribution aggregators, and `get_metric_distribution(metric_name)`.

### `compare_scenarios(baseline, candidates, metrics)`
Produces structured machine-readable comparison dictionaries and formatted summary tables.

### `RecedingHorizonSimulator` (`ewm_engine.simulation.mpc`)
Online Model Predictive Control (MPC) simulator performing closed-loop rolling horizon re-grounding:
- Evaluates candidate interventions over a lookahead horizon.
- Re-grounds to real-world observations at each decision step.
- Emits structured `MPCDecisionRecord` audit trails.

---

## Declarative Specifications (`ewm_engine.core.spec`)

### `WorldSpecification`
Safe declarative schema parser for YAML, JSON, and Python dictionaries:
- `from_yaml(path_or_str)`: Load specification from YAML string or file.
- `from_json(path_or_str)`: Load specification from JSON string or file.
- `build_state()`: Construct an immutable `WorldState`.
- `build_world()`: Construct an operational `World` with state, dynamics, and constraints.

---

## Integrations & Adapters (`ewm_engine.integrations`)

### `CallableActorAdapter`
Wraps any Python callable, LangGraph workflow, AutoGen agent, or LLM agent into a standard EWM `Actor`:
```python
actor = CallableActorAdapter(actor_id="agent_planner", fn=my_llm_decision_fn)
```

### `Z3ConstraintAdapter`
Bridges formal SMT verification via Microsoft Z3 into the EWM constraints pipeline with zero runtime crash if Z3 is not installed:
```python
adapter = Z3ConstraintAdapter(
    constraint_id="smt_safety_rule",
    solver_fn=my_smt_checker,
    severity=ConstraintSeverity.HARD,
)
```

---

## Provenance & Traceability (`ewm_engine.provenance`)

### `SystemicTrace`
Causal lineage graph recording causal dependencies between interventions, actions, transitions, exogenous events, and constraint violations:
- `add_dependency(source_id, target_id, relation)`: Records directed relationship edge.
- `to_mermaid()`: Exports the execution trajectory to standard Mermaid flowchart markdown.
