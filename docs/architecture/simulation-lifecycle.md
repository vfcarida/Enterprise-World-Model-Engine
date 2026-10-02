# Simulation Execution Lifecycle

The simulation engine coordinates state evolution through an explicit, reproducible step execution loop.

---

## Detailed Step Lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant Engine as SimulationEngine
    participant Shocks as ExogenousEventSources
    participant Actors as ActorPolicies
    participant Constraints as ConstraintRegistry
    participant Dynamics as CompositeDynamics
    participant Trace as SystemicTrace

    Engine->>Shocks: sample(state, step, rng)
    Shocks-->>Engine: ExogenousEvents (E_t)
    Engine->>Trace: record_event_nodes(E_t)

    Engine->>Actors: act(state, ActorContext)
    Actors-->>Engine: ProposedActions (A_prop)

    Engine->>Constraints: validate_actions(state, A_prop)
    Constraints-->>Engine: AcceptedActions (A_acc), ActionViolations
    Engine->>Trace: record_action_and_violation_nodes()

    Engine->>Dynamics: transition(state, A_acc, E_t, rng)
    Dynamics-->>Engine: TransitionResult (next_state, deltas)

    Engine->>Constraints: validate_state(next_state, A_acc)
    Constraints-->>Engine: StateViolations
    Engine->>Trace: record_state_violation_nodes()

    Engine->>Engine: advance_time(step + 1, timestamp + dt)
    Engine->>Engine: record_step_metrics()
```

---

## Reproducible Seed Sequence Architecture

To guarantee exact bit-for-bit determinism across Monte Carlo rollouts:
1. The user supplies a master seed $s$ (e.g. `seed=42`).
2. The engine constructs a NumPy `SeedSequence(s)`.
3. Independent child seeds are spawned for each sample:
   ```python
   child_seeds = SeedSequence(scenario.seed).spawn(scenario.samples)
   ```
4. Each rollout trajectory receives its own dedicated pseudo-random generator `default_rng(child_seeds[k])`.
5. Rollouts are completely independent and immune to order-of-execution variations.
