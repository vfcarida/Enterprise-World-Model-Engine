# Minimal Supply World Example

This example demonstrates the core workflow of the **Enterprise World Model Engine (EWM Engine)** in approximately 50 lines of clear Python code.

## Scenario Description
Two regional distribution warehouses (`wh_north` and `wh_south`) manage perishable stock:
- `wh_north`: Has ample stock (150 units) and serves as an upstream supply depot.
- `wh_south`: Has lower stock (35 units) and faces stochastic customer demand averaging 18 units/step.

We evaluate two counterfactual strategies:
1. **Status Quo (Baseline)**: No proactive replenishment between warehouses.
2. **Proactive Policy**: An autonomous actor transfers 40 units from North to South whenever South stock drops $\le 40$ units.

## Running the Example
```bash
python examples/minimal_world/run.py
```

## Key Architectural Concepts Demonstrated
1. **Immutable WorldState**: Declares entities, typed relations, bounded resources, and initial capacity limits.
2. **First-Class Constraints**: `ResourceCapacityConstraint` and `ActionTransferAvailabilityConstraint` validate physical rules explicitly.
3. **Pluggable Composite Dynamics**: Combines deterministic conservation-preserving transfer flows with stochastic customer demand.
4. **Counterfactual Branching**: Evaluates the baseline and proactive policy from the identical initial world state $S_0$.
5. **Systemic Trace**: Generates an auditable dependency graph mapping policy intervention $\to$ transfers $\to$ stock buffers $\to$ demand satisfaction.
