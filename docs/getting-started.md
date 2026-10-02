# Getting Started with EWM Engine

Get up and running with the **Enterprise World Model Engine** in 5 minutes.

---

## Installation

### Baseline Installation (Zero Heavy ML / No Cloud Dependencies)
```bash
pip install ewm-engine
```

### With Optional Extensions
```bash
# Graph inspection & NetworkX export
pip install "ewm-engine[graphs]"

# Learned dynamics & neural transitions
pip install "ewm-engine[ml]"

# Documentation tools
pip install "ewm-engine[docs]"

# Full development & test suite
pip install "ewm-engine[all]"
```

---

## 5-Minute Minimal Example

Here is a complete, working world model demonstrating state initialization, capacity constraints, stochastic demand, counterfactual policy branching, and comparative evaluation:

```python
from ewm_engine.core import World, WorldState, Entity, Relationship, Resource, Intervention
from ewm_engine.constraints import ConstraintRegistry, ResourceCapacityConstraint, ActionTransferAvailabilityConstraint
from ewm_engine.dynamics import CompositeDynamics, DeterministicTransferDynamics, StochasticDemandDynamics
from ewm_engine.actors import ThresholdReplenishmentActor
from ewm_engine.simulation import Scenario
from ewm_engine.evaluation import compare_scenarios

# 1. Represent the World State (S_0)
state = WorldState(
    entities=[
        Entity(id="depot_north", type="warehouse"),
        Entity(id="depot_south", type="warehouse"),
    ],
    relationships=[
        Relationship(source="depot_north", target="depot_south", type="connected_to"),
    ],
    resources=[
        Resource(id="stock_north", current=150.0, min_value=0.0, max_value=250.0),
        Resource(id="stock_south", current=35.0, min_value=0.0, max_value=200.0),
    ],
)

# 2. Configure Constraints and Dynamics
constraints = ConstraintRegistry([
    ResourceCapacityConstraint(resource_id="stock_north"),
    ResourceCapacityConstraint(resource_id="stock_south"),
    ActionTransferAvailabilityConstraint(),
])

dynamics = CompositeDynamics([
    DeterministicTransferDynamics(),
    StochasticDemandDynamics(resource_id="stock_south", mean_demand=18.0, std_demand=3.0),
])

base_world = World(state=state, dynamics=dynamics, constraints=constraints)

# 3. Simulate Baseline Policy (Status Quo)
res_baseline = base_world.simulate(
    scenario=Scenario(name="Status Quo", horizon=10, samples=20, seed=42)
)

# 4. Branch and Simulate Counterfactual Intervention
proactive_world = base_world.branch()
proactive_world.add_actor(
    ThresholdReplenishmentActor(
        actor_id="agent",
        source_resource="stock_north",
        target_resource="stock_south",
        reorder_point=40.0,
        order_quantity=40.0,
    )
)

res_proactive = proactive_world.simulate(
    scenario=Scenario(
        name="Proactive Policy",
        horizon=10,
        samples=20,
        seed=42,
        intervention=Intervention(
            id="proactive_policy",
            description="Reorder 40 units whenever stock <= 40",
        ),
    )
)

# 5. Compare Scenarios
comparison = compare_scenarios(
    baseline=res_baseline,
    candidates=[res_proactive],
    metrics=["resource_stock_south", "resource_stock_north", "violations_count"],
)

print(comparison.summary_table())
```

---

## Running from the CLI

Run built-in reference simulations directly from your terminal:

```bash
# Run minimal two-warehouse inventory balancing
ewm example minimal

# Run flagship CivicFlow flood response research simulation
ewm example civicflow
```
