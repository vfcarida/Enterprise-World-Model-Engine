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

The following self-contained script demonstrates defining a two-warehouse world, attaching conservation dynamics, executing a baseline rollout, branching a scenario intervention, and comparing outcomes:

```python
from ewm_engine import (
    Action,
    Entity,
    Relationship,
    Resource,
    Scenario,
    SimulationEngine,
    World,
    WorldState,
    compare_scenarios,
)
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.scenario import ScheduledAction

# 1. Initialize World State S_0
state = WorldState(
    entities=[
        Entity(id="wh_north", type="warehouse", attributes={"region": "north"}),
        Entity(id="wh_south", type="warehouse", attributes={"region": "south"}),
    ],
    relationships=[
        Relationship(source="wh_north", target="wh_south", type="connected_to"),
    ],
    resources=[
        Resource(id="stock_north", current=100.0, min_value=0.0, max_value=200.0),
        Resource(id="stock_south", current=20.0, min_value=0.0, max_value=200.0),
    ],
)

# 2. Attach Constraints and Dynamics
world = World(
    state=state,
    dynamics=DeterministicTransferDynamics(),
    constraints=[
        ResourceCapacityConstraint(resource_id="stock_north"),
        ResourceCapacityConstraint(resource_id="stock_south"),
    ],
)

# 3. Simulate Baseline Scenario (Status Quo)
engine = SimulationEngine()
scenario_baseline = Scenario(
    scenario_id="baseline",
    name="Status Quo",
    horizon=2,
    samples=1,
    seed=42,
)
res_baseline = engine.run(world, scenario_baseline)

# 4. Branch and Simulate Scenario Intervention
world_alt = world.branch()
scenario_intervention = Scenario(
    scenario_id="intervention",
    name="Transfer 30 North -> South",
    horizon=2,
    samples=1,
    seed=42,
    scheduled_actions=(
        ScheduledAction(
            step=1,
            action=Action(
                id="act_transfer",
                type="transfer_resource",
                parameters={
                    "source_resource": "stock_north",
                    "target_resource": "stock_south",
                    "quantity": 30.0,
                },
            ),
        ),
    ),
)
res_intervention = engine.run(world_alt, scenario_intervention)

# 5. Evaluate and Compare Scenarios
comparison = compare_scenarios(
    baseline=res_baseline,
    candidates=[res_intervention],
    metrics=["resource_stock_north", "resource_stock_south", "violations_count"],
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
