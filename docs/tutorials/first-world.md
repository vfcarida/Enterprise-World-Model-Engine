# Tutorial: Building Your First Enterprise World Model

In this tutorial, you will learn how to build an executable, action-conditioned enterprise world model from scratch.

By the end of this lesson, you will understand how to:
1. Initialize an immutable snapshot state ($S_0$) with typed entities and bounded resources.
2. Attach deterministic dynamics and capacity constraints.
3. Run a Monte Carlo simulation rollout over a multi-step horizon.

---

## 1. Defining the World State ($S_0$)

Every simulation begins with an immutable `WorldState` snapshot. Let's model a regional distribution hub with two supply depots: `depot_east` and `depot_west`.

```python
from ewm_engine import Entity, Relationship, Resource, WorldState

state = WorldState(
    entities=[
        Entity(id="depot_east", type="facility", attributes={"tier": 1}),
        Entity(id="depot_west", type="facility", attributes={"tier": 2}),
    ],
    relationships=[
        Relationship(source="depot_east", target="depot_west", type="supplies"),
    ],
    resources=[
        Resource(id="inventory_east", current=150.0, min_value=0.0, max_value=300.0),
        Resource(id="inventory_west", current=50.0, min_value=0.0, max_value=200.0),
    ],
)

print(f"State initialized with fingerprint: {state.fingerprint}")
```

> [!NOTE]
> Every `WorldState` computes an automatic SHA-256 fingerprint from its canonical contents. States are deeply immutable — transitions always return fresh snapshots.

---

## 2. Attaching Dynamics and Constraints

Next, wrap the state inside a `World` container along with transition dynamics and operational constraints:

```python
from ewm_engine import World
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics

world = World(
    state=state,
    dynamics=DeterministicTransferDynamics(),
    constraints=[
        ResourceCapacityConstraint(resource_id="inventory_east"),
        ResourceCapacityConstraint(resource_id="inventory_west"),
    ],
)
```

---

## 3. Running a Baseline Simulation

Now configure a baseline `Scenario` specifying the time horizon, number of sample rollouts, and random seed:

```python
from ewm_engine import Scenario, SimulationEngine

engine = SimulationEngine()
scenario = Scenario(
    scenario_id="baseline_status_quo",
    name="Status Quo Rollout",
    horizon=3,
    samples=1,
    seed=1001,
)

result = engine.run(world, scenario)
trajectory = result.trajectories[0]

print(f"Trajectory completed with status: {trajectory.status}")
print(f"Final East Inventory: {trajectory.final_state.get_resource('inventory_east').current}")
print(f"Final West Inventory: {trajectory.final_state.get_resource('inventory_west').current}")
```

Congratulations! You have successfully built and simulated your first Enterprise World Model. In the next tutorial, you will learn how to branch this world to simulate alternative policy interventions.
