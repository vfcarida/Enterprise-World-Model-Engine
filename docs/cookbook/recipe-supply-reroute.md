# Recipe: Constrained Multi-Hub Inventory Reroute

> **Task**: Rebalance inventory between two distribution hubs prior to an exogenous demand spike, ensuring that storage capacities and mass balance invariants are never violated.

---

## 1. The Recipe Code

```python
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.core.actions import Action, ScheduledAction
from ewm_engine.core.entities import Entity
from ewm_engine.core.relationships import Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario


def execute_reroute_recipe(transfer_volume: float = 35.0) -> None:
    # 1. Build World with Strict Capacity & Transfer Constraints
    state = WorldState(
        entities=[
            Entity(id="hub_a", type="warehouse", attributes={"cap": 150.0}),
            Entity(id="hub_b", type="warehouse", attributes={"cap": 150.0}),
        ],
        relationships=[
            Relationship(source="hub_a", target="hub_b", type="transfer_corridor"),
        ],
        resources=[
            Resource(id="stock_a", current=100.0, min_value=0.0, max_value=150.0),
            Resource(id="stock_b", current=20.0, min_value=0.0, max_value=150.0),
        ],
    )

    world = World(
        state=state,
        dynamics=DeterministicTransferDynamics(),
        constraints=[
            ResourceCapacityConstraint(resource_id="stock_a"),
            ResourceCapacityConstraint(resource_id="stock_b"),
            ActionTransferAvailabilityConstraint(),
        ],
    )

    # 2. Schedule Transfer Action at step 1
    scenario = Scenario(
        scenario_id="recipe-reroute-v1",
        horizon=2,
        samples=1,
        seed=42,
        actions=[
            ScheduledAction(
                step=1,
                action=Action(
                    action_type="transfer",
                    source="stock_a",
                    target="stock_b",
                    amount=transfer_volume,
                ),
            )
        ],
    )

    # 3. Simulate and Inspect
    engine = SimulationEngine()
    result = engine.run(world, scenario)
    traj = result.trajectories[0]
    final_a = traj.final_state.get_resource("stock_a").current
    final_b = traj.final_state.get_resource("stock_b").current

    print(f"Transfer of {transfer_volume} units succeeded!")
    print(f"Final Stock A: {final_a} | Final Stock B: {final_b}")
    print(f"Total System Stock Conserved: {final_a + final_b == 120.0}")


if __name__ == "__main__":
    execute_reroute_recipe(35.0)
```

---

## 2. Key Safeguards
* **Negative Stock Prevention:** If `transfer_volume` exceeds `stock_a` (e.g. `120.0`), `ActionTransferAvailabilityConstraint` immediately raises a violation and marks the rollout `TrajectoryStatus.INVALID`.
* **Capacity Overflow Prevention:** If `transfer_volume` pushes `stock_b` past its `max_value` of `150.0`, `ResourceCapacityConstraint` halts the rollout.
