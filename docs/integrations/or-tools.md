# Operations Research with Google OR-Tools

The `ORToolsAllocationAdapter` integrates [Google OR-Tools](https://developers.google.com/optimization) with EWM Engine to automatically synthesize optimal, constraint-compliant action sequences.

---

## Why Mathematical Programming in World Models?

While reinforcement learning learns policies through trial and error over thousands of episodes, many enterprise resource allocation problems (e.g. transportation, minimum-cost network flows, capacity planning) can be solved **optimally in milliseconds** using Mixed-Integer Linear Programming (MILP).

`ORToolsAllocationAdapter` formulates network flow problems directly over an immutable [`WorldState`](../api/reference.md), solving for flow quantities that minimize cost or unmet demand, and emits concrete, validated [`Action`](../api/reference.md) sequences ready for simulation.

---

## Example: Optimal Warehouse Stock Rebalancing

```python
from ewm_engine import Resource, WorldState
from ewm_engine.integrations.ortools import ORToolsAllocationAdapter

# 1. State with supplying and demanding nodes
state = WorldState(
    resources=[
        Resource(id="wh_a_stock", current=100.0, min_value=0.0, max_value=200.0),
        Resource(id="wh_b_stock", current=150.0, min_value=0.0, max_value=200.0),
        Resource(id="store_c_stock", current=10.0, min_value=0.0, max_value=200.0),
        Resource(id="store_d_stock", current=5.0, min_value=0.0, max_value=200.0),
    ]
)

# 2. Configure OR-Tools Optimizer
adapter = ORToolsAllocationAdapter(solver_type="GLOP")

# 3. Solve Optimal Transfer Quantities
actions = adapter.optimize_transfers(
    state=state,
    sources=["wh_a_stock", "wh_b_stock"],
    destinations=["store_c_stock", "store_d_stock"],
    demands={"store_c_stock": 40.0, "store_d_stock": 50.0},
    costs={
        ("wh_a_stock", "store_c_stock"): 2.0,
        ("wh_a_stock", "store_d_stock"): 5.0,
        ("wh_b_stock", "store_c_stock"): 4.0,
        ("wh_b_stock", "store_d_stock"): 1.5,
    },
)

# 4. Resulting Actions
for act in actions:
    params = act.parameters
    print(f"Action: Transfer {params['quantity']} from {params['source_resource']} to {params['target_resource']}")
```
