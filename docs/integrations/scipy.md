# Continuous Optimization with SciPy

> [!NOTE]
> **Maturity**: Beta. Implements formal `ActionPlanner` and `Actor` protocols with enforced finite time limits. Requires `ewm-engine[or]`.

The `SciPyAllocationPlanner` integrates [SciPy Linear Programming](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) (using the modern **HiGHS** simplex and interior-point solvers) to solve continuous optimal allocation and min-cost flow problems over an EWM [`WorldState`](../reference/api/core.md).

---

## Why SciPy HiGHS?

- **Continuous Optimal Flows**: Ideal for continuous media, power distribution, fluid dynamics, and monetary budget rebalancing.
- **Fast Execution**: HiGHS is written in C++ and benchmarks among the fastest open-source LP solvers available.
- **Strict Time Limits**: Enforces `options={"time_limit": ...}` to prevent runaway solves under pathological constraint sets.
- **ActionPlanner Protocol**: Candidate actions are proposed as `transfer_resource` actions and validated by the engine's constraint pipeline.

---

## Example: Continuous Regional Budget Allocation

```python
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.integrations.scipy_planner import SciPyAllocationPlanner

# 1. State with funding reserves and regional budget deficits
state = WorldState(
    resources=[
        Resource(id="treasury_reserve", current=250.0, min_value=0.0, max_value=1000.0),
        Resource(id="region_north", current=10.0, min_value=0.0, max_value=500.0),
        Resource(id="region_south", current=5.0, min_value=0.0, max_value=500.0),
    ]
)

# 2. Configure SciPy Allocation Planner with mandatory 5-second time limit
planner = SciPyAllocationPlanner(
    actor_id="scipy_budget_allocator",
    sources=["treasury_reserve"],
    destinations=["region_north", "region_south"],
    demands={"region_north": 75.5, "region_south": 60.0},
    costs={
        ("treasury_reserve", "region_north"): 1.1,
        ("treasury_reserve", "region_south"): 1.4,
    },
    capacities={
        ("treasury_reserve", "region_north"): 100.0,
        ("treasury_reserve", "region_south"): 80.0,
    },
    time_limit_seconds=5.0,  # Mandatory finite timeout
    method="highs",  # High-performance HiGHS solver
)

# 3. Propose Actions
actions = planner.propose(state=state)

for act in actions:
    p = act.parameters
    print(f"Transfer: {p['quantity']:.2f} from {p['source_resource']} to {p['target_resource']}")

# Diagnostic output
print(f"Status: {planner.last_result.status.value}")
print(f"Solve Time: {planner.last_result.solve_time_seconds:.5f}s")
```

---

## Integrating into Simulation Rollouts

Because `SciPyAllocationPlanner` implements [`Actor`](../reference/api/core.md), it can be placed directly in a `World`:

```python
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario

world = World(
    state=state,
    dynamics=DeterministicTransferDynamics(),
    actors=[planner],
)

scenario = Scenario(scenario_id="continuous_alloc", horizon=3, samples=1, seed=42)
result = SimulationEngine().run(world, scenario)

print("Rollout completed:", result.trajectories[0].status.name)
```
