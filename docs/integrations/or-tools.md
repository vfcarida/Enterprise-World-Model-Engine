# Operations Research with Google OR-Tools

> [!NOTE]
> **Maturity**: Beta. Enforces mandatory finite time limits and implements formal `ConstraintSolver` and `ActionPlanner` protocols. Requires `ewm-engine[or]` or `ewm-engine[solvers]`.

EWM Engine integrates [Google OR-Tools](https://developers.google.com/optimization) via two dedicated components:

1. **`ORToolsAllocationAdapter`**: Solves continuous network flow and resource reallocation problems using Mixed-Integer Linear Programming (MILP / GLOP).
2. **`CPSATAllocationPlanner`**: Solves discrete integer multi-depot distribution problems using Constraint Programming (CP-SAT), with support for extracting auditable unsatisfiable cores (infeasibility certificates).

---

## 1. Discrete Allocation Planner: `CPSATAllocationPlanner`

The `CPSATAllocationPlanner` acts as an interventional decision [`Actor`](../reference/api/core.md) and [`ActionPlanner`](overview.md). It proposes candidate actions to the simulation engine, which are strictly validated through the engine's constraint pipeline.

### Runnable Example: Optimal Warehouse Dispatch

```python
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.integrations.ortools import CPSATAllocationPlanner

# 1. State with supplying depots and demanding clinics
state = WorldState(
    resources=[
        Resource(id="depot_1", current=100.0, min_value=0.0, max_value=200.0),
        Resource(id="depot_2", current=80.0, min_value=0.0, max_value=200.0),
        Resource(id="clinic_north", current=0.0, min_value=0.0, max_value=100.0),
        Resource(id="clinic_south", current=0.0, min_value=0.0, max_value=100.0),
    ]
)

# 2. Configure CP-SAT Planner with finite time limit (mandatory)
planner = CPSATAllocationPlanner(
    actor_id="cpsat_logistics",
    sources=["depot_1", "depot_2"],
    destinations=["clinic_north", "clinic_south"],
    demands={"clinic_north": 50, "clinic_south": 40},
    capacities={
        ("depot_1", "clinic_north"): 60,
        ("depot_2", "clinic_south"): 50,
    },
    costs={
        ("depot_1", "clinic_north"): 2,
        ("depot_2", "clinic_south"): 3,
    },
    time_limit_seconds=5.0,  # Mandatory resource limit
)

# 3. Propose Candidate Actions (Planners NEVER mutate state directly)
actions = planner.propose(state=state)

for act in actions:
    p = act.parameters
    print(f"Action: Transfer {p['quantity']} from {p['source_resource']} to {p['target_resource']}")

# Diagnostic output
print(f"Solver Status: {planner.last_result.status.value}")
print(f"Solve Time: {planner.last_result.solve_time_seconds:.4f}s")
```

---

## 2. Infeasibility Certificates & Unsat Core Auditability

When an allocation request is mathematically impossible due to physical bottlenecks (e.g. total supply < total demand, or corridor throughput limits), CP-SAT tracks assumption variables and surfaces the minimal **unsatisfiable core (`unsat_core`)**.

```python
# Demand exceeds total depot capacity
infeasible_planner = CPSATAllocationPlanner(
    actor_id="audit_planner",
    sources=["depot_empty"],
    destinations=["hospital_urgent"],
    demands={"hospital_urgent": 250},
    time_limit_seconds=5.0,
)

state = WorldState(
    resources=[
        Resource(id="depot_empty", current=50.0, min_value=0.0, max_value=100.0),
        Resource(id="hospital_urgent", current=0.0, min_value=0.0, max_value=500.0),
    ]
)

actions = infeasible_planner.propose(state=state)
result = infeasible_planner.last_result

print(f"Feasible: {result.satisfied}")  # False
print(f"Status: {result.status.value}")  # 'infeasible'
print("Unsat Core Certificate:", result.unsat_core)
# Output: ('demand_target_hospital_urgent_250', 'supply_capacity_depot_empty_50')
```

---

## 3. Continuous Flow Adapter: `ORToolsAllocationAdapter`

For continuous linear programming (e.g. fluid transfers, continuous monetary budgets):

```python
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.integrations.ortools import ORToolsAllocationAdapter

adapter = ORToolsAllocationAdapter(solver_type="GLOP", default_time_limit_seconds=5.0)

state = WorldState(
    resources=[
        Resource(id="wh_a", current=100.0, min_value=0.0, max_value=200.0),
        Resource(id="wh_b", current=150.0, min_value=0.0, max_value=200.0),
        Resource(id="store_c", current=10.0, min_value=0.0, max_value=200.0),
    ]
)

actions = adapter.optimize_transfers(
    state=state,
    sources=["wh_a", "wh_b"],
    destinations=["store_c"],
    demands={"store_c": 50.0},
    costs={("wh_a", "store_c"): 2.0, ("wh_b", "store_c"): 1.5},
)

# Verify actions against state invariants
check_result = adapter.check(state=state, actions=actions)
print(f"Feasible: {check_result.satisfied}")
```
