# Recipe: How Do I Enforce Strict Conservation and Accounting Laws?

> **Question:** *Unlike pure statistical models or LLMs that hallucinate money or inventory, how do I guarantee that physical and financial balance equations hold strictly across all simulation steps?*

---

## The Scenario

In socio-technical systems, certain conservation laws must never be breached:
- **Mass / Inventory Balance:** Inflow - Outflow = Change in Stock.
- **Cash Flow Accounting:** Cash cannot be negative without an explicit credit facility.
- **Physical Capacity:** Warehouses cannot exceed their physical cubic meter capacity.

EWM Engine guarantees these invariants by construction via **Pre-Action** and **Post-Transition** constraint gates.

---

## The Recipe Code

```python
import ewm_engine as ewm
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics

# Step 1: Initialize resources with hard bounds [min_value, max_value]
state = ewm.WorldState(
    resources=[
        ewm.Resource(
            id="treasury_cash",
            current=1000.0,
            min_value=0.0,  # Hard floor: insolvency barrier
            max_value=5000.0,
        ),
        ewm.Resource(
            id="escrow_reserve",
            current=200.0,
            min_value=0.0,
            max_value=2000.0,
        ),
    ]
)

# Step 2: Attach hard conservation constraints
world = ewm.World(
    state=state,
    dynamics=DeterministicTransferDynamics(),
    constraints=[
        ResourceCapacityConstraint(resource_id="treasury_cash"),
        ResourceCapacityConstraint(resource_id="escrow_reserve"),
    ],
)

# Step 3: Propose a valid transfer within conservation boundaries
valid_rollout = world.simulate(
    actions=[
        ewm.Action(
            type="transfer_resource",
            parameters={
                "quantity": 300.0,
                "source_resource": "treasury_cash",
                "target_resource": "escrow_reserve",
            },
        )
    ],
    horizon=2,
    seed=42,
)

final_treasury = valid_rollout.trajectories[0].final_state.resources["treasury_cash"].current
final_escrow = valid_rollout.trajectories[0].final_state.resources["escrow_reserve"].current
print(f"Valid Transfer: Treasury = {final_treasury}, Escrow = {final_escrow}")
assert final_treasury + final_escrow == 1200.0  # Exact conservation preserved

# Step 4: Attempt an invalid action that breaches minimum liquidity (overdraft)
invalid_branch = world.branch()
overdraft_rollout = invalid_branch.simulate(
    actions=[
        ewm.Action(
            type="transfer_resource",
            parameters={
                "quantity": 1500.0,  # Exceeds 1000.0 cash!
                "source_resource": "treasury_cash",
                "target_resource": "escrow_reserve",
            },
        )
    ],
    horizon=2,
    seed=42,
)

# The engine detects the hard violation and invalidates the trajectory
traj = overdraft_rollout.trajectories[0]
print(f"Overdraft Status: {traj.status.value}")  # "invalid"
print(f"Hard Violations Count: {traj.hard_violations}")
```

---

## Verification & Audit Guarantees

1. **Deterministic Invalidation:**  
   When a hard invariant is violated, the rollout is marked `INVALID`. It is permanently flagged in `SimulationResult.invalid_count` and prevented from biasing downstream decision calculations.
2. **Teaching Error on Demand:**  
   If step results are queried directly, `step.constraint_violations[0].raise_for_status()` raises `ConstraintViolationTeachingError` explaining exactly which resource breached its bounds.
