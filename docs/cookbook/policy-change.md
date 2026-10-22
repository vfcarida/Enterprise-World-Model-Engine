# Recipe: How Do I Test a Policy Change Before Deploying It?

> **Question:** *Our operations team wants to change a critical routing rule or capacity allocation. How can we simulate the downstream effects under stochastic variation before deploying it to production?*

---

## The Scenario

Suppose your organization manages two regional fulfillment nodes (`stock_north` and `stock_south`). Under current operations, `stock_south` experiences recurrent stockouts. Leadership proposes shifting 30 inventory units from North to South at $t=1$.

You need to evaluate:
1. Does this intervention statistically improve availability in South?
2. Does it breach capacity or create upstream stockouts in North?
3. What is the 90% confidence interval on the intervention delta?

---

## The Recipe Code

```python
import ewm_engine as ewm
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics

# Step 1: Define initial state with bounded resources
world = ewm.World(
    state=ewm.WorldState(
        resources=[
            ewm.Resource(id="stock_north", current=100.0, min_value=0.0, max_value=200.0),
            ewm.Resource(id="stock_south", current=20.0, min_value=0.0, max_value=200.0),
        ]
    ),
    dynamics=DeterministicTransferDynamics(),
)

# Step 2: Simulate baseline factual rollout (Status Quo)
baseline = world.simulate(horizon=5, seed=42, name="Status Quo")

# Step 3: Branch world and simulate intervention do(transfer = 30)
counterfactual = world.branch().simulate(
    actions=[
        ewm.Action(
            type="transfer_resource",
            parameters={
                "quantity": 30.0,
                "source_resource": "stock_north",
                "target_resource": "stock_south",
            },
        )
    ],
    horizon=5,
    seed=42,
    name="Transfer 30 North -> South",
)

# Step 4: Compare outcomes with non-parametric bootstrap confidence intervals
diff = ewm.compare_scenarios(
    baseline=baseline,
    candidates=[counterfactual],
    metrics=["resource_stock_north", "resource_stock_south"],
    confidence_level=0.95,
)

print(diff.summary_table())
```

---

## Verification & Audit Guarantees

1. **Conservation Invariance:**
   Total stock across nodes remains exactly constant:
   $$\text{stock}_{\text{north}}(t) + \text{stock}_{\text{south}}(t) = 120.0 \quad \forall t$$
2. **Branch Isolation:**
   Mutating or simulating `counterfactual` does not alter the baseline `world` object or its cryptographic SHA-256 state fingerprint.
3. **Deterministic vs. Stochastic Delta Status:**
   Zero-variance deterministic differences (such as single-trajectory deterministic runs) are labeled `(det)` in `summary_table()`, whereas stochastic Monte Carlo runs (`samples >= 2`) whose bootstrap confidence intervals exclude zero are flagged with `*`.
