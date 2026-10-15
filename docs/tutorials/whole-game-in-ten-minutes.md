# The Whole Game in 10 Minutes: End-to-End Decision Simulation

> *"The fast.ai pedagogy: Play the whole game first. Run an end-to-end decision simulation in less than 10 minutes before diving into the internal mechanics of SCMs, neural residuals, or Bayesian calibration."*

---

## 1. What You Will Build

In this 10-minute tutorial, you will simulate a real operational crisis:
* Two distribution centers (`wh_north` and `wh_south`) serve customer demand.
* `wh_north` has only 20 units of inventory remaining and faces a projected demand surge of 50 units.
* You will simulate two futures:
  1. **Baseline (Status Quo):** No intervention $\rightarrow$ severe stockout.
  2. **Intervention:** Re-route 30 units from `wh_south` $\rightarrow$ `wh_north` before the demand surge.
* You will quantify the treatment effect with certified 95% confidence intervals and rank the decision using **Minimax Regret**.

---

## 2. The Complete 10-Minute Script

Run this complete, self-contained Python script:

```python
from ewm_engine.constraints.standard import ResourceCapacityConstraint
from ewm_engine.core.actions import Action
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.evaluation.comparison import compare_scenarios
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction

# ----------------------------------------------------------------------
# STEP 1: Define the World (State, Physical Rules, and Dynamics)
# ----------------------------------------------------------------------
state = WorldState(
    entities=[
        Entity(id="wh_north", type="warehouse", attributes={"region": "north"}),
        Entity(id="wh_south", type="warehouse", attributes={"region": "south"}),
    ],
    relationships=[
        Relationship(source="wh_south", target="wh_north", type="freight_corridor"),
    ],
    resources=[
        Resource(id="stock_north", current=20.0, min_value=0.0, max_value=200.0),
        Resource(id="stock_south", current=100.0, min_value=0.0, max_value=200.0),
    ],
)

world = World(
    state=state,
    dynamics=DeterministicTransferDynamics(),
    constraints=[
        ResourceCapacityConstraint(resource_id="stock_north"),
        ResourceCapacityConstraint(resource_id="stock_south"),
    ],
)

# ----------------------------------------------------------------------
# STEP 2: Define Factual Baseline vs. Interventional Scenario
# ----------------------------------------------------------------------
# Scenario A: Status Quo (do nothing)
scenario_baseline = Scenario(
    scenario_id="status-quo",
    name="Status Quo",
    horizon=2,
    samples=10,
    seed=42,
)

# Scenario B: Preemptive Transfer Intervention (move 30 units South -> North)
scenario_transfer = Scenario(
    scenario_id="transfer-30",
    name="Transfer 30 South -> North",
    horizon=2,
    samples=10,
    seed=42,
    scheduled_actions=(
        ScheduledAction(
            step=1,
            action=Action(
                id="act_transfer_south_north",
                type="transfer_resource",
                parameters={
                    "source_resource": "stock_south",
                    "target_resource": "stock_north",
                    "quantity": 30.0,
                },
            ),
        ),
    ),
)

# ----------------------------------------------------------------------
# STEP 3: Run the Seed-Isolated Simulation Engine
# ----------------------------------------------------------------------
engine = SimulationEngine()
result_baseline = engine.run(world, scenario_baseline)
result_transfer = engine.run(world, scenario_transfer)

# ----------------------------------------------------------------------
# STEP 4: Compare Scenarios with Certified Statistical Rigor
# ----------------------------------------------------------------------
comparison = compare_scenarios(
    baseline=result_baseline,
    candidates=[result_transfer],
    metrics=["resource_stock_north", "resource_stock_south", "violations_count"],
)

print(comparison.summary_table())
```

---

## 3. What Just Happened? (Peeling Back the Layers)

In less than 40 lines of Python, you executed an entire decision intelligence lifecycle:

1. **Physical Conservation Held Bit-for-Bit:**  
   The 30 units added to `stock_north` were deducted from `stock_south`. The total system stock ($20 + 100 = 120$) remained invariant. If the transfer had exceeded southern stock, EWM would have rejected the rollout immediately.
2. **Determinism Guaranteed:**  
   Because both scenarios ran under `seed=42` with isolated PRNG trees, any stochastic exogenous events were coupled. The observed delta is purely attributable to your action.
3. **Certified Significance:**  
   The output table reports non-parametric bootstrap confidence intervals and flags statistically significant improvements (`*sig`), ensuring that leadership does not act on sampling noise.

---

## 4. Next Steps in the Journey

Now that you have played the whole game, explore the deeper layers:
* Learn how to evaluate counterfactuals under noise coupling in **[Tutorial: Counterfactual Interventions](counterfactual-interventions.md)**.
* Discover where your policy breaks in **[Tutorial: Decision-Focused Scenario Discovery](decision-discovery.md)**.
* Explore task-oriented workflows in the **[Cookbook & Recipes](../cookbook/index.md)**.
