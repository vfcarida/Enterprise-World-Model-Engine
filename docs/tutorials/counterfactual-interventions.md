# Tutorial: Simulating Interventions & Comparing Scenarios

In this tutorial, you will learn how to evaluate policy interventions against a status-quo baseline using counterfactual simulation and bootstrap evaluation.

By the end of this lesson, you will understand how to:
1. Branch an existing world to ensure complete state isolation (AC-005).
2. Schedule interventional actions at specific simulation steps.
3. Quantify distributional deltas with bootstrap confidence intervals.

---

## 1. Branching the Baseline World

EWM Engine guarantees that branching a world creates an independent logical state without shared mutable references:

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

# Baseline world setup
state = WorldState(
    entities=[
        Entity(id="wh_a", type="warehouse"),
        Entity(id="wh_b", type="warehouse"),
    ],
    resources=[
        Resource(id="stock_a", current=100.0, min_value=0.0, max_value=200.0),
        Resource(id="stock_b", current=20.0, min_value=0.0, max_value=200.0),
    ],
)
world_baseline = World(
    state=state,
    dynamics=DeterministicTransferDynamics(),
    constraints=[ResourceCapacityConstraint(resource_id="stock_a")],
)

# Branch the world for the intervention candidate
world_intervention = world_baseline.branch()
```

---

## 2. Scheduling an Interventional Action

Schedule a transfer action ($A_1$) transferring 40 units of stock from `wh_a` to `wh_b` at step 1:

```python
scenario_baseline = Scenario(
    scenario_id="baseline",
    name="Baseline",
    horizon=2,
    samples=1,
    seed=42,
)

scenario_intervention = Scenario(
    scenario_id="intervention",
    name="Transfer 40 Units A -> B",
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
                    "source_resource": "stock_a",
                    "target_resource": "stock_b",
                    "quantity": 40.0,
                },
            ),
        ),
    ),
)

engine = SimulationEngine()
res_baseline = engine.run(world_baseline, scenario_baseline)
res_intervention = engine.run(world_intervention, scenario_intervention)
```

---

## 3. Comparing Counterfactual Scenarios

Compare the baseline and candidate scenario distributions:

```python
comparison = compare_scenarios(
    baseline=res_baseline,
    candidates=[res_intervention],
    metrics=["resource_stock_a", "resource_stock_b", "violations_count"],
)

print(comparison.summary_table())
```

### Expected Output

```text
=== Scenario Comparison (Baseline: Baseline) ===
---------------------------------------------------------------------------------------------------------------------------
Metric                    | Scenario             | Mean (Std)         | p50 [p05, p95]       | Delta vs Base [CI] (*sig)   
---------------------------------------------------------------------------------------------------------------------------
resource_stock_a          | Baseline             | 100.00 (+/-0.00)   | 100.00 [100.00, 100.00] | -                           
                          | Transfer 40 Units A -> B | 60.00 (+/-0.00)    | 60.00 [60.00, 60.00] | -40.00 (-40.0%) [-40.00, -40.00] *
---------------------------------------------------------------------------------------------------------------------------
resource_stock_b          | Baseline             | 20.00 (+/-0.00)    | 20.00 [20.00, 20.00] | -                           
                          | Transfer 40 Units A -> B | 60.00 (+/-0.00)    | 60.00 [60.00, 60.00] | +40.00 (+200.0%) [+40.00, +40.00] *
---------------------------------------------------------------------------------------------------------------------------
violations_count          | Baseline             | 0.00 (+/-0.00)     | 0.00 [0.00, 0.00]    | -                           
                          | Transfer 40 Units A -> B | 0.00 (+/-0.00)     | 0.00 [0.00, 0.00]    | +0.00 (+0.0%) [+0.00, +0.00]
---------------------------------------------------------------------------------------------------------------------------
```

Notice how the engine records exact empirical percentile confidence intervals and automatically flags statistically significant shifts under $P_{\text{model}}$.
