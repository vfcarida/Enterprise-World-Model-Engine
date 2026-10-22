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
                          | Transfer 40 Units A -> B | 60.00 (+/-0.00)    | 60.00 [60.00, 60.00] | -40.00 (-40.0%) [-40.00, -40.00] (det)
---------------------------------------------------------------------------------------------------------------------------
resource_stock_b          | Baseline             | 20.00 (+/-0.00)    | 20.00 [20.00, 20.00] | -                           
                          | Transfer 40 Units A -> B | 60.00 (+/-0.00)    | 60.00 [60.00, 60.00] | +40.00 (+200.0%) [+40.00, +40.00] (det)
---------------------------------------------------------------------------------------------------------------------------
violations_count          | Baseline             | 0.00 (+/-0.00)     | 0.00 [0.00, 0.00]    | -                           
                          | Transfer 40 Units A -> B | 0.00 (+/-0.00)     | 0.00 [0.00, 0.00]    | +0.00 (+0.0%) [+0.00, +0.00]
---------------------------------------------------------------------------------------------------------------------------
```

Notice how the engine distinguishes single-sample/zero-variance deterministic deltas (`(det)`, `status_label="deterministic_difference"`) from multi-sample stochastic Monte Carlo shifts where non-degenerate percentile bootstrap confidence intervals exclude zero (`*`, `status_label="statistically_significant"` under $P_{\text{model}}$).

---

## 4. Native Pearl Rung-3 Counterfactuals (Twin-Rollout SCM Coupling)

While scenario branching evaluates forward interventions at **Pearl Rung 2** ($do(X)$), EWM Engine natively supports **Pearl Rung 3** counterfactuals: *"Given what was observed for this specific entity, what would have happened had we intervened, under the exact same realized exogenous noise?"*

Anchored in **Richens & Everitt (ICLR 2024 oral, arXiv:2402.10877)**, the engine compiles the typed graph into a Structural Causal Model (SCM) view:

```python
from ewm_engine.core.actions import Intervention

# 1. Run or observe the factual trajectory
factual_traj = world_baseline.simulate(scenario_baseline).trajectories[0]

# 2. Formulate the causal intervention
intervention = Intervention(
    id="emergency_reroute",
    description="Emergency reroute of stock to warehouse B",
    parameters={"stock_b": 60.0},
)

# 3. Execute Pearl's three-step counterfactual:
#    - Abduction: Infer realized exogenous noise U from the factual trace
#    - Action: Apply structural graph surgery do(stock_b = 60.0)
#    - Prediction: Re-run with Twin-Rollout noise coupling (identical U)
cf_result = world_baseline.counterfactual(
    observed=factual_traj,
    intervention=intervention,
)

print(f"Evidence Level: {cf_result.evidence_level.value}")
# -> 'counterfactual'

print(f"Disclaimer: {cf_result.disclaimer}")
# -> Model-counterfactual evaluated within the fully specified simulated SCM...

# Export the systemic divergence diagram in Mermaid notation
print(cf_result.trace_diff.to_mermaid())
```

For a comprehensive worked example preventing warehouse stockouts under identical demand shocks, see [examples/counterfactual_warehouse_reroute.py](file:///examples/counterfactual_warehouse_reroute.py) and [ADR-040](file:///docs/adr/ADR-040-twin-rollout-counterfactual-scm-kernel.md).
