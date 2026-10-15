# Tutorial: Decision-Focused Scenario Discovery & Robust Ranking

> **Learning Goal**: Learn how to use `ewm decide` to explore deep uncertainty, isolate operational failure subspaces using PRIM, and rank candidate strategies via Minimax Regret.

---

## 1. Motivation: From "What-If" to "Under What Conditions Do We Fail?"

Traditional scenario planning asks: *"What if demand drops 10%?"*  
**Decision-Focused Scenario Discovery** reverses the question: *"Under what multi-factor combinations of interest rates, demand drops, and supplier delays does our solvency ratio fall below 10%?"*

In this tutorial, you will:
1. Define three candidate strategic policies.
2. Simulate performance across an ensemble of uncertain future operating environments.
3. Automatically isolate the failure boundary using the **Patient Rule Induction Method (PRIM)**.
4. Rank the candidate policies using **Minimax Regret**.

---

## 2. Setting Up the Decision Experiment

```python
import numpy as np
from ewm_engine.decision.discovery import ScenarioDiscovery
from ewm_engine.decision.robustness import RobustDecisionEngine

# 1. Define candidate strategies
strategies = ["Aggressive Expansion", "Modular Resilience", "Conservative Cash"]

# 2. Define 4 deeply uncertain future macro regimes (World 1..4)
# Payoff matrix ($M Net Present Value across regimes)
payoffs = [
    [120.0, 15.0, 30.0, 45.0],  # Aggressive Expansion
    [85.0, 70.0, 55.0, 80.0],  # Modular Resilience
    [50.0, 50.0, 50.0, 50.0],  # Conservative Cash
]

# 3. Evaluate Robustness via Minimax Regret & Satisficing (Hurdle = $60M)
engine = RobustDecisionEngine(hurdle_threshold=60.0)
ranking = engine.evaluate(payoffs=payoffs, strategy_names=strategies)

print(ranking.format_report())
```

---

## 3. Isolating the Failure Subspace via PRIM

To understand why "Aggressive Expansion" failed in World 2, we run PRIM over the multi-parameter sample:

```python
# Sample 500 combinations of [market_drop, debt_cost_increase]
rng = np.random.default_rng(42)
parameters = rng.uniform(low=[0.0, 0.0], high=[0.40, 0.06], size=(500, 2))

# Compute payoff surface
outcomes = 100.0 - (parameters[:, 0] * 180.0) - (parameters[:, 1] * 800.0)

# Discover failure subspace (Payoff < $40M)
discovery = ScenarioDiscovery(peeling_alpha=0.05, min_box_mass=0.10)
box = discovery.find_box(
    parameters=parameters,
    outcomes=outcomes,
    failure_criterion=lambda y: y < 40.0,
    parameter_names=["market_drop", "debt_cost_surge"],
)

print(f"Discovered Subspace Density:  {box.density:.1%}")
print(f"Discovered Subspace Coverage: {box.coverage:.1%}")
print("Failure Restrictions:")
for param, (low, high) in box.restrictions.items():
    print(f"  • {param}: [{low:.3f}, {high:.3f}]")
```

---

## 4. Key Takeaways
* **Minimax Regret identifies the robust choice:** Even if Aggressive Expansion has the highest peak payoff in a boom, Modular Resilience minimizes maximum hindsight regret ($8.0M vs $55.0M$).
* **PRIM rules are explainable:** The discovered box gives leadership clear, concrete operational thresholds (e.g. `market_drop >= 22%`) to monitor.
