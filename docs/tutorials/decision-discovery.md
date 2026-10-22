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
from ewm_engine.decision.discovery import ParameterSpace, run_prim
from ewm_engine.decision.robustness import evaluate_decision_robustness

# 1. Define candidate strategies across 4 deeply uncertain future macro regimes ($M NPV)
outcomes_by_strategy = {
    "Aggressive Expansion": [120.0, 15.0, 30.0, 45.0],
    "Modular Resilience": [85.0, 70.0, 55.0, 80.0],
    "Conservative Cash": [50.0, 50.0, 50.0, 50.0],
}

# 2. Evaluate Robustness via Minimax Regret & Satisficing (Hurdle = $60M)
ranking = evaluate_decision_robustness(
    outcomes=outcomes_by_strategy,
    satisficing_threshold=60.0,
    higher_is_better=True,
)
print("Recommended (Minimax Regret):", ranking.recommended_by_regret)
```

---

## 3. Isolating the Failure Subspace via PRIM

To understand why "Aggressive Expansion" failed in World 2, we run PRIM over the multi-parameter sample:

```python
# Sample 500 combinations of [market_drop, debt_cost_increase]
rng = np.random.default_rng(42)
raw = rng.uniform(low=[0.0, 0.0], high=[0.40, 0.06], size=(500, 2))
points = [{"market_drop": float(r[0]), "debt_cost_surge": float(r[1])} for r in raw]
outcomes = 100.0 - (raw[:, 0] * 180.0) - (raw[:, 1] * 800.0)
failures = [bool(y < 40.0) for y in outcomes]

# Discover failure subspace (Payoff < $40M)
space = ParameterSpace(bounds={"market_drop": (0.0, 0.40), "debt_cost_surge": (0.0, 0.06)})
result = run_prim(points=points, targets=failures, parameter_space=space, alpha=0.05)
box = result.selected_box

print(f"Discovered Subspace Density:  {box.density:.1%}")
print(f"Discovered Subspace Coverage: {box.coverage:.1%}")
```

---

## 4. Key Takeaways
* **Minimax Regret identifies the robust choice:** Even if Aggressive Expansion has the highest peak payoff in a boom, Modular Resilience minimizes maximum hindsight regret ($8.0M vs $55.0M$).
* **PRIM rules are explainable:** The discovered box gives leadership clear, concrete operational thresholds (e.g. `market_drop >= 22%`) to monitor.
