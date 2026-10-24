# Result Card: Scenario Discovery (PRIM) & Minimax Regret Ranking

> **Identity & Scientific Domain**: Decision-Focused Learning & Robust Decision Making  
> **Benchmark Reference**: Friedman, J. H., & Fisher, N. I. (1999). "Bump hunting in high-dimensional data." *Statistics and Computing*, 9(2), 123–143.  
> **Reproduction Command**: `python scripts/reproduce_results.py --card decision_discovery`

---

## 1. Benchmark Overview & Scenario

Evaluates candidate corporate strategies across an ensemble of 500 multi-factor macroeconomic stress scenarios:
* Parameters: Exogenous demand shock $A \in [0.0, 0.5]$ and cost shock $B \in [0.0, 0.5]$.
* Failure Criterion: Operating performance metric $Y < 10.0$.
* Objectives:
  1. Automatically isolate the failure boundary using PRIM.
  2. Rank candidate decisions ("Fragile" vs. "Robust") using Savage's Minimax Regret.

---

## 2. Verified Metrics & Honesty Scorecard

```json
{
  "card": "decision_discovery_s02",
  "prim_box_density": 0.9116279069767442,
  "prim_box_coverage": 0.8,
  "minimax_regret_winner": "Robust",
  "dominant_action": "Robust",
  "status": "PASSED",
  "card_verification_status": "PASSED_PRIM_AND_MINIMAX_REGRET_VERIFIED"
}
```

* **PRIM Subspace Density:** $91.2\%$ (over 9 out of 10 points within the discovered box are true failure cases).
* **PRIM Subspace Coverage:** $80.0\%$ (the single rule captures $80\%$ of all simulated enterprise failures).
* **Minimax Regret Winner:** "Robust" strategy dominates with bounded worst-case hindsight regret.

---

## 3. Single-Command Reproduction

```bash
python scripts/reproduce_results.py --card decision_discovery
```
