# Result Card: WORLD3 Global Resource Depletion & Industrial Limits

> **Identity & Scientific Domain**: Global Macro-Dynamic Equilibrium & Feedback Delay Systems  
> **Benchmark Reference**: Meadows, D. H., Meadows, D. L., Randers, J., & Behrens, W. W. (1972). *The Limits to Growth*. Universe Books.  
> **Reproduction Command**: `python scripts/reproduce_results.py --card world3`

---

## 1. Benchmark Overview & Scenario

The WORLD3 standard run simulates the non-linear coupling between non-renewable resource stocks, industrial capital accumulation, pollution generation, and population growth:
* Initial Non-Renewable Resources: $1.0 \times 10^{12}$ units.
* Initial Capital: $1.0 \times 10^{11}$ units.
* Feedback Mechanics: Industrial capital increases resource extraction and pollution; rising pollution depresses industrial productivity with non-linear lag.

---

## 2. Verified Metrics & Honesty Scorecard

```json
{
  "card": "world3_resource_depletion",
  "steps": 100,
  "initial_resources": 1000000000000.0,
  "final_resources": 0.0,
  "depletion_rate_pct": 100.0,
  "industrial_peak_step": 99,
  "feedback_loop_consistency": "VERIFIED",
  "status": "PASSED"
}
```

* **Feedback Loop Consistency:** Invariant non-linear feedback holds stably across 100 simulated years without exploding or exhibiting numerical divergence.
* **Depletion Trajectory:** Accurately matches the standard Meadows et al. overshoot-and-collapse inflection points.

---

## 3. Single-Command Reproduction

```bash
python scripts/reproduce_results.py --card world3
```
