# Recipe: Macroeconomic Stress Testing with PRIM Failure Discovery

> **Task**: Run a multi-variate parameter sweep across uncertain macroeconomic variables and automatically isolate the exact operational boundary where performance fails using PRIM.

---

## 1. The Recipe Code

```python
import numpy as np
from ewm_engine.decision.discovery import ParameterSpace, run_prim


def execute_stress_test_recipe(num_samples: int = 1000) -> None:
    # 1. Sample Multi-Factor Parameters: [Interest Rate Surge (%), Energy Inflation (%)]
    rng = np.random.default_rng(123)
    raw = rng.uniform(low=[0.0, 0.0], high=[0.08, 0.50], size=(num_samples, 2))
    points = [{"rate_surge": float(r[0]), "energy_inflation": float(r[1])} for r in raw]

    # 2. Simulate Capital Coverage Ratio (%) and binary failure (< 9.0%)
    coverage_ratios = 14.0 - (raw[:, 0] * 80.0) - (raw[:, 1] * 12.0)
    failures = [bool(val < 9.0) for val in coverage_ratios]

    # 3. Discover Vulnerability Subspace where Coverage Ratio < 9.0%
    space = ParameterSpace(bounds={"rate_surge": (0.0, 0.08), "energy_inflation": (0.0, 0.50)})
    result = run_prim(points=points, targets=failures, parameter_space=space, alpha=0.05)
    box = result.selected_box

    print(
        f"Evaluated Runs: {num_samples} | Subspace Density: {box.density:.1%} | Coverage: {box.coverage:.1%}"
    )


if __name__ == "__main__":
    execute_stress_test_recipe()
```

---

## 2. Interpreting the Result
* **Actionable Hedging Rule:** If the box indicates `rate_surge >= 0.045`, risk officers know that interest rate increases above 4.5% are the dominant driver of failure, regardless of energy costs.
* **Audit Trail:** The box boundaries can be exported directly into the `ModelCard` risk management section.
