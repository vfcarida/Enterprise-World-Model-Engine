# Recipe: Macroeconomic Stress Testing with PRIM Failure Discovery

> **Task**: Run a multi-variate parameter sweep across uncertain macroeconomic variables and automatically isolate the exact operational boundary where performance fails using PRIM.

---

## 1. The Recipe Code

```python
import numpy as np
from ewm_engine.decision.discovery import ScenarioDiscovery


def execute_stress_test_recipe(num_samples: int = 1000) -> None:
    # 1. Sample Multi-Factor Parameters: [Interest Rate Surge (%), Energy Inflation (%)]
    rng = np.random.default_rng(123)
    parameters = rng.uniform(low=[0.0, 0.0], high=[0.08, 0.50], size=(num_samples, 2))

    # 2. Simulate Capital Coverage Ratio (%)
    # Synthetic response surface: baseline ratio = 14.0%
    coverage_ratios = 14.0 - (parameters[:, 0] * 80.0) - (parameters[:, 1] * 12.0)

    # 3. Discover Vulnerability Subspace where Coverage Ratio < 9.0% (Statutory Minimum)
    discovery = ScenarioDiscovery(peeling_alpha=0.05, min_box_mass=0.10)
    box = discovery.find_box(
        parameters=parameters,
        outcomes=coverage_ratios,
        failure_criterion=lambda y: y < 9.0,
        parameter_names=["rate_surge", "energy_inflation"],
    )

    print("==================================================")
    print("PRIM VULNERABILITY SUBSPACE DISCOVERY RESULT")
    print("==================================================")
    print(f"Evaluated Runs:  {num_samples}")
    print(f"Subspace Density: {box.density:.1%} (Concentration of failures in box)")
    print(f"Subspace Coverage: {box.coverage:.1%} (Fraction of all failures captured)")
    print("\nIdentified Failure Restrictions:")
    for name, (low, high) in box.restrictions.items():
        print(f"  • {name}: [{low:.4f}, {high:.4f}]")


if __name__ == "__main__":
    execute_stress_test_recipe()
```

---

## 2. Interpreting the Result
* **Actionable Hedging Rule:** If the box indicates `rate_surge >= 0.045`, risk officers know that interest rate increases above 4.5% are the dominant driver of failure, regardless of energy costs.
* **Audit Trail:** The box boundaries can be exported directly into the `ModelCard` risk management section.
