# How-To Guide: Scenario Discovery with PRIM

> **Goal**: Use the Patient Rule Induction Method (PRIM) to isolate actionable, hyper-rectangular vulnerability bounds from high-dimensional simulation datasets.

---

## 1. Prerequisites
Ensure `ewm_engine.decision.discovery` is imported:

```python
from ewm_engine.decision.discovery import ParameterSpace, run_prim
```

---

## 2. Step-by-Step Procedure

### Step 1: Prepare Parameters and Outcomes
Assemble your input simulation matrix $X$ ($N \times D$) and the corresponding scalar outcome array $Y$ ($N$):

```python
import numpy as np

# N=1000 runs, D=3 parameters: [labor_shortage_pct, transit_delay_days, tariff_surge_pct]
rng = np.random.default_rng(101)
X = rng.uniform(low=[0.0, 1.0, 0.0], high=[0.30, 14.0, 0.25], size=(1000, 3))

# Target metric: on-time delivery rate (in %)
Y = 98.0 - (X[:, 0] * 50.0) - (X[:, 1] * 2.5) - (X[:, 2] * 20.0)
```

### Step 2: Configure and Run PRIM
Configure `ParameterSpace` and run `run_prim` with peeling fraction $\alpha$ (typically $0.05$) and minimum support ratio:

```python
points = [
    {"labor_shortage": float(r[0]), "transit_delay": float(r[1]), "tariff_surge": float(r[2])}
    for r in X
]
failures = [bool(val < 85.0) for val in Y]  # SLA failure threshold
space = ParameterSpace(
    bounds={"labor_shortage": (0.0, 0.30), "transit_delay": (1.0, 14.0), "tariff_surge": (0.0, 0.25)}
)

result = run_prim(points=points, targets=failures, parameter_space=space, alpha=0.05)
box = result.selected_box
```

### Step 3: Interpret and Export Discovered Box
```python
print(f"Box Purity (Density):  {box.density:.1%}")
print(f"Box Coverage of Total Failures: {box.coverage:.1%}")
print(f"Total Trajectories in Box:      {box.mass:.1%}")

# Export restrictions dictionary
restrictions = box.restrictions
for dim, (lower, upper) in restrictions.items():
    print(f"Rule: {lower:.2f} <= {dim} <= {upper:.2f}")
```

---

## 3. Best Practices
* **Tune `min_box_mass`:** If rules are too complex, increase `min_box_mass` to $0.15$ to force broader, more robust boxes.
* **Visualize with Scenarios:** Feed discovered boundaries back into `Scenario` definitions to test targeted risk-mitigation interventions.
