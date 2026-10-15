# Tutorial: Amortized Calibration with Certified Coverage

> **Learning Goal**: Learn how to calibrate unobservable simulation parameters from real-world telemetry using Simulation-Based Inference (SBI) and certify the calibration coverage using TARP diagnostics.

---

## 1. Motivation: Why Standard Curve-Fitting Fails
In complex organizational simulators, many parameters (e.g. true customer elasticity, internal processing latency, counterparty panic thresholds) cannot be directly measured.
* **Standard Least-Squares:** Optimizes a single point estimate $\hat{\theta}$, concealing parametric uncertainty.
* **Likelihood-Free Bayesian Inference:** Infers the complete posterior distribution $p(\theta \mid x_o)$, exposing multimodal or degenerate parameter regimes.
* **Certified Coverage (TARP):** Empirically proves that the computed credible intervals cover the ground truth at the stated confidence level.

---

## 2. Running SBI Calibration

```python
import numpy as np
from ewm_engine.calibration.sbi import SBICalibrationEngine
from ewm_engine.calibration.certification import evaluate_tarp_coverage

# 1. Define parameter search bounds (Priors)
prior_bounds = {
    "transit_delay": (1.0, 10.0),
    "demand_elasticity": (0.1, 2.0),
}

# 2. Instantiate Calibration Engine with Target World
engine = SBICalibrationEngine(world=world, priors=prior_bounds)

# 3. Fit Neural Posterior on Target Telemetry
posterior = engine.fit(
    observations=historical_telemetry_array,
    num_simulations=1000,
)

# 4. Evaluate Certified Coverage via TARP
tarp_score, is_certified = evaluate_tarp_coverage(
    posterior=posterior,
    test_simulations=200,
    alpha_threshold=0.05,
)

print(f"Calibration TARP Score: {tarp_score:.3f}")
print(f"Coverage Certified:     {is_certified}")
```

---

## 3. Key Takeaways
* **Amortized Execution:** Once trained, the posterior can infer parameters for new incoming daily telemetry in milliseconds without re-running thousands of simulations.
* **TARP Certification:** Prevents deploying overconfident models to the Board of Directors.
