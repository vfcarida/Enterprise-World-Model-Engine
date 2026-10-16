# Recipe: How Do I Calibrate My World Model to Historical Telemetry?

> **Question:** *We have observed operational event logs or timeseries metrics. How do we tune our simulator's structural parameters so simulation outputs reliably match reality with certified coverage?*

---

## The Scenario

A structural simulator contains latent parameters $\theta$ (e.g., service rates, transit delays, demand elasticity). Rather than guessing point values or running ad-hoc grid search, EWM Engine uses **Amortized Simulation-Based Inference (SBI)** to recover the full posterior distribution:

$$p(\theta \mid x_{\text{obs}})$$

Crucially, EWM Engine enforces **mandatory coverage certification** via TARP (Tests of Accuracy and Relative Precision, [arXiv:2302.03026](https://arxiv.org/abs/2302.03026)). If the posterior is overconfident or under-covered, the engine refuses decision use.

---

## The Recipe Code

```python
import numpy as np
from ewm_engine.calibration.certification import certify_coverage
from ewm_engine.calibration.sbi import AmortizedPosteriorEstimator

# Step 1: Generate synthetic training samples from prior (theta -> x)
np.random.seed(42)
n_samples = 300
# Prior: parameter theta ~ Uniform(0.5, 2.5)
thetas = np.random.uniform(0.5, 2.5, size=(n_samples, 1))
# Forward simulator: observation x = 2.0 * theta + noise
xs = 2.0 * thetas + np.random.normal(0.0, 0.2, size=(n_samples, 1))

# Step 2: Fit amortized posterior estimator p(theta | x)
estimator = AmortizedPosteriorEstimator(n_neighbors=15)
estimator.fit(thetas, xs)

# Step 3: Run TARP coverage certification on holdout test set
test_thetas = np.random.uniform(0.5, 2.5, size=(60, 1))
test_xs = 2.0 * test_thetas + np.random.normal(0.0, 0.2, size=(60, 1))

certificate = certify_coverage(
    test_thetas=test_thetas,
    test_xs=test_xs,
    estimator=estimator,
    nominal_alphas=[0.1, 0.3, 0.5, 0.7, 0.9],
)

# Step 4: Validate certification gate
print(f"Certification Status: {certificate.status}")
print(f"Decision Usable:      {certificate.decision_usable}")
print(f"TARP Max Discrepancy: {certificate.tarp_max_discrepancy:.3f}")

# Epistemic Gate: If uncertified, this call raises a structured TeachingError
certificate.raise_for_status()
```

---

## Verification & Audit Guarantees

1. **Certified Coverage:**  
   The posterior credible intervals are empirically proven to cover the true parameters at nominal rates (e.g., a 90% interval contains the ground truth $\ge 90\%$ of the time).
2. **Honesty-as-DX Guardrail:**  
   If the simulator is structurally misspecified (i.e. observed data cannot be generated under any prior), `certificate.raise_for_status()` triggers `UncertifiedCalibrationError`, teaching the engineer why it refused rather than silently outputting collapsed intervals.
