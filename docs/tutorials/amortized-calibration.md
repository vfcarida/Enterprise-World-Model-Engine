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
from ewm_engine.calibration.certification import compute_tarp_coverage
from ewm_engine.calibration.priors import CalibrationPrior
from ewm_engine.calibration.sbi import AmortizedPosteriorEstimator, PriorSet

# 1. Define parameter search bounds (Priors)
priors = PriorSet(
    priors=(
        CalibrationPrior(name="transit_delay", distribution="uniform", low=1.0, high=10.0),
        CalibrationPrior(name="demand_elasticity", distribution="uniform", low=0.1, high=2.0),
    )
)

# 2. Instantiate and fit Amortized Posterior Estimator
estimator = AmortizedPosteriorEstimator(prior_set=priors)
estimator.fit(thetas=train_thetas, xs=train_xs)

# 3. Evaluate Certified Coverage via TARP
alphas, coverages, ece, max_disc = compute_tarp_coverage(
    test_thetas=test_thetas,
    test_xs=test_xs,
    estimator=estimator,
)

print(f"Calibration TARP ECE: {ece:.3f} | Max Discrepancy: {max_disc:.3f}")
```

---

## 3. Key Takeaways
* **Amortized Execution:** Once trained, the posterior can infer parameters for new incoming daily telemetry in milliseconds without re-running thousands of simulations.
* **TARP Certification:** Prevents deploying overconfident models to the Board of Directors.
