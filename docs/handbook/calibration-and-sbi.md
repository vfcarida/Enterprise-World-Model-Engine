# Simulation Calibration & SBI: "How Do We Calibrate Parameters Honestly?"

> **Core Question**: *"Given real-world observational telemetry and an uncalibrated organizational simulator, how do we infer the Bayesian posterior distribution over simulator parameters with certified coverage, without assuming Gaussian error or intractable likelihoods?"*

---

## 1. Mathematical & Theoretical Formulation

### Likelihood-Free Simulation-Based Inference (SBI)
Complex enterprise simulators (discrete-event queues, multi-tier supply networks) do not possess a closed-form likelihood $p(x \mid \theta)$. Instead, they act as **implicit generative models**: they can sample forward trajectories $x \sim p(x \mid \theta)$, but computing the probability density of an observed trajectory $x_o$ is intractable.

Traditional Approximate Bayesian Computation (ABC) requires rejection sampling, which scales poorly when parameter dimensionality $D > 4$.

EWM implements **Simulation-Based Inference (SBI)** via amortized Neural Posterior Estimation (NPE) and Sequential Neural Posterior Estimation (SNPE) (Papamakarios et al., 2019; Cranmer et al., 2020). We learn a density estimator $q_\phi(\theta \mid x)$ minimizing the negative log-likelihood:

$$\mathcal{L}(\phi) = - \mathbb{E}_{\theta \sim p(\theta), x \sim p(x \mid \theta)} \left[ \log q_\phi(\theta \mid x) \right]$$

### Certified Coverage: The TARP Diagnostic
A calibrated Bayesian posterior must be neither overconfident nor underconfident. EWM evaluates calibration quality using the **Test of Accuracy of Related Probabilities (TARP)** (Lemos et al., 2023).

For any credible region level $\alpha \in [0, 1]$, the probability that the true parameter $\theta^*$ falls within the estimated $\alpha$-credible region must equal $\alpha$:

$$P(\theta^* \in \mathcal{R}_\alpha(x)) = \alpha$$

If the empirical coverage curve deviates significantly from the diagonal, the calibration is uncertified and must not be used for high-stakes decisions.

```
       TARP Expected vs. Observed Credible Coverage Diagnostic
  Observed
  Coverage ▲
    1.0 ┼                                        / (Certified Calibration)
        │                                      /
        │                           . - - - -
        │                         /   (Overconfident: Danger!)
        │                       /
    0.0 ┼──────────────────────/─────────────────────────► Expected Coverage
        0.0                                          1.0
```

---

## 2. Practical Usage & API Pattern

```bash
# Calibrate simulator parameters from real telemetry using CLI
ewm calibrate --spec world.yaml --data telemetry.csv --samples 5000 --out posterior.json
```

```python
from ewm_engine.calibration.sbi import SBICalibrationEngine
from ewm_engine.calibration.certification import evaluate_tarp_coverage

# 1. Define Parameter Prior Envelope and Target Observations
prior_bounds = {
    "warehouse_transit_delay": (1.0, 7.0),
    "supplier_elasticity": (0.2, 1.8),
}

# 2. Run Amortized Simulation-Based Inference
engine = SBICalibrationEngine(world=world, priors=prior_bounds)
posterior = engine.fit(
    observations=real_world_telemetry_df,
    num_simulations=2000,
)

# 3. Certify Calibration Coverage using TARP
tarp_score, is_certified = evaluate_tarp_coverage(
    posterior=posterior,
    test_simulations=500,
    alpha_threshold=0.05,
)

print(f"Calibration TARP Score: {tarp_score:.3f}")
print(f"Is Posterior Certified? {is_certified}")
```

---

## 3. Implementation & Computational Notes

* **Simulation-Based Calibration (SBC) Rank Statistics:** EWM computes SBC rank histograms $\text{Rank}(\theta^*) = \sum_{l=1}^L \mathbb{I}(\theta^{(l)} < \theta^*)$. Uniformity of ranks across parameters is formally tested using Kolmogorov-Smirnov and $\chi^2$ goodness-of-fit tests.
* **Lightweight CPU Architecture:** SBI calibration runs purely on CPU using vectorized NumPy sampling, or delegates to PyTorch when optional deep neural density estimators are installed (`pip install ewm-engine[calibration]`).
* **Prior Envelope Enforcement:** Parameters sampled during calibration cannot violate hard physical invariants. If a prior draw creates an invalid state, it is immediately discarded before neural fitting.

---

## 4. Common Pitfalls & Epistemic Traps

> [!WARNING]
> **Pitfall 1: Prior Misspecification (The Truncated Truth Trap)**  
> If reality operates at a parameter value $\theta = 12.5$, but your prior bounds are set to $[0.0, 10.0]$, SBI will concentrate all posterior mass at the boundary ($10.0$) while claiming high confidence. Always inspect prior sensitivity curves.

> [!CAUTION]
> **Pitfall 2: Confusing Calibration with Physical Correctness**  
> A simulator can be perfectly calibrated to observed telemetry while harboring completely incorrect structural mechanisms (equifinality). Calibration ensures parameters match the data; only causal intervention tests verify structural mechanisms.

> [!TIP]
> **Best Practice: Never Deploy Without Passing the TARP Gate**  
> If the empirical TARP curve bows significantly below the diagonal, the posterior is overconfident. In high-stakes settings, deploy [`WideUncertaintyWarning`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/core/trust.py) and widen confidence intervals to avoid underestimating risk.
