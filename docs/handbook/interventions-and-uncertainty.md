# Interventions & Uncertainty: "Will It Help, by How Much, and How Sure Are We?"

> **Core Question**: *"If we intervene with action $a$ instead of baseline $a'$, what is the expected difference in organizational outcome $Y$, what is the certified confidence interval, and does the evidence meet anytime-valid testing thresholds?"*

---

## 1. Mathematical & Theoretical Formulation

### The Interventional Distribution and Treatment Effect
In observational data, observing an action $A=a$ gives the conditional distribution $P(Y \mid A=a)$. Under Judea Pearl’s causal framework, enacting an active organizational intervention severs the natural structural equations governing $A$, replacing them with a fixed assignment $do(A=a)$:

$$P(Y \mid do(A=a)) \neq P(Y \mid A=a)$$

The Average Treatment Effect ($\text{ATE}$) between a proposed intervention $a$ and a baseline policy $a'$ is defined over the interventional distributions:

$$\tau(a, a') = \mathbb{E}[Y \mid do(A=a)] - \mathbb{E}[Y \mid do(A=a')]$$

### Non-Parametric BCa Bootstrap Confidence Intervals
In socio-technical systems, outcome distributions $Y$ are frequently skewed, heavy-tailed, or multi-modal (e.g., supply chain stockouts, insurance claim losses). Gaussian normality assumptions fail.

EWM implements the **Bias-Corrected and Accelerated (BCa) non-parametric bootstrap** (Efron, 1987). Given $B$ bootstrap replications of the treatment effect $\hat{\tau}^{*b}$:

$$CI_{1-\alpha} = \left[ \hat{\tau}^{*(\alpha_1)}, \hat{\tau}^{*(\alpha_2)} \right]$$

where percentiles $\alpha_1, \alpha_2$ are adjusted for median bias $\hat{z}_0$ and skewness acceleration $\hat{a}$:

$$\alpha_1 = \Phi\left( \hat{z}_0 + \frac{\hat{z}_0 + z_{\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{\alpha/2})} \right), \quad \alpha_2 = \Phi\left( \hat{z}_0 + \frac{\hat{z}_0 + z_{1-\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{1-\alpha/2})} \right)$$

### Anytime-Valid Sequential E-Values
Under classical hypothesis testing (p-values), continuously monitoring results and stopping early inflates the false-positive rate. EWM evaluates scenario comparisons using **E-values** (Grünwald et al., 2024):

$$\mathbb{E}_{H_0}[E] \le 1$$

By Ville’s inequality, for any stopping time $\tau$:

$$P_{H_0}\left( \sup_{t} E_t \ge \frac{1}{\alpha} \right) \le \alpha$$

An E-value of $E \ge 20$ certifies a Type-I error rate $\alpha \le 0.05$ under optional stopping.

---

## 2. Practical Usage & API Pattern

```python
from ewm_engine.core.actions import ScheduledAction
from ewm_engine.evaluation.comparison import compare_scenarios
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario

# 1. Define Baseline Scenario vs. Proposed Strategic Intervention
scenario_base = Scenario(
    scenario_id="logistics-base",
    horizon=10,
    samples=500,
    seed=101,
)

scenario_reroute = Scenario(
    scenario_id="logistics-reroute",
    horizon=10,
    samples=500,
    seed=101,
    actions=[
        ScheduledAction(
            step=2,
            action_type="reroute_freight",
            payload={"volume": 35.0, "source": "hub_south", "target": "hub_north"},
        )
    ],
)

# 2. Execute Simulation via Seed-Isolated Engine
engine = SimulationEngine()
res_base = engine.run(world, scenario_base)
res_reroute = engine.run(world, scenario_reroute)

# 3. Evaluate Statistical Comparison with BCa Intervals and Significance
comparison = compare_scenarios(
    baseline_result=res_base,
    scenario_results=[res_reroute],
    confidence_level=0.95,
)

# Render formatted multi-objective comparison table
print(comparison.format_text_table())
```

---

## 3. Implementation & Computational Notes

* **Vectorized Bootstrap Sampling:** To maintain high performance without large external dependencies, EWM’s BCa engine ([`ewm_engine.evaluation.uncertainty`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/uncertainty.py)) computes jackknife influence values $\hat{a}$ via vectorized matrix leave-one-out operations in NumPy.
* **Hermetic PRNG Seed Sequence:** When running 500 samples across multiple scenarios, both scenarios share the identical root `SeedSequence` entropy to guarantee that stochastic customer arrivals in sample $k$ are coupled between baseline and intervention.
* **Memory Management:** Trajectories store time-series metrics in contiguous NumPy arrays; computing $p05, p50, p95$ quantiles does not copy graph topologies across steps.

---

## 4. Common Pitfalls & Epistemic Traps

> [!WARNING]
> **Pitfall 1: Confusing Observational Difference with Interventional Gain**  
> Looking at historical records and finding that "warehouses that shipped goods had lower stockouts" is an observational association. If those warehouses shipped goods because they had higher staffing, intervening to force shipments from an unstaffed warehouse will backfire. Always model interventions through `ScheduledAction` or `do()`.

> [!CAUTION]
> **Pitfall 2: Treating a 95% Confidence Interval as Real-World Truth**  
> A 95% BCa confidence interval measures **sampling variance given the model specifications**. It does *not* account for structural misspecification. If your simulation ignores supplier lead-time elasticity, the confidence interval is precisely measuring the wrong distribution.

> [!TIP]
> **Best Practice: Monitor E-Values Under Adaptive Experimentation**  
> When running batch simulations or live pilot rollouts, do not use classical p-values to stop experiments early. Use EWM's sequential E-values, which remain valid regardless of when the decision-maker inspects the data.
