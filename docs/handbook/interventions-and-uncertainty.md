# Interventions & Uncertainty: "Will It Help, by How Much, and How Sure Are We?"

> **Core Question**: *"If we intervene with action $a$ instead of baseline $a'$, what is the expected difference in organizational outcome $Y$, what is the certified confidence interval, and does the evidence meet anytime-valid testing thresholds?"*

---

## 1. Mathematical & Theoretical Formulation

### The Interventional Distribution and Treatment Effect
In observational data, observing an action $A=a$ gives the conditional distribution $P(Y \mid A=a)$. Under Judea Pearl’s causal framework, enacting an active organizational intervention severs the natural structural equations governing $A$, replacing them with a fixed assignment $do(A=a)$:

$$P(Y \mid do(A=a)) \neq P(Y \mid A=a)$$

The Average Treatment Effect ($\text{ATE}$) between a proposed intervention $a$ and a baseline policy $a'$ is defined over the interventional distributions:

$$\tau(a, a') = \mathbb{E}[Y \mid do(A=a)] - \mathbb{E}[Y \mid do(A=a')]$$

### Non-Parametric Bootstrap Confidence Intervals (`percentile`, `basic`, `normal`, `studentized`, `bca`)
In socio-technical systems, outcome distributions $Y$ are frequently skewed, heavy-tailed, or multi-modal (e.g., supply chain stockouts, insurance claim losses). No single bootstrap interval is universally optimal across all sample sizes and skewness regimes, so EWM provides five explicitly named strategies in `compute_bootstrap_ci`, `compute_bootstrap_delta`, and `compare_scenarios`:

* `"percentile"` (default): $[\hat{\tau}^*_{(\alpha/2)}, \hat{\tau}^*_{(1-\alpha/2)}]$.
* `"basic"`: pivotal reflection $[2\hat{\tau} - \hat{\tau}^*_{(1-\alpha/2)}, 2\hat{\tau} - \hat{\tau}^*_{(\alpha/2)}]$.
* `"normal"`: first-order asymptotic normal interval $\hat{\tau} \pm z_{1-\alpha/2}\widehat{\text{SE}}_{\text{boot}}$.
* `"studentized"`: bootstrap-$t$ interval using per-resample plug-in standard errors.
* `"bca"`: **Bias-Corrected and Accelerated (BCa) bootstrap** (Efron, 1987), adjusting quantile endpoints for median bias $\hat{z}_0$ and jackknife skewness acceleration $\hat{a}$:

$$\alpha_1 = \Phi\left( \hat{z}_0 + \frac{\hat{z}_0 + z_{\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{\alpha/2})} \right), \quad \alpha_2 = \Phi\left( \hat{z}_0 + \frac{\hat{z}_0 + z_{1-\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{1-\alpha/2})} \right)$$

Furthermore, when baseline and candidate scenarios share coupled trajectory seeds (common random numbers), setting `paired=True` resamples paired trajectory differences $d_i = Y_{\text{cand}}(U_i) - Y_{\text{base}}(U_i)$, eliminating shared exogenous variance.

### Safe-Testing e-Values vs. Confounding Sensitivity E-Values
Under classical hypothesis testing (p-values), continuously monitoring results and stopping early inflates the false-positive rate. EWM provides **safe-testing e-processes** (`SafeTestingEProcess`, `compute_safe_e_value`; Grünwald et al., 2024; Ramdas et al., 2023):

$$\mathbb{E}_{H_0}[e_t \mid \mathcal{F}_{t-1}] \le 1$$

By Ville’s maximal inequality, for any filtration-adapted stopping time $\tau$:

$$\mathbb{P}_{H_0}\left( \exists t \ge 1 : E_t \ge \frac{1}{\alpha} \right) \le \alpha$$

A safe-testing e-process exceeding $E_t \ge 20$ certifies a Type-I error rate $\alpha \le 0.05$ under optional stopping.

> [!IMPORTANT]
> **E-Value Disambiguation:**
> Never confuse a sequential **safe-testing e-value / e-process** (`SafeTestingEProcess`, $\mathbb{E}_{H_0}[e_t \mid \mathcal{F}_{t-1}] \le 1$) with VanderWeele & Ding's (2017) **confounding Sensitivity E-value** (`SensitivityEValueResult`, $E_{\text{sens}} = RR + \sqrt{RR(RR-1)} \ge 1$ in `ewm_engine.experimental.causal`), which measures the minimum risk-ratio association an unmeasured confounder would need to explain away an observational association.

---

## 2. Practical Usage & API Pattern

```python
from ewm_engine.core.actions import Action
from ewm_engine.evaluation.comparison import compare_scenarios
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario, ScheduledAction

# 1. Define Baseline Scenario vs. Proposed Strategic Intervention
scenario_base = Scenario(
    scenario_id="logistics-base",
    name="Baseline",
    horizon=10,
    samples=500,
    seed=101,
)

scenario_reroute = Scenario(
    scenario_id="logistics-reroute",
    name="Reroute Intervention",
    horizon=10,
    samples=500,
    seed=101,
    scheduled_actions=(
        ScheduledAction(
            step=2,
            action=Action(
                id="act-reroute",
                type="reroute_freight",
                parameters={"volume": 35.0, "source": "hub_south", "target": "hub_north"},
            ),
        ),
    ),
)

# 2. Execute Simulation via Seed-Isolated Engine
engine = SimulationEngine()
res_base = engine.run(world, scenario_base)
res_reroute = engine.run(world, scenario_reroute)

# 3. Evaluate Statistical Comparison under P_model (Paired CRN + BCa Bootstrap)
comparison = compare_scenarios(
    baseline=res_base,
    candidates=[res_reroute],
    confidence_level=0.95,
    paired=True,
    interval_method="bca",
    seed=42,
)

# Render formatted multi-objective comparison table (reports P_model estimand)
print(comparison.summary_table())
```

---

## 3. Implementation & Computational Notes

* **Vectorized Bootstrap Sampling:** EWM’s bootstrap engine ([`ewm_engine.evaluation.uncertainty`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/evaluation/uncertainty.py)) computes jackknife acceleration values $\hat{a}$ and studentized plug-in standard errors via vectorized NumPy operations.
* **Hermetic PRNG Seed Sequence:** When running 500 samples across multiple scenarios with the same root `seed`, both scenarios share coupled trajectory seeds so `paired=True` isolates the policy delta under common random numbers.
* **6-Part Uncertainty Taxonomy:** Every `UncertaintyDistribution` and `BootstrapDelta` carries an explicit `UncertaintyTaxonomyBreakdown` separating `aleatoric_std`, `parameter_std`, `structural_model_std`, `runtime_context_std`, `monte_carlo_standard_error`, and `model_discrepancy_vs_reality` (explicitly flagging unestimated sources such as `model_discrepancy_vs_reality` when external real-world telemetry is absent).

---

## 4. Common Pitfalls & Epistemic Traps

> [!WARNING]
> **Pitfall 1: Confusing Observational Difference with Interventional Gain**
> Looking at historical records and finding that "warehouses that shipped goods had lower stockouts" is an observational association ($P(Y \mid X)$). If those warehouses shipped goods because they had higher staffing, intervening to force shipments from an unstaffed warehouse will backfire. Always model interventions through `ScheduledAction` or `do()` with explicit causal identification provenance.

> [!CAUTION]
> **Pitfall 2: Treating a $P_{\text{model}}$ Confidence Interval as Real-World Truth ($P_{\text{real}}$)**
> A 95% bootstrap confidence interval from simulated trajectories estimates the in-simulator estimand $\mathbb{E}_{P_{\text{model}}}[Y_{\text{cand}}] - \mathbb{E}_{P_{\text{model}}}[Y_{\text{base}}]$. It measures **Monte Carlo sampling variance under the declared simulator**, *not* structural discrepancy against physical reality (`model_discrepancy_vs_reality`).

> [!TIP]
> **Best Practice: Use `SafeTestingEProcess` for Optional Stopping**
> When running sequential simulations or adaptive pilot rollouts where sample size is not fixed in advance, do not use fixed-horizon p-values to stop early. Use `SafeTestingEProcess` (`compute_safe_e_value`), which controls Type-I error under arbitrary filtration-adapted stopping rules via Ville's inequality.
