# When to Refuse a Claim: The Epistemic Honesty Guide

> *"The highest duty of a decision engine is not giving an answer; it is knowing when to refuse to answer. In high-stakes organizational planning, an ungrounded algorithmic recommendation is worse than honest silence."*

---

## Executive Overview: The EWM Identity

Most machine learning platforms and business intelligence tools are programmed to produce an answer regardless of input quality. If an executive feeds confounded historical data into a regression pipeline or an LLM prompt, the software emits a confident forecast without hesitation.

Enterprise World Model Engine (`ewm-engine`) operates under the opposite philosophy: **Epistemic Honesty**. 

When mathematical assumptions are violated, EWM **actively refuses** to rank scenarios, rejects causal promotion, and halts automated execution. This guide explains how to read and interpret EWM’s epistemic refusal mechanisms.

```
                         The Epistemic Gate Hierarchy
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     ▼                                 ▼                                 ▼
1. CAUSAL REFUSAL             2. MANDATORY UNCERTAINTY          3. EPISTEMIC ABSTENTION
   Blocks promotion from         Rejects bare point estimates;     Flags out-of-distribution
   association to intervention   enforces non-parametric bounds.   states beyond telemetry.
   without identifiability.            │                                 │
     │                                 ▼                                 ▼
     ▼                        `ScenarioForecastResult`          `WideUncertaintyWarning`
   `UnqualifiedCausalClaimError`
```

---

## 1. How to Read an Anytime-Valid E-Value

In traditional null hypothesis significance testing, researchers compute a p-value. However, in corporate decision pipelines where data arrives continuously:
* **The Peeking Problem:** If you inspect a p-value every week and stop the simulation as soon as $p < 0.05$, the true false-positive rate balloons to $30\%$ or higher.
* **The E-Value Solution:** An **E-value** is a non-negative random variable whose expectation under the null hypothesis $H_0$ is at most 1:
  $$\mathbb{E}_{H_0}[E] \le 1$$

### How to Interpret the E-Value Scorecard

| E-Value Metric | Statistical Meaning | Decision-Grade Interpretation |
| :--- | :--- | :--- |
| $E < 1.0$ | Evidence favors the null hypothesis (no effect). | The proposed policy shows no measurable advantage over status quo. |
| $1.0 \le E < 5.0$ | Weak anecdotal evidence. | Inconclusive. Do not commit capital. |
| $5.0 \le E < 20.0$ | Moderate evidence. | Promising signal, but fails rigorous regulatory significance thresholds. |
| $E \ge 20.0$ | Strong evidence ($1/\alpha = 20 \implies \alpha \le 0.05$). | **Decision-Grade Evidence.** Safe to reject $H_0$ even under optional stopping. |
| $E \ge 100.0$ | Overwhelming evidence ($\alpha \le 0.01$). | Certified significant effect. Robust to extreme scrutiny. |

> [!TIP]
> **Reading EWM Comparison Tables:**  
> When comparing scenarios via `compare_scenarios()`, any claim marked with an asterisk (`*sig`) is backed by both a BCa bootstrap confidence interval excluding zero and an anytime-valid E-value exceeding $20.0$.

---

## 2. Why Uncertainty Quantification is Mandatory

EWM’s public simulation API **strictly refuses** to return bare point estimates:

```python
# The following code fails at the type boundary by design:
# mean_val = result.final_point_estimate("inventory") -> AttributeError!

# The correct, enforced pattern:
forecast = result.scenario_forecast("inventory")
print(forecast.p50)  # Median
print(forecast.ci_lower)  # 90% BCa Lower Bound (e.g., p05)
print(forecast.ci_upper)  # 90% BCa Upper Bound (e.g., p95)
print(forecast.dispersion)  # Empirical Inter-Quartile Range
```

### Why Point Forecasts Are Corporate Liabilities
1. **The Flaw of Averages:** In non-linear systems, the expected outcome of an average shock does not equal the average outcome of all shocks ($\mathbb{E}[f(X)] \neq f(\mathbb{E}[X])$).
2. **Concealed Insolvency Tails:** A project with an expected return of $+12\%$ might harbor a $4\%$ probability of total liquidity wipeout. Point forecasts hide this catastrophe from the Board of Directors.
3. **Regulatory Inadmissibility:** Under the EU AI Act (Article 15) and financial supervisory standards (SR 11-7), high-stakes risk models that fail to disclose confidence intervals are legally non-compliant.

---

## 3. When `do()` is Non-Identifiable: Epistemic Refusal

In observational enterprise data, correlation is ubiquitous while genuine causal identification is rare. Under SPEC 9 and [ADR-033](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-033-scientific-ml-provenance-uq-and-causal-gates.md), EWM forbids auto-promoting an observational association to a causal claim.

### The Refusal Gate in Code
```python
from ewm_engine.core.trust import validate_causal_claim, UnqualifiedCausalClaimError
from ewm_engine.provenance.evidence import EvidenceLevel

try:
    # Attempting to declare an INTERVENTIONAL claim when unobserved confounders exist:
    level = validate_causal_claim(
        claimed_level=EvidenceLevel.INTERVENTIONAL,
        is_identifiable=False,  # Backdoor criterion violated
        overlap_satisfied=False,  # Positivity violated
        strict=True,
    )
except UnqualifiedCausalClaimError as exc:
    print(f"AUTOMATED REFUSAL ACTIVATED: {exc}")
```

### When Must the Engine Refuse?
1. **Open Backdoor Paths:** When an unobserved variable $U$ points to both the proposed action and the measured outcome, and cannot be blocked by conditioning on observed covariates.
2. **Positivity Violations (Zero Overlap):** When the historical data contains certain operating regimes where the proposed action was *never* observed (e.g., negative interest rates, 100% remote workforce).
3. **Ambiguous Structural Ordering:** When feedback loops prevent establishing a directed acyclic relationship without time-indexing.

---

## 4. What a "Counterfactual of the Model" Does and Does Not Mean

EWM supports exact Pearlian counterfactual surgery via Twin-Rollout noise coupling ([ADR-040](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-040-twin-rollout-counterfactual-scm-kernel.md)). Every counterfactual trajectory emitted by EWM automatically includes an explicit epistemic disclaimer:

> *"Model-counterfactual evaluated within the fully specified simulated SCM under Twin-Rollout noise coupling (arXiv:2402.10877). Epistemic disclaimer: this represents a counterfactual of the model, NOT a validated real-world counterfactual."*

### The Epistemic Boundary

| What a Model Counterfactual IS | What a Model Counterfactual IS NOT |
| :--- | :--- |
| An exact mathematical deduction of what the model would have done had action $a'$ been taken under identical realized exogenous noise $U$. | Empirical proof that the physical real world would have behaved this way. |
| A rigorous isolation of net treatment effect free of random sampling noise divergence. | An excuse to ignore unmodeled real-world dynamics or human behavioral adaptations. |
| A verifiable test of policy logic and conservation law preservation under historical shocks. | A legal guarantee of future performance. |

---

## 5. Epistemic Abstention: Out-of-Distribution Detection (RWM-U)

When a simulation trajectory enters regions of state space that lie outside historical empirical support, standard simulators continue forecasting with false confidence.

EWM implements the **Reliable World Model Uncertainty (RWM-U)** protocol ([ADR-044](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-044-causal-correctness-metamorphic-evaluation.md)):
1. **Epistemic Support Boundary:** Tracks whether the current state $\mathbf{s}_t$ falls within the convex hull or density envelope of calibrated historical data.
2. **Automated Down-Weighting in `ewm decide`:** When evaluating candidate policies, trajectories that venture into out-of-support states accumulate an epistemic penalty, preventing the optimizer from selecting fragile, ungrounded actions.
3. **Structured Warning Emissions:** Issues structured `WideUncertaintyWarning` and `ExtrapolationWarning` alerts that propagate directly to executive dashboards.

---

## Summary Checklist: The Epistemic Honesty Scorecard

Before signing off on any simulation-backed enterprise policy, verify:
* [ ] Is the claim backed by an E-value $E \ge 20$ or a BCa confidence interval that excludes zero?
* [ ] Does the report disclose the 5th percentile ($p05$) downside tail, not just the median?
* [ ] Did the proposed intervention pass the formal Backdoor Criterion identifiability check?
* [ ] Is the counterfactual recognized as a property of the model SCM rather than a real-world certainty?
* [ ] Did the rollout stay within the verified empirical support boundary without triggering RWM-U abstention?
