# Scientific Evaluation & Benchmarking Guide (`EWM-006`)

## 1. Overview & Purpose

To prevent claiming world-model effectiveness solely from unchallenged self-designed rollouts, `EWM-006` provides a reproducible **Scientific Evaluation Framework** ([`EWM_Evaluator`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/evaluator.py)) that benchmarks the **Enterprise World Model (`EWMPolicyAgent`)** head-to-head against naive and heuristic baselines under identical, cryptographically derived exogenous noise streams (Common Random Numbers).

The framework adheres to the statistical evaluation methodology of **Agarwal et al. (2021)** (*"Deep Reinforcement Learning at the Edge of the Statistical Precipice"*, NeurIPS 2021).

---

## 2. Evaluated Agents

| Agent Class | Paradigm | Decision & Prediction Mechanism |
| :--- | :--- | :--- |
| [`RandomAgent`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/evaluator.py) | `random_walk` | Samples uniform replenishment actions across the raw action space and predicts next-step state via an unconstrained Gaussian random walk $\hat{S}_{t+1} = S_t + \mathcal{N}(0, \sigma^2)$. |
| [`StaticRuleAgent`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/evaluator.py) | `static_rule` | Executes fixed threshold heuristics (`if inventory < threshold: order fixed_qty`) and predicts next-step state assuming static pre-shift demand without regime-shift or liquidity-floor awareness. |
| [`EWMPolicyAgent`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/evaluator.py) | `ewm_hybrid` | Evaluates candidate actions across isolated counterfactual branches (`engine.branch().do()`), enforces hard `Pre-rules -> ML -> Post-rules` via `HybridTransitionRouter`, and combines structural conservation laws with calibrated residual predictions. |

---

## 3. Evaluation Metrics

Every seed rollout records three primary orthogonal dimensions plus task return:

1. **Predictive Accuracy (`predictive_accuracy` $\in [0, 1]$ & `prediction_mae`)**:
   $$\text{PredictiveAccuracy} = \max\left(0,\; 1 - \frac{1}{T}\sum_{t=1}^{T} \frac{1}{K}\sum_{k=1}^{K}\frac{|\hat{y}_{t,k} - y_{t,k}|}{R_k^{\max} - R_k^{\min}}\right)$$
2. **Constraint Violation Rate (`constraint_violation_rate` $\in [0, 1]$)**:
   $$\text{ConstraintViolationRate} = \frac{1}{T}\sum_{t=1}^{T} \mathbb{I}\!\left(\text{state or action at step } t \text{ violates hard structural constraints}\right)$$
   - **Contract**: `EWMPolicyAgent` achieves **`0.0000` (`0%`)** Constraint Violation Rate across all seeds and horizons.
3. **Execution Time (`execution_time_ms`)**:
   - High-resolution wall-clock duration (`time.perf_counter()`) in milliseconds per rollout seed.

---

## 4. Statistical Rigor & $N \ge 30$ Significance Guard

Following **Agarwal et al. (2021)**:
- **Sample Standard Deviation (`std`)**: Reported with Bessel's correction ($\text{ddof}=1$ for $N > 1$) across independent seeds spawned via `numpy.random.SeedSequence`.
- **Interquartile Mean (`iqm`)**: 25% trimmed mean discarding the bottom 25% and top 25% of seed scores to eliminate outlier sensitivity.
- **95% Bootstrap Confidence Intervals (`ci_lower_95`, `ci_upper_95`)**: Non-parametric bootstrap intervals over $N$ seeds.
- **Probability of Improvement ($P(\text{EWM} > \text{Baseline})$)**: Mann-Whitney $U$ statistic normalized by $N_{\text{EWM}} \times N_{\text{Base}}$.
- **Strict $N < 30$ Significance Guard (`MIN_SIGNIFICANCE_SEEDS = 30`)**:
  - When $N < 30$, `is_statistically_significant` is strictly forced to `False` and `significance_status` is set to `"INSUFFICIENT_SEEDS_N_LT_30"`.
  - Statistical significance (`"SIGNIFICANT_AT_ALPHA"`) is only evaluated when $N \ge 30$ independent seeds are executed.

---

## 5. Operational Timeout Safeguard

`EWM_Evaluator` enforces a hard wall-clock deadline (`timeout_seconds`, default `30.0s`) checked before every seed and every simulation step. If execution exceeds `timeout_seconds`, [`BenchmarkTimeoutError`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/evaluator.py) is raised immediately to prevent hanging CI pipelines.

---

## 6. Reproducing the Scientific Claims

### Run the Verification Test Suite

```bash
pytest tests/core/test_scientific_evaluation_framework.py -v
```

### Generate Reproducible JSON and CSV Benchmark Artifacts ($N = 30$ Seeds, 10-Step Problem)

```python
from pathlib import Path
from ewm_engine.evaluation.evaluator import EWM_Evaluator, EnvironmentSpec

evaluator = EWM_Evaluator(
    env_spec=EnvironmentSpec(horizon_steps=10),
    timeout_seconds=30.0,
)
report = evaluator.evaluate(n_seeds=30, master_seed=42)
json_path, csv_path = evaluator.save_artifacts(
    report,
    output_dir=Path("artifacts/benchmarks"),
    prefix="ewm_006_10step_n30",
)
```
