# ADR-016: Evaluation Upgrade — Bootstrap Confidence Intervals, Significance Flags, and Pareto Frontiers

## Status
Accepted

## Date
2026-10-05

## Context
The Enterprise World Model Engine simulates complex, stochastic socio-technical systems. While `v1.0.0` established deterministic Monte Carlo simulation and summary quantile reporting (`p05, p50, p95`), counterfactual evaluation (`compare_scenarios`) lacked:
1. **Sample uncertainty quantification on counterfactual deltas**: Delta estimates between scenarios ($\Delta = \mu_{cand} - \mu_{base}$) did not communicate sampling variation or confidence intervals.
2. **Statistical significance testing under simulation variance**: Users could not distinguish true systematic policy improvements from stochastic noise.
3. **Multi-objective optimization & Pareto trade-offs**: Real-world interventions require balancing conflicting objectives (e.g. operational cost vs. service reliability vs. constraint violations). Point rankings inevitably distort decision quality.
4. **Epistemic humility**: Automated winner selection without explicit preference functions produces misleading recommendations that violate the engine's core design philosophy ("no fake certainty, no unsubstantiated causal claims").

## Decision

### 1. Non-Parametric Bootstrap Confidence Intervals
We implement non-parametric percentile bootstrap resampling to compute $(1-\alpha)$ confidence intervals for counterfactual deltas:
$$\Delta^*_b = \bar{X}^*_{cand, b} - \bar{X}^*_{base, b}, \quad b = 1, \dots, B$$
This avoids unrealistic Gaussian assumptions over non-normal, skewed, or multimodal socio-technical rollout distributions.
A deterministic `np.random.Generator` seeded via optional parameter ensures bitwise reproducibility of evaluation reports.

### 2. Empirical Hypothesis Testing & Epistemic Guardrail
We compute empirical two-tailed bootstrap p-values and an `is_significant` boolean flag indicating whether the confidence interval strictly excludes zero.
**Epistemic Guardrail:** Documentation and docstrings explicitly declare that significance measures *simulation model variance* ($P_{model}$), not empirical real-world causal proof ($P(Y \mid \text{do}(X))$).

### 3. First-Class Multi-Objective Pareto Analysis
We introduce `ObjectiveSpec`, `ObjectiveDirection` (`MINIMIZE`, `MAXIMIZE`), and `ParetoFrontier`:
- Identifies the non-dominated Pareto frontier across candidate scenarios and baseline.
- Tracks exact dominance relations (`dominates`, `dominated_by`).
- **No Auto-Winner Rule:** If multiple non-dominated scenarios exist and no explicit scalarization weights are supplied, the engine returns the non-dominated set and explicitly states that decision-makers must weigh trade-offs.

### 4. Zero-Breakage Contract Compatibility
- `compare_scenarios` signature preserves `baseline` and `candidates` as first arguments with default values for new parameters (`confidence_level=0.95`, `n_bootstrap=1000`, `seed=None`, `objectives=None`).
- Existing dictionary keys in `ScenarioComparison.to_dict()` (`"baseline"`, `"scenarios"`, `"deltas_vs_baseline"`) are preserved, with `"bootstrap_deltas"` and `"pareto_frontier"` added additively.
- The 22 Stable root symbols in `ewm_engine.__all__` are unchanged, adhering strictly to ADR-015 and AC-024. New evaluation data models are exported from `ewm_engine.evaluation`.

## Consequences

### Positive
- Decision-makers receive scientifically grounded estimates of policy impact with rigorous bounds.
- Multi-criteria trade-offs are made transparent without artificial winner bias.
- Retains 100% backward compatibility with existing tests, documentation, and user code.

### Negative / Trade-offs
- Bootstrap computation introduces a modest CPU overhead for large sample sizes ($B = 1000$ iterations over $N$ rollouts), mitigated by vectorized NumPy operations.
