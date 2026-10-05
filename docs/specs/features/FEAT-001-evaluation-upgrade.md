# Feature Specification: FEAT-001 — Evaluation Upgrade

**Feature ID:** `FEAT-001`  
**Milestone:** `v1.1.0` (Prompt `P02`)  
**Status:** Approved  
**Related ADR:** [`docs/adr/ADR-016-evaluation-upgrade-bootstrap-cis-and-pareto.md`](../../adr/ADR-016-evaluation-upgrade-bootstrap-cis-and-pareto.md)  
**Governing Specification:** [`docs/specs/spec-driven-development.md`](../spec-driven-development.md)  

---

## 1. Problem Statement

In `v1.0.0`, `ewm_engine.evaluation.compare_scenarios` reported point estimates (mean, std) and fixed quantiles ($p05, p50, p95$) alongside raw absolute and percentage deltas ($\Delta = \mu_{cand} - \mu_{base}$).

While technically functional, this was insufficient for rigorous decision-making in socio-technical systems:
1. **No Uncertainty on Deltas**: A delta of $-10.0$ could represent a statistically decisive reduction in risk or mere Monte Carlo sampling noise depending on rollout sample size ($N$) and trajectory variance.
2. **Absence of Significance Testing**: Decision-makers could not determine if an observed improvement was statistically distinguishable from zero under the stochastic dynamics.
3. **No Multi-Objective / Pareto Analysis**: Enterprise interventions involve inherent trade-offs (e.g. inventory holding costs vs. stockout rates vs. constraint violations). Ranking scenarios by a single metric creates false optima.
4. **Anti-Overclaim Requirement**: Automated winner selection without user-specified weights violates epistemic humility; the engine must report the non-dominated Pareto frontier rather than declaring an arbitrary "best" scenario.

---

## 2. Requirements & Acceptance Criteria

### AC-P02-1: Bootstrap Confidence Intervals on Deltas
- `compare_scenarios` must compute non-parametric bootstrap confidence intervals (default: 95% CI) for the difference in means between candidate scenarios and the baseline for each evaluated metric.
- Computations must be deterministic given an optional seed.
- If $N < 2$, bootstrap gracefully falls back with clear indicators rather than crashing.

### AC-P02-2: Significance Flags & Epistemic Guardrails
- For each metric delta, the engine must compute an empirical p-value and a boolean `is_significant` flag indicating whether the confidence interval excludes zero at the chosen $\alpha = 1 - \text{confidence\_level}$.
- Documentation and docstrings must explicitly state that significance reflects *simulation stochastic variance* conditioned on model mechanics, NOT empirical real-world ground truth.

### AC-P02-3: Multi-Objective & Pareto Frontier Analysis
- Provide `ObjectiveDirection` (`MINIMIZE`, `MAXIMIZE`) and `ObjectiveSpec`.
- `compute_pareto_frontier` evaluates candidate scenarios (and baseline) across multiple specified objectives, partitioning scenarios into:
  - Non-dominated Pareto frontier (`frontier`).
  - Dominated scenarios with explicit dependency sets (`dominated_by`, `dominates`).
- When no weights or utility functions are supplied, the engine MUST refuse to declare an arbitrary single winner, returning the full non-dominated set.

### AC-P02-4: Backwards Compatibility (AC-024)
- Existing `compare_scenarios(baseline, candidates, metrics=...)` calls must remain 100% compatible.
- `ScenarioComparison.to_dict()` and `summary_table()` must preserve existing keys and structure while adding enriched bootstrap and Pareto sections.
- Core public API contract in `ewm_engine.__all__` remains invariant; new evaluation types are exported from `ewm_engine.evaluation`.

---

## 3. Data Contracts & Types

```python
class BootstrapDelta(BaseModel):
    absolute_delta: float
    percent_delta: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    p_value: float
    is_significant: bool
    n_resamples: int


class ObjectiveDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveSpec(BaseModel):
    metric_name: str
    direction: ObjectiveDirection = ObjectiveDirection.MINIMIZE
    weight: float | None = None


class ParetoFrontier(BaseModel):
    objectives: list[ObjectiveSpec]
    frontier: list[str]
    dominated_by: dict[str, list[str]]
    dominates: dict[str, list[str]]
    scores: dict[str, dict[str, float]]
    weighted_scores: dict[str, float] | None = None
```
