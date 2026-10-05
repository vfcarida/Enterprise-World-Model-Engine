# API Change Proposal (ACP-001): Decision-Grade Scenario Evaluation & Uncertainty Upgrade

- **Target Release:** v1.1.0
- **Author:** Principal ML/Decision Engineer
- **Status:** Implemented
- **Impact Level:** Non-breaking enhancement (Additive evaluation capabilities)
- **Related Spec:** `docs/specs/features/FEAT-001-evaluation-upgrade.md`
- **Related ADR:** `docs/adr/ADR-016-evaluation-upgrade-bootstrap-cis-and-pareto.md`
- **Acceptance Criteria:** AC-024 (Contract Compatibility), FEAT-001

---

## 1. Summary

This proposal enhances `ewm_engine.evaluation` to deliver decision-grade counterfactual scenario evaluation. It introduces non-parametric bootstrap confidence intervals on delta metrics, simulation significance testing under model stochasticity ($P_{model}$), calibration diagnostics (empirical interval coverage and Continuous Ranked Probability Score - CRPS), multi-objective Pareto frontier analysis, and user-supplied objective ranking with an epistemic "No Objective $\implies$ No Ranking" guarantee.

All additions are strictly additive, pure NumPy/stdlib (zero heavy dependencies), and maintain 100% backwards compatibility with `v1.0.0` callers and serialization schemas.

---

## 2. Motivation & Use Case

In `v1.0.0`, `compare_scenarios` reported summary point estimates and raw deltas. Decision-makers evaluating enterprise interventions face two critical challenges:
1. **Sampling Uncertainty**: Distinguishing genuine policy improvements from Monte Carlo variance across finite rollout sets.
2. **Conflicting Objectives**: Managing unavoidable operational trade-offs without arbitrary single-metric bias.

Furthermore, the engine's epistemic discipline forbids declaring an automated "winner" without explicit user preferences.

---

## 3. Proposed API Surface Diff

```python
# Before (v1.0.0)
def compare_scenarios(
    baseline: SimulationResult,
    candidates: Sequence[SimulationResult],
    metrics: Sequence[str] | None = None,
) -> ScenarioComparison: ...


# After (v1.1.0 - Additive optional parameters with backwards-compatible defaults)
def compare_scenarios(
    baseline: SimulationResult,
    candidates: Sequence[SimulationResult],
    metrics: Sequence[str] | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
    objectives: Sequence[ObjectiveSpec] | dict[str, str | ObjectiveDirection] | None = None,
    objective: Callable[[dict[str, float]], float] | None = None,
) -> ScenarioComparison: ...
```

### Affected Symbols in `ewm_engine.__all__`
- [x] **No change to `ewm_engine.__all__`**: Root public surface remains exactly the 22 canonical Stable symbols + `__version__`.
- New evaluation data models and functions are exported cleanly from `ewm_engine.evaluation`:
  - `BootstrapConfidenceInterval`, `BootstrapDelta`
  - `compute_bootstrap_ci`, `compute_bootstrap_delta`
  - `CalibrationDiagnostic`, `compute_interval_coverage`, `compute_crps`, `compute_tail_metrics`
  - `ObjectiveDirection`, `ObjectiveSpec`, `ParetoFrontier`, `compute_pareto_frontier`

### Serialized Shape Extensions in `ScenarioComparison.to_dict()`
All existing keys (`"baseline"`, `"scenarios"`, `"deltas_vs_baseline"`) are preserved:
- `"bootstrap_deltas"`: Dictionary of `BootstrapDelta` dicts per scenario per metric.
- `"pareto_frontier"`: Optional serialized `ParetoFrontier` (or `None`).
- `"rankings"`: Optional list of `{"scenario": name, "score": score}` (or `None` when `objective=None`).

---

## 4. Backwards Compatibility & Migration Strategy

- **Is this a breaking change?** No. Existing callers invoking `compare_scenarios(baseline, candidates)` or `compare_scenarios(baseline, candidates, metrics=["..."])` receive the exact same statistical distributions, summary tables, and dictionary keys with additive fields.
- **Deprecation lifecycle:** No deprecations.
- **Migration instructions:** No migration required. Callers seeking confidence intervals or multi-objective analysis can pass `objectives={...}` or `objective=Callable`.

---

## 5. Affected Files & Tests

- **Public API / Interface:** `src/ewm_engine/evaluation/__init__.py`
- **Implementation:**
  - `src/ewm_engine/evaluation/uncertainty.py`
  - `src/ewm_engine/evaluation/comparison.py`
  - `src/ewm_engine/evaluation/pareto.py`
- **Documentation:**
  - `docs/concepts/uncertainty.md`
  - `docs/concepts/evaluation.md`
  - `docs/adr/ADR-016-evaluation-upgrade-bootstrap-cis-and-pareto.md`
  - `docs/specs/features/FEAT-001-evaluation-upgrade.md`
- **Enforcing Tests:**
  - `tests/contract/test_api_compatibility.py`
  - `tests/contract/test_public_api.py`
  - `tests/unit/test_evaluation_upgrade.py` (20 tests)
  - `tests/examples/test_minimal_warehouse.py`

---

## 6. Review & Approval Checklist

- [x] Reviewed against `docs/stability-policy.md` invariants.
- [x] Verified parameter ordering: `baseline` and `candidates` remain positional arguments.
- [x] Zero external dependencies added to core (`numpy` only).
- [x] Epistemic guardrail enforced: `rankings` is `None` unless user supplies `objective` callable.
- [x] Contract test `tests/contract/test_api_compatibility.py` passing with 0 warnings.
