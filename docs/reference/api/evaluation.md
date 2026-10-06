# Scenario Evaluation API Reference

This module provides decision-grade evaluation: counterfactual scenario comparisons, non-parametric percentile bootstrap confidence intervals, and multi-objective Pareto analysis.

---

## Scenario Comparison

::: ewm_engine.evaluation.comparison
    options:
      show_root_heading: true
      show_source: false
      members:
        - ScenarioComparison
        - compare_scenarios

---

## Multi-Objective Pareto Frontiers

::: ewm_engine.evaluation.pareto
    options:
      show_root_heading: true
      show_source: false
      members:
        - ParetoFrontier
        - ObjectiveSpec
        - ObjectiveDirection
        - compute_pareto_frontier

---

## Uncertainty & Bootstrap Intervals

::: ewm_engine.evaluation.uncertainty
    options:
      show_root_heading: true
      show_source: false
      members:
        - BootstrapConfidenceInterval
        - BootstrapDelta
        - compute_bootstrap_ci
        - compute_bootstrap_delta
        - compute_crps
