"""Evaluation metrics, uncertainty estimation, and counterfactual scenario comparison."""

from __future__ import annotations

from ewm_engine.evaluation.comparison import ScenarioComparison, compare_scenarios
from ewm_engine.evaluation.metrics import (
    DistributionalEquityMetric,
    FinalResourceLevelMetric,
    HardViolationsMetric,
    Metric,
    TotalViolationsMetric,
)
from ewm_engine.evaluation.pareto import (
    ObjectiveDirection,
    ObjectiveSpec,
    ParetoFrontier,
    compute_pareto_frontier,
)
from ewm_engine.evaluation.uncertainty import (
    BootstrapConfidenceInterval,
    BootstrapDelta,
    CalibrationDiagnostic,
    UncertaintyDistribution,
    compute_bootstrap_ci,
    compute_bootstrap_delta,
    compute_crps,
    compute_interval_coverage,
    compute_tail_metrics,
    summarize_distribution,
)

__all__ = [
    "BootstrapConfidenceInterval",
    "BootstrapDelta",
    "CalibrationDiagnostic",
    "DistributionalEquityMetric",
    "FinalResourceLevelMetric",
    "HardViolationsMetric",
    "Metric",
    "ObjectiveDirection",
    "ObjectiveSpec",
    "ParetoFrontier",
    "ScenarioComparison",
    "TotalViolationsMetric",
    "UncertaintyDistribution",
    "compare_scenarios",
    "compute_bootstrap_ci",
    "compute_bootstrap_delta",
    "compute_crps",
    "compute_interval_coverage",
    "compute_pareto_frontier",
    "compute_tail_metrics",
    "summarize_distribution",
]
