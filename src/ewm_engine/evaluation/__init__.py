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
from ewm_engine.evaluation.uncertainty import (
    UncertaintyDistribution,
    summarize_distribution,
)

__all__ = [
    "DistributionalEquityMetric",
    "FinalResourceLevelMetric",
    "HardViolationsMetric",
    "Metric",
    "ScenarioComparison",
    "TotalViolationsMetric",
    "UncertaintyDistribution",
    "compare_scenarios",
    "summarize_distribution",
]
