"""Evaluation metrics, uncertainty estimation, and counterfactual scenario comparison."""

from __future__ import annotations

from ewm_engine.evaluation.comparison import ScenarioComparison, compare_scenarios
from ewm_engine.evaluation.conformal import (
    ACIState,
    AdaptiveConformalInference,
    ConformalEvaluationSummary,
    ConformalInterval,
    ConformalIntervalPredictor,
    UQParetoCandidate,
    compute_uq_pareto_front,
)
from ewm_engine.evaluation.metrics import (
    DistributionalEquityMetric,
    FinalResourceLevelMetric,
    HardViolationsMetric,
    Metric,
    TotalViolationsMetric,
)
from ewm_engine.evaluation.multiseed import (
    MultiSeedMetricSummary,
    MultiSeedReport,
    evaluate_multiseed,
)
from ewm_engine.evaluation.pareto import (
    ObjectiveDirection,
    ObjectiveSpec,
    ParetoFrontier,
    compute_pareto_frontier,
)
from ewm_engine.evaluation.uncertainty import (
    AdaptiveCalibrationResult,
    BootstrapConfidenceInterval,
    BootstrapDelta,
    CalibrationDiagnostic,
    PITCalibrationResult,
    ReliabilityBin,
    UncertaintyDistribution,
    compute_adaptive_calibration_error,
    compute_bootstrap_ci,
    compute_bootstrap_delta,
    compute_brier_score,
    compute_crps,
    compute_interval_coverage,
    compute_log_score,
    compute_mean_crps,
    compute_pit_calibration,
    compute_tail_metrics,
    summarize_distribution,
)

__all__ = [
    "ACIState",
    "AdaptiveCalibrationResult",
    "AdaptiveConformalInference",
    "BootstrapConfidenceInterval",
    "BootstrapDelta",
    "CalibrationDiagnostic",
    "ConformalEvaluationSummary",
    "ConformalInterval",
    "ConformalIntervalPredictor",
    "DistributionalEquityMetric",
    "FinalResourceLevelMetric",
    "HardViolationsMetric",
    "Metric",
    "MultiSeedMetricSummary",
    "MultiSeedReport",
    "ObjectiveDirection",
    "ObjectiveSpec",
    "PITCalibrationResult",
    "ParetoFrontier",
    "ReliabilityBin",
    "ScenarioComparison",
    "TotalViolationsMetric",
    "UQParetoCandidate",
    "UncertaintyDistribution",
    "compare_scenarios",
    "compute_adaptive_calibration_error",
    "compute_bootstrap_ci",
    "compute_bootstrap_delta",
    "compute_brier_score",
    "compute_crps",
    "compute_interval_coverage",
    "compute_log_score",
    "compute_mean_crps",
    "compute_pareto_frontier",
    "compute_pit_calibration",
    "compute_tail_metrics",
    "compute_uq_pareto_front",
    "evaluate_multiseed",
    "summarize_distribution",
]
