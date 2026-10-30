"""Financial World Model evaluation, backtesting, and comparative baselines package."""

from __future__ import annotations

from ewm_engine.financial.evaluation.backtester import (
    STANDARD_UNIVERSE,
    EvaluationMetrics,
    FinancialBacktester,
    build_historical_regime_episode,
    compute_coverage_calibration,
    compute_mae,
    compute_rank_ic,
)
from ewm_engine.financial.evaluation.baselines import (
    ARIMABaseline,
    BaselinePrediction,
    FullFWMBaseline,
    GARCHBaseline,
    PredictOnlyJEPABaseline,
    RegimeEpisode,
    XGBoostBaseline,
    XGBoostEngineeredBaseline,
)

__all__ = [
    "STANDARD_UNIVERSE",
    "ARIMABaseline",
    "BaselinePrediction",
    "EvaluationMetrics",
    "FinancialBacktester",
    "FullFWMBaseline",
    "GARCHBaseline",
    "PredictOnlyJEPABaseline",
    "RegimeEpisode",
    "XGBoostBaseline",
    "XGBoostEngineeredBaseline",
    "build_historical_regime_episode",
    "compute_coverage_calibration",
    "compute_mae",
    "compute_rank_ic",
]
