"""Financial World Model (FWM) vertical package.

Combines Market-1T 1Hz microstructure + SEC EDGAR bitemporal ingestion,
Vision Transformer JEPA state encoding with VICReg regularization,
CT-TGNN (EvolveGCN) contagion propagation conditioned on CausalInterventionVectors,
MLP financial risk decoding, and multi-regime comparative backtesting.
"""

from __future__ import annotations

from ewm_engine.financial.data import (
    FUNDAMENTAL_METRICS,
    REQUIRED_MARKET_COLUMNS,
    BitemporalSynchronizer,
    EDGARLoader,
    Market1TLoader,
)
from ewm_engine.financial.evaluation import (
    ARIMABaseline,
    BaselinePrediction,
    EvaluationMetrics,
    FinancialBacktester,
    FullFWMBaseline,
    GARCHBaseline,
    PredictOnlyJEPABaseline,
    RegimeEpisode,
    XGBoostBaseline,
    XGBoostEngineeredBaseline,
    build_historical_regime_episode,
    compute_coverage_calibration,
    compute_mae,
    compute_rank_ic,
)
from ewm_engine.financial.models import (
    CausalInterventionEmbedder,
    FinancialJEPAEncoder,
    FinancialJEPAPair,
    FinancialMetricsDecoder,
    FinancialPredictionOutput,
    FinancialTransitionPredictor,
    FinancialWorldModel,
    VICRegLoss,
    VICRegLossComponents,
    augment_cross_stock_mask,
    augment_time_warp,
    compute_rankme,
    sigreg_loss,
)

__all__ = [
    "FUNDAMENTAL_METRICS",
    "REQUIRED_MARKET_COLUMNS",
    "ARIMABaseline",
    "BaselinePrediction",
    "BitemporalSynchronizer",
    "CausalInterventionEmbedder",
    "EDGARLoader",
    "EvaluationMetrics",
    "FinancialBacktester",
    "FinancialJEPAEncoder",
    "FinancialJEPAPair",
    "FinancialMetricsDecoder",
    "FinancialPredictionOutput",
    "FinancialTransitionPredictor",
    "FinancialWorldModel",
    "FullFWMBaseline",
    "GARCHBaseline",
    "Market1TLoader",
    "PredictOnlyJEPABaseline",
    "RegimeEpisode",
    "VICRegLoss",
    "VICRegLossComponents",
    "XGBoostBaseline",
    "XGBoostEngineeredBaseline",
    "augment_cross_stock_mask",
    "augment_time_warp",
    "build_historical_regime_episode",
    "compute_coverage_calibration",
    "compute_mae",
    "compute_rank_ic",
    "compute_rankme",
    "sigreg_loss",
]
