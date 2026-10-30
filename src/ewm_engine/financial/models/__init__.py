"""Financial World Model neural components (JEPA encoder, VICReg, predictor, decoder)."""

from __future__ import annotations

from ewm_engine.financial.models.financial_decoder import (
    FinancialMetricsDecoder,
    FinancialPredictionOutput,
)
from ewm_engine.financial.models.jepa_encoder import (
    FinancialJEPAEncoder,
    FinancialJEPAPair,
    augment_cross_stock_mask,
    augment_time_warp,
)
from ewm_engine.financial.models.transition_predictor import (
    CausalInterventionEmbedder,
    FinancialTransitionPredictor,
    FinancialWorldModel,
)
from ewm_engine.financial.models.vicreg import (
    VICRegLoss,
    VICRegLossComponents,
    compute_rankme,
    sigreg_loss,
)

__all__ = [
    "CausalInterventionEmbedder",
    "FinancialJEPAEncoder",
    "FinancialJEPAPair",
    "FinancialMetricsDecoder",
    "FinancialPredictionOutput",
    "FinancialTransitionPredictor",
    "FinancialWorldModel",
    "VICRegLoss",
    "VICRegLossComponents",
    "augment_cross_stock_mask",
    "augment_time_warp",
    "compute_rankme",
    "sigreg_loss",
]
