"""Pluggable dynamics models and composition engine for EWM Engine."""

from __future__ import annotations

from ewm_engine.dynamics.base import (
    ContextualDynamicsModel,
    DynamicsModel,
    InfeasibleResidualPolicy,
    MechanismKind,
    MechanismProvenance,
    ResidualFeasibilityRecord,
    TransitionResult,
    ValidityDomain,
)
from ewm_engine.dynamics.calibration import (
    AbstainException,
    CalibrationEvaluationReport,
    ConfidenceInterval,
    ConformalPredictorWrapper,
    DriftStatusReport,
    LearnedModelRegistry,
    MahalanobisOODAndDriftDetector,
    Predictor,
    compute_ensemble_variance,
    compute_expected_calibration_error,
)
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import (
    DeterministicDemandDynamics,
    DeterministicDynamics,
    DeterministicTransferDynamics,
)
from ewm_engine.dynamics.learned import (
    LearnedDynamics,
    LinearResidualDynamics,
    TransitionDataset,
    TransitionSample,
)
from ewm_engine.dynamics.router import (
    CalibratedPrediction,
    CalibratedStateDistribution,
    HybridTransitionRouter,
    LearnedPredictorProtocol,
    NumericalInstabilityError,
    TransitionEngine,
    TransitionStepOutcome,
    validate_learned_prediction_no_nan,
)
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.experimental.grey_box import (
    GreyBoxDynamics,
    HybridDynamics,
    ResidualEstimator,
    ZeroResidualEstimator,
)

__all__ = [
    "AbstainException",
    "CalibratedPrediction",
    "CalibratedStateDistribution",
    "CalibrationEvaluationReport",
    "CompositeDynamics",
    "ConfidenceInterval",
    "ConformalPredictorWrapper",
    "ContextualDynamicsModel",
    "DeterministicDemandDynamics",
    "DeterministicDynamics",
    "DeterministicTransferDynamics",
    "DriftStatusReport",
    "DynamicsModel",
    "GreyBoxDynamics",
    "HybridDynamics",
    "HybridTransitionRouter",
    "InfeasibleResidualPolicy",
    "LearnedDynamics",
    "LearnedModelRegistry",
    "LearnedPredictorProtocol",
    "LinearResidualDynamics",
    "MahalanobisOODAndDriftDetector",
    "MechanismKind",
    "MechanismProvenance",
    "NumericalInstabilityError",
    "Predictor",
    "ResidualEstimator",
    "ResidualFeasibilityRecord",
    "StochasticDemandDynamics",
    "TransitionDataset",
    "TransitionEngine",
    "TransitionResult",
    "TransitionSample",
    "TransitionStepOutcome",
    "ValidityDomain",
    "ZeroResidualEstimator",
    "compute_ensemble_variance",
    "compute_expected_calibration_error",
    "validate_learned_prediction_no_nan",
]
