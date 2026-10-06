"""Experimentation suite: Design of Experiments (DoE), backtesting, sensitivity, calibration, and intervention optimization.

Conforms to Track T4 (Horizon v1.4):
- Part A: Core substrate (zero-dep): DoE sweeps, tornado diagrams, walk-forward backtesting,
  scoring metrics (RMSE, moments, empirical coverage/SBC, CRPS, spectral distance),
  Sampler, Surrogate, Optimizer protocols and pure NumPy built-ins.
- Part B: Statistical & optimization extras: SALib (Sobol/Morris), DistanceCalibrator (MSM loss),
  pycma, Nevergrad, pymoo (NSGA-II multi-objective Pareto front), scikit-learn GaussianProcessSurrogate.
- Part C: Heavy adapters (quarantined): SBINeuralInferenceAdapter, AxBayesianOptimizerAdapter.
"""

from __future__ import annotations

from ewm_engine.experimentation.adapters import (
    AxBayesianOptimizerAdapter,
    SBINeuralInferenceAdapter,
)
from ewm_engine.experimentation.backtest import (
    BacktestReport,
    BacktestWindow,
    run_walk_forward_backtest,
    score_crps,
    score_empirical_coverage,
    score_moments_distance,
    score_rmse,
    score_spectral_distance,
)
from ewm_engine.experimentation.calibration import (
    CalibrationResult,
    DistanceCalibrator,
    PyABCPosteriorAdapter,
    compute_msm_moments,
    is_scipy_available,
    msm_distance_loss,
)
from ewm_engine.experimentation.doe import (
    DesignType,
    SweepResult,
    TornadoBar,
    TornadoData,
    compute_tornado_data,
    generate_full_factorial,
    generate_halton,
    generate_lhs,
    generate_oat,
    run_doe_sweep,
)
from ewm_engine.experimentation.evolutionary import (
    NevergradOptimizer,
    PyCMAOptimizer,
    is_cma_available,
    is_nevergrad_available,
)
from ewm_engine.experimentation.optimizers import (
    HillClimbingOptimizer,
    RandomSearchOptimizer,
)
from ewm_engine.experimentation.params import (
    ParameterDef,
    ParameterSpace,
    ParameterType,
)
from ewm_engine.experimentation.pareto_opt import (
    PymooParetoOptimizer,
    is_pymoo_available,
)
from ewm_engine.experimentation.protocols import (
    ObjectiveVector,
    OptimizationDirection,
    OptimizationResult,
    Optimizer,
    ParetoFront,
    ParetoPolicy,
    Sampler,
    Surrogate,
)
from ewm_engine.experimentation.samplers import (
    HaltonSampler,
    LatinHypercubeSampler,
    RandomSampler,
)
from ewm_engine.experimentation.sensitivity import (
    GlobalSensitivityAnalyzer,
    MorrisResult,
    SobolResult,
    is_salib_available,
)
from ewm_engine.experimentation.surrogate_sklearn import (
    GaussianProcessSurrogate,
    is_sklearn_available,
)

__all__ = [
    "AxBayesianOptimizerAdapter",
    "BacktestReport",
    "BacktestWindow",
    "CalibrationResult",
    "DesignType",
    "DistanceCalibrator",
    "GaussianProcessSurrogate",
    "GlobalSensitivityAnalyzer",
    "HaltonSampler",
    "HillClimbingOptimizer",
    "LatinHypercubeSampler",
    "MorrisResult",
    "NevergradOptimizer",
    "ObjectiveVector",
    "OptimizationDirection",
    "OptimizationResult",
    "Optimizer",
    "ParameterDef",
    "ParameterSpace",
    "ParameterType",
    "ParetoFront",
    "ParetoPolicy",
    "PyABCPosteriorAdapter",
    "PyCMAOptimizer",
    "PymooParetoOptimizer",
    "RandomSampler",
    "RandomSearchOptimizer",
    "SBINeuralInferenceAdapter",
    "Sampler",
    "SobolResult",
    "Surrogate",
    "SweepResult",
    "TornadoBar",
    "TornadoData",
    "compute_msm_moments",
    "compute_tornado_data",
    "generate_full_factorial",
    "generate_halton",
    "generate_lhs",
    "generate_oat",
    "is_cma_available",
    "is_nevergrad_available",
    "is_pymoo_available",
    "is_salib_available",
    "is_scipy_available",
    "is_sklearn_available",
    "msm_distance_loss",
    "run_doe_sweep",
    "run_walk_forward_backtest",
    "score_crps",
    "score_empirical_coverage",
    "score_moments_distance",
    "score_rmse",
    "score_spectral_distance",
]
