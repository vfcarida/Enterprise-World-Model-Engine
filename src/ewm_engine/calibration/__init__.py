"""Simulation-based Bayesian calibration (SBI) with certified coverage.

[EXPERIMENTAL] This module implements amortized Bayesian parameter calibration (NPE),
mandatory coverage certification (TARP, Posterior-SBC), misspecification-robust diagnostics,
and refusal gates for uncertified posteriors.

Anchors:
- Lavin et al. "Simulation Intelligence", arXiv:2112.03235
- Gloeckler et al. "All-in-one SBI (Simformer)", arXiv:2404.09636 (ICML 2024)
- Boelts et al. "sbi reloaded", arXiv:2411.17337
- Huang et al. "Robust SBI under Model Misspecification", arXiv:2305.15871 (NeurIPS 2023)
- Lemos et al. "Sampling-based coverage tests (TARP)", arXiv:2302.03026 (ICML 2023)
"""

from __future__ import annotations

from ewm_engine.calibration.certification import (
    CoverageCertificate,
    certify_coverage,
    compute_misspecification_diagnostic,
    compute_posterior_sbc,
    compute_tarp_coverage,
)
from ewm_engine.calibration.priors import CalibrationPrior, PriorSet
from ewm_engine.calibration.report import CalibrationReport
from ewm_engine.calibration.sbi import AmortizedPosteriorEstimator, PosteriorDistribution
from ewm_engine.calibration.simulator import ForwardSimulatorWrapper
from ewm_engine.calibration.workflow import CalibrationWorkflow

__all__ = [
    "AmortizedPosteriorEstimator",
    "CalibrationPrior",
    "CalibrationReport",
    "CalibrationWorkflow",
    "CoverageCertificate",
    "ForwardSimulatorWrapper",
    "PosteriorDistribution",
    "PriorSet",
    "certify_coverage",
    "compute_misspecification_diagnostic",
    "compute_posterior_sbc",
    "compute_tarp_coverage",
]
