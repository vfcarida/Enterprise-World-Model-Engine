"""Quarantined adapters for neural simulation-based inference (sbi) and Bayesian optimization (Ax/BoTorch).

Conforms to Track T4: Heavy ADAPTERS (quarantined).
Strictly isolated: does not import torch, sbi, or ax at top level.
Provides SBINeuralInferenceAdapter and AxBayesianOptimizerAdapter.
"""

from __future__ import annotations

import importlib
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError


class SBINeuralInferenceAdapter:
    """Documented adapter for Simulation-Based Inference (SBI) using neural posterior estimators.

    Treats the simulation engine as a stochastic black-box simulator x ~ p(x | theta) to learn
    amortized neural posteriors q(theta | x) via Neural Posterior Estimation (NPE / SNPE).

    Epistemic Note (SBI Trust Crisis):
        Neural posterior estimators are vulnerable to overconfidence and false certainty
        when the generative world model is misspecified relative to empirical real-world data.
        Always run Simulation-Based Calibration (SBC) and TARP (Test of Accuracy of
        Rank Posteriors) diagnostics before treating amortized neural posteriors as credible.
    """

    def __init__(self, method: str = "SNPE") -> None:
        self.method = method
        try:
            importlib.import_module("sbi")
            importlib.import_module("torch")
        except (ImportError, ModuleNotFoundError) as err:
            raise SimulationConfigurationError(
                "sbi and torch are required for neural simulation-based inference. "
                "Install with: pip install 'sbi>=0.23.0' 'torch>=2.0.0'"
            ) from err

    def train_posterior(self, *args: Any, **kwargs: Any) -> Any:
        """Fit neural density estimator over simulation rollouts."""
        raise NotImplementedError("SBI neural inference execution.")


class AxBayesianOptimizerAdapter:
    """Documented adapter for Bayesian Optimization via Ax and BoTorch.

    Executes Gaussian Process / Multi-Task Bayesian Optimization over black-box
    simulation rollouts using BoTorch acquisition functions.
    """

    def __init__(self, experiment_name: str = "EWM_Bayesian_Opt") -> None:
        self.experiment_name = experiment_name
        try:
            importlib.import_module("ax")
            importlib.import_module("botorch")
        except (ImportError, ModuleNotFoundError) as err:
            raise SimulationConfigurationError(
                "ax-platform and botorch are required for Bayesian optimization. "
                "Install with: pip install 'ax-platform>=0.3.0' 'botorch>=0.10.0'"
            ) from err

    def run_optimization(self, *args: Any, **kwargs: Any) -> Any:
        """Run Ax Bayesian optimization loop."""
        raise NotImplementedError("Ax Bayesian optimization execution.")
