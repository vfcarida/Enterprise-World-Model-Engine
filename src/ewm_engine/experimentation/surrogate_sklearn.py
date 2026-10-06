"""Gaussian Process emulator surrogate via scikit-learn.

Conforms to Track T4: Extra [surrogate] (scikit-learn).
Strictly isolated: requires optional scikit-learn dependency.
Approximates expensive simulation rollouts with a fast GaussianProcessRegressor emulator.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.experimentation.protocols import Surrogate


def is_sklearn_available() -> bool:
    """Check whether scikit-learn package is installed."""
    try:
        importlib.import_module("sklearn.gaussian_process")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _load_gp_regressor() -> Any:
    """Lazily load GaussianProcessRegressor from scikit-learn."""
    try:
        mod = importlib.import_module("sklearn.gaussian_process")
        return mod.GaussianProcessRegressor
    except (ImportError, ModuleNotFoundError) as err:
        raise SimulationConfigurationError(
            "scikit-learn is required for GaussianProcessSurrogate. Install with: pip install 'ewm-engine[surrogate]'"
        ) from err


def _load_gp_kernel() -> Any:
    """Lazily load Matern kernel from scikit-learn."""
    mod = importlib.import_module("sklearn.gaussian_process.kernels")
    return mod.Matern


class GaussianProcessSurrogate(Surrogate):
    """Gaussian Process Regressor emulator approximating simulation rollouts."""

    def __init__(self, nu: float = 2.5, alpha: float = 1e-6, random_state: int = 42) -> None:
        self.nu = nu
        self.alpha = alpha
        self.random_state = random_state
        self._model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP regressor emulator on input matrix X and target y."""
        gp_cls = _load_gp_regressor()
        kernel_cls = _load_gp_kernel()

        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        kernel = kernel_cls(nu=self.nu)
        self._model = gp_cls(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=True,
            random_state=self.random_state,
        )
        self._model.fit(X_arr, y_arr)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected value and standard deviation across points X."""
        if self._model is None:
            raise ValueError("GaussianProcessSurrogate must be fit before calling predict().")
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y_mean, y_std = self._model.predict(X_arr, return_std=True)
        return np.asarray(y_mean, dtype=np.float64), np.asarray(y_std, dtype=np.float64)
