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
        self._x_train: np.ndarray | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._chol: np.ndarray | None = None
        self._alpha_vec: np.ndarray | None = None
        self._length_scale: float = 1.0

    @staticmethod
    def _matern52_kernel(x1: np.ndarray, x2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
        """Compute Matérn (nu=2.5) covariance matrix between x1 (N, D) and x2 (M, D)."""
        diff = (x1[:, None, :] - x2[None, :, :]) / max(length_scale, 1e-6)
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        sqrt5_d = np.sqrt(5.0) * dist
        return np.asarray(
            (1.0 + sqrt5_d + (5.0 / 3.0) * (dist**2)) * np.exp(-sqrt5_d), dtype=np.float64
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit GP regressor emulator on input matrix X and target y."""
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        if is_sklearn_available():
            gp_cls = _load_gp_regressor()
            kernel_cls = _load_gp_kernel()
            kernel = kernel_cls(nu=self.nu)
            self._model = gp_cls(
                kernel=kernel,
                alpha=self.alpha,
                normalize_y=True,
                random_state=self.random_state,
            )
            self._model.fit(X_arr, y_arr)
            return

        # Pure-NumPy exact Gaussian Process via Cholesky factorization
        self._x_train = X_arr.copy()
        self._y_mean = float(np.mean(y_arr))
        raw_std = float(np.std(y_arr))
        self._y_std = raw_std if raw_std > 1e-12 else 1.0
        y_norm = (y_arr - self._y_mean) / self._y_std

        pairwise = np.sqrt(np.sum((X_arr[:, None, :] - X_arr[None, :, :]) ** 2, axis=-1))
        non_zero = pairwise[pairwise > 0]
        self._length_scale = float(np.median(non_zero)) if len(non_zero) > 0 else 1.0

        k_mat = self._matern52_kernel(X_arr, X_arr, self._length_scale)
        k_reg = k_mat + max(self.alpha, 1e-8) * np.eye(X_arr.shape[0], dtype=np.float64)
        self._chol = np.linalg.cholesky(k_reg)
        # Solve (L L^T) alpha = y_norm
        v = np.linalg.solve(self._chol, y_norm)
        self._alpha_vec = np.linalg.solve(self._chol.T, v)
        self._model = "numpy_gp"

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict expected value and standard deviation across points X."""
        if self._model is None:
            raise ValueError("GaussianProcessSurrogate must be fit before calling predict().")
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self._model != "numpy_gp":
            y_mean, y_std = self._model.predict(X_arr, return_std=True)
            return np.asarray(y_mean, dtype=np.float64), np.asarray(y_std, dtype=np.float64)

        assert self._x_train is not None and self._chol is not None and self._alpha_vec is not None
        k_trans = self._matern52_kernel(X_arr, self._x_train, self._length_scale)
        y_mean_norm = k_trans @ self._alpha_vec
        y_mean = self._y_mean + self._y_std * y_mean_norm

        v = np.linalg.solve(self._chol, k_trans.T)
        y_var_norm = np.maximum(0.0, 1.0 - np.sum(v**2, axis=0))
        y_std = self._y_std * np.sqrt(y_var_norm)
        return np.asarray(y_mean, dtype=np.float64), np.asarray(y_std, dtype=np.float64)
