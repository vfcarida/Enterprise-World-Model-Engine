"""Global Sensitivity Analysis via SALib (Sobol, Morris, FAST).

Conforms to Track T4: Extra [sensitivity].
Strictly isolated: requires optional dependency SALib.
Computes first-order, total-order, and interaction sensitivity indices
over continuous and integer parameters.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.experimentation.params import ParameterSpace


def is_salib_available() -> bool:
    """Check whether the optional SALib package is installed."""
    try:
        importlib.import_module("SALib")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _load_salib() -> Any:
    """Lazily load SALib module or raise informative error."""
    try:
        return importlib.import_module("SALib")
    except (ImportError, ModuleNotFoundError) as err:
        raise SimulationConfigurationError(
            "SALib is required for global sensitivity analysis. Install with: pip install 'ewm-engine[sensitivity]'"
        ) from err


class SobolResult(BaseModel):
    """Sobol global sensitivity analysis indices."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    names: tuple[str, ...] = Field(description="Parameter names analyzed.")
    s1: dict[str, float] = Field(
        description="First-order Sobol indices (direct individual variance contribution)."
    )
    s1_conf: dict[str, float] = Field(description="First-order confidence intervals (95%).")
    st: dict[str, float] = Field(
        description="Total-order Sobol indices (including all parameter interactions)."
    )
    st_conf: dict[str, float] = Field(description="Total-order confidence intervals (95%).")
    s2: dict[str, dict[str, float]] | None = Field(
        default=None,
        description="Second-order interaction indices between pairs of parameters.",
    )


class MorrisResult(BaseModel):
    """Morris elementary effects sensitivity analysis indices."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    names: tuple[str, ...] = Field(description="Parameter names analyzed.")
    mu: dict[str, float] = Field(description="Mean elementary effect.")
    mu_star: dict[str, float] = Field(
        description="Absolute mean elementary effect (overall importance)."
    )
    sigma: dict[str, float] = Field(
        description="Standard deviation of elementary effects (non-linear/interaction degree)."
    )


class GlobalSensitivityAnalyzer:
    """Adapter executing SALib sensitivity analysis across a ParameterSpace."""

    def __init__(self, space: ParameterSpace) -> None:
        self.space = space
        # Filter continuous and integer parameters
        self.numeric_params = [p for p in space.parameters if p.type in ("continuous", "integer")]
        if not self.numeric_params:
            raise ValueError(
                "GlobalSensitivityAnalyzer requires at least one numeric parameter with bounds."
            )

        self.problem: dict[str, Any] = {
            "num_vars": len(self.numeric_params),
            "names": [p.name for p in self.numeric_params],
            "bounds": [list(p.bounds) for p in self.numeric_params if p.bounds is not None],
        }

    def analyze_sobol(
        self,
        evaluator: Callable[[dict[str, Any]], float],
        n_samples: int = 128,
        calc_second_order: bool = False,
        seed: int = 42,
    ) -> SobolResult:
        """Execute Sobol variance-based sensitivity analysis (SALib or pure-NumPy Saltelli fallback)."""
        if not is_salib_available():
            return self._analyze_sobol_numpy(
                evaluator=evaluator,
                n_samples=n_samples,
                calc_second_order=calc_second_order,
                seed=seed,
            )

        _load_salib()
        sobol_sample = importlib.import_module("SALib.sample.sobol")
        sobol_analyze = importlib.import_module("SALib.analyze.sobol")

        # Generate sample matrix
        param_values = sobol_sample.sample(
            self.problem,
            N=n_samples,
            calc_second_order=calc_second_order,
            seed=seed,
        )

        # Evaluate model for each row
        baseline = self.space.get_baseline_point()
        y = np.zeros(param_values.shape[0], dtype=np.float64)

        for i in range(param_values.shape[0]):
            pt = dict(baseline)
            for j, p in enumerate(self.numeric_params):
                val = param_values[i, j]
                pt[p.name] = round(val) if p.type == "integer" else float(val)
            y[i] = float(evaluator(pt))

        # Perform Sobol analysis
        res = sobol_analyze.analyze(
            self.problem,
            y,
            calc_second_order=calc_second_order,
            print_to_console=False,
            seed=seed,
        )

        names = tuple(self.problem["names"])
        s1_dict = {names[k]: float(res["S1"][k]) for k in range(len(names))}
        s1_conf = {names[k]: float(res["S1_conf"][k]) for k in range(len(names))}
        st_dict = {names[k]: float(res["ST"][k]) for k in range(len(names))}
        st_conf = {names[k]: float(res["ST_conf"][k]) for k in range(len(names))}

        s2_dict: dict[str, dict[str, float]] | None = None
        if calc_second_order and "S2" in res and res["S2"] is not None:
            s2_dict = {}
            for i, n1 in enumerate(names):
                s2_dict[n1] = {}
                for j, n2 in enumerate(names):
                    val = res["S2"][i][j]
                    s2_dict[n1][n2] = float(val) if not np.isnan(val) else 0.0

        return SobolResult(
            names=names,
            s1=s1_dict,
            s1_conf=s1_conf,
            st=st_dict,
            st_conf=st_conf,
            s2=s2_dict,
        )

    def _eval_matrix(
        self,
        mat: np.ndarray,
        evaluator: Callable[[dict[str, Any]], float],
    ) -> np.ndarray:
        """Evaluate parameter matrix of shape (N, D) against evaluator."""
        baseline = self.space.get_baseline_point()
        y = np.zeros(mat.shape[0], dtype=np.float64)
        for i in range(mat.shape[0]):
            pt = dict(baseline)
            for j, p in enumerate(self.numeric_params):
                val = float(mat[i, j])
                pt[p.name] = round(val) if p.type == "integer" else val
            y[i] = float(evaluator(pt))
        return y

    def _analyze_sobol_numpy(
        self,
        evaluator: Callable[[dict[str, Any]], float],
        n_samples: int = 128,
        calc_second_order: bool = False,
        seed: int = 42,
    ) -> SobolResult:
        """Pure-NumPy Saltelli (2010) / Jansen (1999) Sobol sensitivity estimator."""
        rng = np.random.default_rng(seed)
        d = len(self.numeric_params)
        names = tuple(self.problem["names"])
        bounds = np.asarray(self.problem["bounds"], dtype=np.float64)
        lows = bounds[:, 0]
        spans = bounds[:, 1] - bounds[:, 0]

        # Stratified Latin-hypercube style independent matrices A and B in parameter bounds
        u_a = (
            np.arange(n_samples, dtype=np.float64)[:, None] + rng.random((n_samples, d))
        ) / n_samples
        u_b = (
            np.arange(n_samples, dtype=np.float64)[:, None] + rng.random((n_samples, d))
        ) / n_samples
        for j in range(d):
            rng.shuffle(u_a[:, j])
            rng.shuffle(u_b[:, j])

        mat_a = lows + u_a * spans
        mat_b = lows + u_b * spans

        y_a = self._eval_matrix(mat_a, evaluator)
        y_b = self._eval_matrix(mat_b, evaluator)
        var_total = float(np.var(np.concatenate([y_a, y_b]), ddof=1))
        if var_total <= 1e-15:
            var_total = 1.0

        y_ab_list: list[np.ndarray] = []
        for j in range(d):
            mat_ab_j = mat_a.copy()
            mat_ab_j[:, j] = mat_b[:, j]
            y_ab_list.append(self._eval_matrix(mat_ab_j, evaluator))

        s1_dict: dict[str, float] = {}
        s1_conf: dict[str, float] = {}
        st_dict: dict[str, float] = {}
        st_conf: dict[str, float] = {}

        n_boot = 50
        for j, name in enumerate(names):
            y_ab = y_ab_list[j]
            # Saltelli (2010) first-order estimator & Jansen (1999) total-order estimator
            s1_val = float(np.mean(y_b * (y_ab - y_a)) / var_total)
            st_val = float(0.5 * np.mean((y_a - y_ab) ** 2) / var_total)
            s1_dict[name] = s1_val
            st_dict[name] = st_val

            boot_s1 = np.zeros(n_boot, dtype=np.float64)
            boot_st = np.zeros(n_boot, dtype=np.float64)
            for b_idx in range(n_boot):
                idx = rng.integers(0, n_samples, size=n_samples)
                v_b = float(np.var(np.concatenate([y_a[idx], y_b[idx]]), ddof=1)) or 1.0
                boot_s1[b_idx] = float(np.mean(y_b[idx] * (y_ab[idx] - y_a[idx])) / v_b)
                boot_st[b_idx] = float(0.5 * np.mean((y_a[idx] - y_ab[idx]) ** 2) / v_b)
            s1_conf[name] = float(1.96 * np.std(boot_s1, ddof=1))
            st_conf[name] = float(1.96 * np.std(boot_st, ddof=1))

        s2_dict: dict[str, dict[str, float]] | None = None
        if calc_second_order:
            s2_dict = {n1: dict.fromkeys(names, 0.0) for n1 in names}
            for i in range(d):
                for j in range(i + 1, d):
                    v_ij = float(np.mean(y_ab_list[i] * y_ab_list[j] - y_a * y_b) / var_total)
                    s2_val = max(0.0, v_ij - s1_dict[names[i]] - s1_dict[names[j]])
                    s2_dict[names[i]][names[j]] = s2_val
                    s2_dict[names[j]][names[i]] = s2_val

        return SobolResult(
            names=names,
            s1=s1_dict,
            s1_conf=s1_conf,
            st=st_dict,
            st_conf=st_conf,
            s2=s2_dict,
        )

    def analyze_morris(
        self,
        evaluator: Callable[[dict[str, Any]], float],
        n_trajectories: int = 10,
        num_levels: int = 4,
        seed: int = 42,
    ) -> MorrisResult:
        """Execute Morris elementary effects screening analysis (SALib or pure-NumPy fallback)."""
        if not is_salib_available():
            return self._analyze_morris_numpy(
                evaluator=evaluator,
                n_trajectories=n_trajectories,
                num_levels=num_levels,
                seed=seed,
            )

        _load_salib()
        morris_sample = importlib.import_module("SALib.sample.morris")
        morris_analyze = importlib.import_module("SALib.analyze.morris")

        param_values = morris_sample.sample(
            self.problem,
            N=n_trajectories,
            num_levels=num_levels,
            seed=seed,
        )

        baseline = self.space.get_baseline_point()
        y = np.zeros(param_values.shape[0], dtype=np.float64)

        for i in range(param_values.shape[0]):
            pt = dict(baseline)
            for j, p in enumerate(self.numeric_params):
                val = param_values[i, j]
                pt[p.name] = round(val) if p.type == "integer" else float(val)
            y[i] = float(evaluator(pt))

        res = morris_analyze.analyze(
            self.problem,
            param_values,
            y,
            print_to_console=False,
            seed=seed,
        )

        names = tuple(self.problem["names"])
        mu_dict = {names[k]: float(res["mu"][k]) for k in range(len(names))}
        mu_star_dict = {names[k]: float(res["mu_star"][k]) for k in range(len(names))}
        sigma_dict = {names[k]: float(res["sigma"][k]) for k in range(len(names))}

        return MorrisResult(
            names=names,
            mu=mu_dict,
            mu_star=mu_star_dict,
            sigma=sigma_dict,
        )

    def _analyze_morris_numpy(
        self,
        evaluator: Callable[[dict[str, Any]], float],
        n_trajectories: int = 10,
        num_levels: int = 4,
        seed: int = 42,
    ) -> MorrisResult:
        """Pure-NumPy Morris (1991) Elementary Effects screening estimator."""
        rng = np.random.default_rng(seed)
        d = len(self.numeric_params)
        names = tuple(self.problem["names"])
        bounds = np.asarray(self.problem["bounds"], dtype=np.float64)
        lows = bounds[:, 0]
        spans = bounds[:, 1] - bounds[:, 0]

        levels = max(2, num_levels)
        delta = levels / (2.0 * (levels - 1.0))
        grid_max = 1.0 - delta
        grid_choices = np.linspace(0.0, max(0.0, grid_max), max(1, levels // 2))

        ee = np.zeros((n_trajectories, d), dtype=np.float64)
        for r in range(n_trajectories):
            u_curr = rng.choice(grid_choices, size=d)
            order = rng.permutation(d)
            x_curr = lows + u_curr * spans
            y_curr = float(self._eval_matrix(x_curr[None, :], evaluator)[0])

            for j_idx in order:
                u_next = u_curr.copy()
                step_dir = 1.0 if u_curr[j_idx] + delta <= 1.0 + 1e-12 else -1.0
                u_next[j_idx] = np.clip(u_curr[j_idx] + step_dir * delta, 0.0, 1.0)
                actual_delta = u_next[j_idx] - u_curr[j_idx]
                if abs(actual_delta) < 1e-12:
                    actual_delta = delta
                x_next = lows + u_next * spans
                y_next = float(self._eval_matrix(x_next[None, :], evaluator)[0])
                ee[r, j_idx] = (y_next - y_curr) / actual_delta
                u_curr = u_next
                y_curr = y_next

        mu_dict = {names[j]: float(np.mean(ee[:, j])) for j in range(d)}
        mu_star_dict = {names[j]: float(np.mean(np.abs(ee[:, j]))) for j in range(d)}
        sigma_dict = {
            names[j]: float(np.std(ee[:, j], ddof=1)) if n_trajectories > 1 else 0.0
            for j in range(d)
        }

        return MorrisResult(
            names=names,
            mu=mu_dict,
            mu_star=mu_star_dict,
            sigma=sigma_dict,
        )
