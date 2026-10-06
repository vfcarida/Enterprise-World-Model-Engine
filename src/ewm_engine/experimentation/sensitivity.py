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
        """Execute Sobol variance-based sensitivity analysis."""
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

    def analyze_morris(
        self,
        evaluator: Callable[[dict[str, Any]], float],
        n_trajectories: int = 10,
        num_levels: int = 4,
        seed: int = 42,
    ) -> MorrisResult:
        """Execute Morris elementary effects screening analysis."""
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
