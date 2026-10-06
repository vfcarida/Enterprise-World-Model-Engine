"""RTAMT Signal Temporal Logic (STL) adapter for full continuous and discrete STL monitoring.

Conforms to Track T3: Extra [stl] (Full STL/MTL via RTAMT).
Strictly isolated: requires optional '[stl]' dependency (rtamt).
Reports quantitative robustness margins and aggregates robustness distributions
across Monte Carlo rollouts.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.simulation.trajectory import Trajectory
from ewm_engine.verification.temporal import extract_trajectory_signals


def is_rtamt_available() -> bool:
    """Check whether the optional rtamt package is installed."""
    try:
        importlib.import_module("rtamt")
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _load_rtamt() -> Any:
    """Lazily load the rtamt module or raise informative error."""
    try:
        return importlib.import_module("rtamt")
    except (ImportError, ModuleNotFoundError) as err:
        raise SimulationConfigurationError(
            "RTAMT is required for full STL evaluation. Install with: pip install 'ewm-engine[stl]'"
        ) from err


class RobustnessDistribution(BaseModel):
    """Statistical summary of quantitative STL robustness margins across Monte Carlo rollouts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    formula: str = Field(
        description="Evaluated Signal Temporal Logic formula.",
    )
    rollout_count: int = Field(
        ge=0,
        description="Number of rollouts evaluated.",
    )
    satisfaction_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Empirical probability that the STL property holds (robustness >= 0).",
    )
    mean_robustness: float = Field(
        description="Expected robustness margin across all rollouts.",
    )
    std_robustness: float = Field(
        ge=0.0,
        description="Standard deviation of robustness margins across rollouts.",
    )
    min_robustness: float = Field(
        description="Minimum observed robustness margin.",
    )
    max_robustness: float = Field(
        description="Maximum observed robustness margin.",
    )
    quantiles: dict[str, float] = Field(
        default_factory=dict,
        description="Quantile thresholds for robustness (p05, p25, p50, p75, p95).",
    )
    cvar_05: float = Field(
        description="Conditional Value-at-Risk at 5% (mean of worst 5% robustness outcomes).",
    )
    individual_robustness: tuple[float, ...] = Field(
        default_factory=tuple,
        description="Individual robustness margins per trajectory rollout.",
    )


class RTAMTEvaluationBackend:
    """Adapter evaluating full STL/MTL formulas via RTAMT discrete-time specification."""

    def __init__(
        self,
        formula: str,
        variables: Mapping[str, str],
        spec_name: str = "TrajectorySTLSpec",
    ) -> None:
        """Initialize RTAMT backend.

        Args:
            formula: Standard STL string formula (e.g. 'always(x >= 0)', 'always(x > 10 -> eventually[0, 5] (y < 2))').
            variables: Mapping of variable name to type ('float' or 'int').
            spec_name: Identifier for the RTAMT specification.
        """
        self.formula = formula
        self.variables = dict(variables)
        self.spec_name = spec_name
        self._rtamt = _load_rtamt()

    def _build_spec(self) -> Any:
        spec = self._rtamt.StlDiscreteTimeSpecification()
        spec.name = self.spec_name
        for var_name, var_type in self.variables.items():
            spec.declare_var(var_name, var_type)
        spec.spec = self.formula
        spec.parse()
        return spec

    def evaluate_signals(self, signals: Mapping[str, Sequence[float]]) -> float:
        """Evaluate formula over a dictionary of 1D signal timeseries.

        Returns:
            Initial robustness margin rho at step t=0.
        """
        spec = self._build_spec()
        length = len(next(iter(signals.values())))
        time_arr = list(range(length))

        dataset: dict[str, Any] = {"time": time_arr}
        for k, v in signals.items():
            dataset[k] = list(v)

        result = spec.evaluate(dataset)
        # result is [[t0, rob0], [t1, rob1], ...]
        if isinstance(result, list) and result:
            return float(result[0][1])
        return 0.0

    def evaluate_trajectory(self, trajectory: Trajectory) -> float:
        """Evaluate formula over an individual Trajectory rollout."""
        signals = extract_trajectory_signals(trajectory)
        relevant_signals = {k: list(signals[k]) for k in self.variables.keys() if k in signals}
        if len(relevant_signals) != len(self.variables):
            missing = set(self.variables.keys()) - set(relevant_signals.keys())
            raise KeyError(f"Trajectory missing required signals for RTAMT evaluation: {missing}")
        return self.evaluate_signals(relevant_signals)

    def evaluate_monte_carlo(
        self,
        trajectories: Sequence[Trajectory],
    ) -> RobustnessDistribution:
        """Evaluate STL formula across a collection of Monte Carlo rollouts.

        Computes summary statistics and robustness distributions (quantiles, CVaR).
        """
        if not trajectories:
            raise ValueError("Cannot evaluate Monte Carlo STL on empty trajectories sequence.")

        robs = [self.evaluate_trajectory(t) for t in trajectories]
        robs_arr = np.array(robs, dtype=np.float64)

        sat_prob = float(np.mean(robs_arr >= 0.0))
        mean_rob = float(np.mean(robs_arr))
        std_rob = float(np.std(robs_arr))
        min_rob = float(np.min(robs_arr))
        max_rob = float(np.max(robs_arr))

        # Quantiles
        quantiles = {
            "p05": float(np.percentile(robs_arr, 5)),
            "p25": float(np.percentile(robs_arr, 25)),
            "p50": float(np.percentile(robs_arr, 50)),
            "p75": float(np.percentile(robs_arr, 75)),
            "p95": float(np.percentile(robs_arr, 95)),
        }

        # CVaR 5%
        p05_val = quantiles["p05"]
        tail_vals = robs_arr[robs_arr <= p05_val]
        cvar_05 = float(np.mean(tail_vals)) if len(tail_vals) > 0 else p05_val

        return RobustnessDistribution(
            formula=self.formula,
            rollout_count=len(trajectories),
            satisfaction_probability=sat_prob,
            mean_robustness=mean_rob,
            std_robustness=std_rob,
            min_robustness=min_rob,
            max_robustness=max_rob,
            quantiles=quantiles,
            cvar_05=cvar_05,
            individual_robustness=tuple(float(x) for x in robs),
        )
