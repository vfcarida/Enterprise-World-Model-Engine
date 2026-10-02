"""Uncertainty quantification and distribution statistics for Monte Carlo rollouts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class UncertaintyDistribution(BaseModel):
    """Statistical summary of metric outcomes across Monte Carlo rollouts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float = Field(description="Sample mean across trajectories.")
    std: float = Field(description="Sample standard deviation.")
    median: float = Field(description="Sample median (50th percentile).")
    p05: float = Field(description="5th percentile (lower tail risk).")
    p25: float = Field(description="25th percentile (first quartile).")
    p75: float = Field(description="75th percentile (third quartile).")
    p95: float = Field(description="95th percentile (upper tail risk).")
    min_val: float = Field(description="Observed minimum.")
    max_val: float = Field(description="Observed maximum.")
    iqr: float = Field(description="Interquartile range (p75 - p25).")
    cvar_05: float = Field(description="Conditional Value-at-Risk below 5th percentile.")


def summarize_distribution(values: Sequence[float]) -> UncertaintyDistribution:
    """Compute comprehensive uncertainty metrics from a sequence of rollout values."""
    if not values:
        return UncertaintyDistribution(
            mean=0.0,
            std=0.0,
            median=0.0,
            p05=0.0,
            p25=0.0,
            p75=0.0,
            p95=0.0,
            min_val=0.0,
            max_val=0.0,
            iqr=0.0,
            cvar_05=0.0,
        )

    arr = np.array(values, dtype=float)
    p05 = float(np.percentile(arr, 5))
    p25 = float(np.percentile(arr, 25))
    p50 = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    p95 = float(np.percentile(arr, 95))

    tail_values = arr[arr <= p05]
    cvar_05 = float(np.mean(tail_values)) if len(tail_values) > 0 else p05

    return UncertaintyDistribution(
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        median=p50,
        p05=p05,
        p25=p25,
        p75=p75,
        p95=p95,
        min_val=float(np.min(arr)),
        max_val=float(np.max(arr)),
        iqr=p75 - p25,
        cvar_05=cvar_05,
    )
