"""Multi-seed statistical evaluation reporting mean +/- bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.evaluation.uncertainty import compute_bootstrap_ci


class MultiSeedMetricSummary(BaseModel):
    """Statistical summary of a metric evaluated across multiple independent seeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str = Field(description="Name of the evaluated simulation metric.")
    mean: float = Field(description="Sample mean across seed repetitions.")
    std: float = Field(description="Sample standard deviation across seed repetitions.")
    ci_lower: float = Field(
        description="Lower bound of non-parametric bootstrap confidence interval."
    )
    ci_upper: float = Field(
        description="Upper bound of non-parametric bootstrap confidence interval."
    )
    confidence_level: float = Field(default=0.95, description="Nominal confidence level.")
    n_seeds: int = Field(description="Number of independent seed evaluations.")
    seed_values: list[float] = Field(description="Raw metric values observed across seeds.")

    @property
    def display_str(self) -> str:
        """Formatted human-readable string: mean +/- half-width (CI: [lower, upper])."""
        half_width = (self.ci_upper - self.ci_lower) / 2.0
        return (
            f"{self.mean:.4f} ± {half_width:.4f} "
            f"({int(self.confidence_level * 100)}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}])"
        )


class MultiSeedReport(BaseModel):
    """Aggregate multi-seed evaluation report ensuring zero bare point estimates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_seeds: int = Field(description="Total number of seeded trials.")
    seeds: tuple[int, ...] = Field(description="Sequence of evaluated seeds.")
    metrics: dict[str, MultiSeedMetricSummary] = Field(
        description="Per-metric multi-seed statistical summaries.",
    )

    def summary_table(self) -> str:
        """Format report into human-readable Markdown table."""
        lines = [
            f"### Multi-Seed Evaluation Report (N={self.n_seeds} seeds)",
            "",
            "| Metric | Mean | Std | 95% Bootstrap CI | Display |",
            "|---|---|---|---|---|",
        ]
        for name, summary in sorted(self.metrics.items()):
            ci_str = f"[{summary.ci_lower:.4f}, {summary.ci_upper:.4f}]"
            lines.append(
                f"| `{name}` | {summary.mean:.4f} | {summary.std:.4f} | {ci_str} | {summary.display_str} |"
            )
        return "\n".join(lines)


def evaluate_multiseed(
    trial_fn: Callable[[int], dict[str, float]],
    n_seeds: int = 5,
    master_seed: int = 42,
    confidence_level: float = 0.95,
) -> MultiSeedReport:
    """Execute evaluation function across N spawned seeds and compile bootstrap CIs.

    Guarantees that metrics are reported as mean +/- CI (never as a bare point estimate).
    Uses SeedSequence.spawn to ensure statistical independence across seeds.

    Args:
        trial_fn: Callable taking an integer seed and returning a dictionary of scalar metrics.
        n_seeds: Number of independent random seeds to evaluate (default: 5).
        master_seed: Master pseudo-random seed.
        confidence_level: Desired bootstrap confidence interval level (default: 0.95).

    Returns:
        MultiSeedReport containing distributions and bootstrap CIs for each metric.
    """
    seed_seq = np.random.SeedSequence(master_seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in seed_seq.spawn(n_seeds)]

    collected_metrics: dict[str, list[float]] = {}

    for s in child_seeds:
        trial_results = trial_fn(s)
        for k, v in trial_results.items():
            collected_metrics.setdefault(k, []).append(float(v))

    metric_summaries: dict[str, MultiSeedMetricSummary] = {}
    for k, values in collected_metrics.items():
        arr = np.asarray(values, dtype=float)
        ci = compute_bootstrap_ci(
            values=values,
            confidence_level=confidence_level,
            n_resamples=1000,
            seed=master_seed,
        )
        metric_summaries[k] = MultiSeedMetricSummary(
            metric_name=k,
            mean=float(np.mean(arr)),
            std=float(np.std(arr)),
            ci_lower=ci.ci_lower,
            ci_upper=ci.ci_upper,
            confidence_level=confidence_level,
            n_seeds=n_seeds,
            seed_values=[float(x) for x in arr],
        )

    return MultiSeedReport(
        n_seeds=n_seeds,
        seeds=tuple(child_seeds),
        metrics=metric_summaries,
    )
