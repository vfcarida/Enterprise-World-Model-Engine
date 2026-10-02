"""Scenario comparison framework for evaluating counterfactual interventions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ewm_engine.evaluation.uncertainty import UncertaintyDistribution, summarize_distribution

if TYPE_CHECKING:
    from ewm_engine.simulation.trajectory import SimulationResult


class ScenarioComparison:
    """Comparative analysis across alternative interventional scenarios against a baseline.

    Provides machine-readable data structures and formatted human-readable delta tables.
    """

    def __init__(
        self,
        baseline_name: str,
        results_by_scenario: dict[str, dict[str, UncertaintyDistribution]],
        deltas: dict[str, dict[str, dict[str, float]]],
    ) -> None:
        self.baseline_name = baseline_name
        self.results_by_scenario = results_by_scenario
        self.deltas = deltas

    def to_dict(self) -> dict[str, Any]:
        """Convert comparative evaluation results to a clean machine-readable dictionary."""
        serialized_results: dict[str, dict[str, dict[str, float]]] = {}
        for scn_name, m_dict in self.results_by_scenario.items():
            serialized_results[scn_name] = {
                m_name: dist.model_dump() for m_name, dist in m_dict.items()
            }

        return {
            "baseline": self.baseline_name,
            "scenarios": serialized_results,
            "deltas_vs_baseline": self.deltas,
        }

    def summary_table(self) -> str:
        """Render a formatted text table comparing scenarios and delta metrics."""
        lines: list[str] = []
        lines.append(f"=== Scenario Comparison (Baseline: {self.baseline_name}) ===")

        all_metrics = (
            list(next(iter(self.results_by_scenario.values())).keys())
            if self.results_by_scenario
            else []
        )

        header = f"{'Metric':<25} | {'Scenario':<22} | {'Mean (Std)':<18} | {'p50 [p05, p95]':<20} | {'Delta vs Base':<14}"
        separator = "-" * len(header)
        lines.append(separator)
        lines.append(header)
        lines.append(separator)

        for metric in all_metrics:
            base_dist = self.results_by_scenario[self.baseline_name][metric]
            base_mean_str = f"{base_dist.mean:.2f} (+/-{base_dist.std:.2f})"
            base_q_str = f"{base_dist.median:.2f} [{base_dist.p05:.2f}, {base_dist.p95:.2f}]"
            lines.append(
                f"{metric:<25} | {self.baseline_name:<22} | {base_mean_str:<18} | {base_q_str:<20} | {'-':<14}"
            )

            for scn_name, scn_data in self.results_by_scenario.items():
                if scn_name == self.baseline_name:
                    continue
                dist = scn_data[metric]
                mean_str = f"{dist.mean:.2f} (+/-{dist.std:.2f})"
                q_str = f"{dist.median:.2f} [{dist.p05:.2f}, {dist.p95:.2f}]"

                delta_info = self.deltas[scn_name][metric]
                abs_delta = delta_info["absolute_delta"]
                pct_delta = delta_info["percent_delta"]
                pct_str = f"({pct_delta:+.1f}%)" if abs(pct_delta) < 10000 else ""
                delta_str = f"{abs_delta:+.2f} {pct_str}".strip()

                lines.append(
                    f"{'':<25} | {scn_name:<22} | {mean_str:<18} | {q_str:<20} | {delta_str:<14}"
                )
            lines.append(separator)

        return "\n".join(lines)


def compare_scenarios(
    baseline: SimulationResult,
    candidates: Sequence[SimulationResult],
    metrics: Sequence[str] | None = None,
) -> ScenarioComparison:
    """Compare candidate counterfactual simulation outcomes against a baseline run.

    Args:
        baseline: The baseline simulation result.
        candidates: Sequence of counterfactual simulation results.
        metrics: Optional list of metric names to evaluate. If omitted, infers common metrics.
    """
    baseline_name = baseline.scenario.name
    all_runs = [baseline, *candidates]

    # Infer metrics if not provided
    if metrics is None:
        inferred_metrics: set[str] = set()
        for t in baseline.trajectories:
            if t.steps:
                inferred_metrics.update(t.steps[-1].step_metrics.keys())
        metric_names = sorted(inferred_metrics)
    else:
        metric_names = list(metrics)

    results_by_scenario: dict[str, dict[str, UncertaintyDistribution]] = {}

    for run in all_runs:
        scn_metrics: dict[str, UncertaintyDistribution] = {}
        for m_name in metric_names:
            values = [t.final_metric(m_name) for t in run.trajectories]
            scn_metrics[m_name] = summarize_distribution(values)
        results_by_scenario[run.scenario.name] = scn_metrics

    # Compute deltas against baseline
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    base_metrics = results_by_scenario[baseline_name]

    for cand in candidates:
        cand_name = cand.scenario.name
        cand_deltas: dict[str, dict[str, float]] = {}
        for m_name in metric_names:
            base_mean = base_metrics[m_name].mean
            cand_mean = results_by_scenario[cand_name][m_name].mean
            abs_delta = cand_mean - base_mean
            pct_delta = (abs_delta / base_mean * 100.0) if base_mean != 0.0 else 0.0
            cand_deltas[m_name] = {
                "absolute_delta": float(abs_delta),
                "percent_delta": float(pct_delta),
            }
        deltas[cand_name] = cand_deltas

    return ScenarioComparison(
        baseline_name=baseline_name,
        results_by_scenario=results_by_scenario,
        deltas=deltas,
    )
