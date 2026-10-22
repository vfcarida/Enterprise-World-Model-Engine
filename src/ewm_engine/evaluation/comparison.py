"""Scenario comparison framework for evaluating counterfactual interventions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from ewm_engine.evaluation.pareto import (
    ObjectiveDirection,
    ObjectiveSpec,
    ParetoFrontier,
    compute_pareto_frontier,
)
from ewm_engine.evaluation.uncertainty import (
    BootstrapDelta,
    BootstrapIntervalMethod,
    UncertaintyDistribution,
    compute_bootstrap_delta,
    summarize_distribution,
)

if TYPE_CHECKING:
    from ewm_engine.simulation.trajectory import SimulationResult


class ScenarioComparison:
    """Comparative analysis across alternative interventional scenarios against a baseline.

    Provides machine-readable data structures, non-parametric bootstrap confidence intervals,
    multi-objective Pareto analysis, optional user-specified objective ranking, and formatted
    human-readable delta tables.

    Epistemic Invariant:
        Unless the user explicitly supplies an evaluation objective function, this engine
        refuses to declare an automated 'winner' among candidate policies.
    """

    def __init__(
        self,
        baseline_name: str,
        results_by_scenario: dict[str, dict[str, UncertaintyDistribution]],
        deltas: dict[str, dict[str, dict[str, float]]],
        bootstrap_deltas: dict[str, dict[str, BootstrapDelta]] | None = None,
        pareto_frontier: ParetoFrontier | None = None,
        rankings: list[tuple[str, float]] | None = None,
        estimand: str = "E_{P_model}[Y_candidate] - E_{P_model}[Y_baseline]",
    ) -> None:
        self.baseline_name = baseline_name
        self.results_by_scenario = results_by_scenario
        self.deltas = deltas
        self.bootstrap_deltas = bootstrap_deltas or {}
        self.pareto_frontier = pareto_frontier
        self.rankings = rankings
        self.estimand = estimand

    def to_dict(self) -> dict[str, Any]:
        """Convert comparative evaluation results to a clean machine-readable dictionary."""
        serialized_results: dict[str, dict[str, dict[str, Any]]] = {}
        for scn_name, m_dict in self.results_by_scenario.items():
            serialized_results[scn_name] = {
                m_name: dist.model_dump() for m_name, dist in m_dict.items()
            }

        serialized_boot: dict[str, dict[str, dict[str, Any]]] = {}
        for scn_name, b_dict in self.bootstrap_deltas.items():
            serialized_boot[scn_name] = {
                m_name: b_delta.model_dump() for m_name, b_delta in b_dict.items()
            }

        res: dict[str, Any] = {
            "baseline": self.baseline_name,
            "estimand": self.estimand,
            "scenarios": serialized_results,
            "deltas_vs_baseline": self.deltas,
            "bootstrap_deltas": serialized_boot,
            "rankings": (
                [{"scenario": name, "score": score} for name, score in self.rankings]
                if self.rankings is not None
                else None
            ),
        }
        if self.pareto_frontier is not None:
            res["pareto_frontier"] = self.pareto_frontier.model_dump()

        return res

    def summary_table(self) -> str:
        """Render a formatted text table comparing scenarios, delta metrics, and Pareto status."""
        lines: list[str] = []
        lines.append(f"=== Scenario Comparison (Baseline: {self.baseline_name}) ===")
        lines.append(
            f"Estimand: {self.estimand} (in-simulator P_model; not P_real unless externally validated)"
        )

        all_metrics = (
            list(next(iter(self.results_by_scenario.values())).keys())
            if self.results_by_scenario
            else []
        )

        header = f"{'Metric':<25} | {'Scenario':<20} | {'Mean (Std)':<18} | {'p50 [p05, p95]':<20} | {'Delta vs Base [CI] (*sig)':<28}"
        separator = "-" * len(header)
        lines.append(separator)
        lines.append(header)
        lines.append(separator)

        for metric in all_metrics:
            base_dist = self.results_by_scenario[self.baseline_name][metric]
            base_mean_str = f"{base_dist.mean:.2f} (+/-{base_dist.std:.2f})"
            base_q_str = f"{base_dist.median:.2f} [{base_dist.p05:.2f}, {base_dist.p95:.2f}]"
            lines.append(
                f"{metric:<25} | {self.baseline_name:<20} | {base_mean_str:<18} | {base_q_str:<20} | {'-':<28}"
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

                ci_suffix = ""
                if scn_name in self.bootstrap_deltas and metric in self.bootstrap_deltas[scn_name]:
                    b_delta = self.bootstrap_deltas[scn_name][metric]
                    if b_delta.is_deterministic:
                        sig_star = " (det)" if b_delta.absolute_delta != 0.0 else ""
                    else:
                        sig_star = " *" if b_delta.is_significant else ""
                    ci_suffix = f" [{b_delta.ci_lower:+.2f}, {b_delta.ci_upper:+.2f}]{sig_star}"

                delta_str = f"{abs_delta:+.2f} {pct_str}{ci_suffix}".strip()

                lines.append(
                    f"{'':<25} | {scn_name:<20} | {mean_str:<18} | {q_str:<20} | {delta_str:<28}"
                )
            lines.append(separator)

        if self.pareto_frontier is not None:
            lines.append("")
            lines.append("=== Multi-Objective Pareto Analysis ===")
            obj_desc = ", ".join(
                f"{o.metric_name} ({o.direction.value})" for o in self.pareto_frontier.objectives
            )
            lines.append(f"Evaluated Objectives: {obj_desc}")
            lines.append(
                f"Non-Dominated Pareto Frontier: {', '.join(self.pareto_frontier.frontier)}"
            )

            dominated_scenarios = [
                f"{name} (dominated by: {', '.join(doms)})"
                for name, doms in self.pareto_frontier.dominated_by.items()
                if doms
            ]
            if dominated_scenarios:
                lines.append(f"Dominated Candidates: {'; '.join(dominated_scenarios)}")
            else:
                lines.append(
                    "Dominated Candidates: None (all candidates are mutually non-dominated)"
                )

            if self.pareto_frontier.weighted_scores:
                w_str = ", ".join(
                    f"{name}: {score:.3f}"
                    for name, score in self.pareto_frontier.weighted_scores.items()
                )
                lines.append(f"Weighted Utility Scores: {w_str}")

            lines.append(f"Epistemic Note: {self.pareto_frontier.epistemic_warning}")
            lines.append(separator)

        if self.rankings is not None:
            lines.append("")
            lines.append("=== User Objective Ranking ===")
            for rank_idx, (name, score) in enumerate(self.rankings, start=1):
                lines.append(f"{rank_idx}. {name:<22} (score: {score:+.2f})")
            lines.append(separator)

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Render a rich, styled HTML card representation for interactive notebooks."""
        all_metrics = (
            list(next(iter(self.results_by_scenario.values())).keys())
            if self.results_by_scenario
            else []
        )
        cand_names = [k for k in self.results_by_scenario.keys() if k != self.baseline_name]

        rows: list[str] = []
        for metric in all_metrics:
            base_dist = self.results_by_scenario[self.baseline_name][metric]
            rows.append(
                f"<tr style='background: #f8fafc; font-weight: 500; border-top: 1px solid #e2e8f0;'>"
                f"<td style='padding: 6px 12px;'><code>{metric}</code></td>"
                f"<td style='padding: 6px 12px;'><span style='color: #64748b;'>{self.baseline_name}</span> (base)</td>"
                f"<td style='padding: 6px 12px; text-align: right; font-family: monospace;'>{base_dist.mean:.2f}</td>"
                f"<td style='padding: 6px 12px; text-align: right; font-family: monospace; color: #64748b;'>[{base_dist.p05:.2f}, {base_dist.p95:.2f}]</td>"
                f"<td style='padding: 6px 12px; text-align: center; color: #94a3b8;'>—</td>"
                f"</tr>"
            )
            for scn_name in cand_names:
                dist = self.results_by_scenario[scn_name][metric]
                delta_info = self.deltas[scn_name][metric]
                abs_delta = delta_info["absolute_delta"]
                pct_delta = delta_info["percent_delta"]

                ci_str = ""
                sig_badge = ""
                if scn_name in self.bootstrap_deltas and metric in self.bootstrap_deltas[scn_name]:
                    b_delta = self.bootstrap_deltas[scn_name][metric]
                    ci_str = f" <small style='color: #64748b;'>[{b_delta.ci_lower:+.2f}, {b_delta.ci_upper:+.2f}]</small>"
                    if b_delta.is_deterministic and b_delta.absolute_delta != 0.0:
                        sig_badge = " <span style='color: #64748b; font-weight: 600;'>(det)</span>"
                    elif b_delta.is_significant:
                        sig_badge = " <span style='color: #16a34a; font-weight: bold;'>*</span>"

                delta_color = (
                    "#16a34a" if abs_delta > 0 else ("#dc2626" if abs_delta < 0 else "#64748b")
                )
                pct_str = f" ({pct_delta:+.1f}%)" if abs(pct_delta) < 10000 else ""

                rows.append(
                    f"<tr style='border-top: 1px dashed #f1f5f9;'>"
                    f"<td style='padding: 6px 12px;'></td>"
                    f"<td style='padding: 6px 12px; font-weight: 600;'>{scn_name}</td>"
                    f"<td style='padding: 6px 12px; text-align: right; font-family: monospace;'>{dist.mean:.2f}</td>"
                    f"<td style='padding: 6px 12px; text-align: right; font-family: monospace; color: #64748b;'>[{dist.p05:.2f}, {dist.p95:.2f}]</td>"
                    f"<td style='padding: 6px 12px; text-align: right; font-family: monospace; color: {delta_color};'>"
                    f"{abs_delta:+.2f}{pct_str}{ci_str}{sig_badge}"
                    f"</td>"
                    f"</tr>"
                )

        table_html = (
            f"<table style='width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px;'>"
            f"<thead><tr style='border-bottom: 2px solid #cbd5e1; text-align: left; color: #475569; font-size: 11px; text-transform: uppercase;'>"
            f"<th style='padding: 6px 12px;'>Metric</th><th style='padding: 6px 12px;'>Scenario</th>"
            f"<th style='padding: 6px 12px; text-align: right;'>Mean</th><th style='padding: 6px 12px; text-align: right;'>p05 - p95</th>"
            f"<th style='padding: 6px 12px; text-align: right;'>Delta vs Base [CI]</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

        pareto_html = ""
        if self.pareto_frontier is not None:
            frontier_pills = " ".join(
                f"<span style='background: #f0fdf4; color: #166534; padding: 2px 8px; border-radius: 12px; font-weight: 600; font-size: 11px;'>{cand}</span>"
                for cand in self.pareto_frontier.frontier
            )
            pareto_html = (
                f"<div style='margin-top: 12px; padding: 10px; background: #f8fafc; border-radius: 6px; font-size: 12px;'>"
                f"<strong>Pareto Non-Dominated Frontier:</strong> {frontier_pills}"
                f"</div>"
            )

        return (
            f"<div style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; "
            f"border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; background: #ffffff; "
            f"box-shadow: 0 1px 3px rgba(0,0,0,0.05); max-width: 720px; margin: 8px 0; color: #1e293b;'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px;'>"
            f"<div>"
            f"<span style='font-weight: 700; font-size: 15px; color: #0f172a;'>ScenarioComparison</span>"
            f"<span style='background: #e0e7ff; color: #4338ca; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px; margin-left: 8px;'>Baseline: {self.baseline_name}</span>"
            f"</div>"
            f"<span style='font-size: 12px; color: #64748b;'>{len(cand_names)} candidate(s) evaluated</span>"
            f"</div>"
            f"{table_html}"
            f"{pareto_html}"
            f"</div>"
        )


def compare_scenarios(
    baseline: SimulationResult,
    candidates: Sequence[SimulationResult],
    metrics: Sequence[str] | None = None,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int | None = None,
    paired: bool = False,
    interval_method: BootstrapIntervalMethod = "percentile",
    estimand: str = "E_{P_model}[Y_candidate] - E_{P_model}[Y_baseline]",
    objectives: Sequence[ObjectiveSpec] | dict[str, str | ObjectiveDirection] | None = None,
    objective: Callable[[dict[str, float]], float] | None = None,
) -> ScenarioComparison:
    """Compare candidate counterfactual simulation outcomes against a baseline run.

    Computes summary quantiles, non-parametric bootstrap confidence intervals on delta metrics,
    empirical significance tests under simulation stochasticity, optional multi-objective
    Pareto frontier analysis, and optional user-objective ranking.

    Args:
        baseline: The baseline simulation result.
        candidates: Sequence of counterfactual simulation results.
        metrics: Optional list of metric names to evaluate. If omitted, infers common metrics.
        confidence_level: Nominal confidence level for delta bootstrap CIs (default: 0.95).
        n_bootstrap: Number of bootstrap resamples (default: 1000). Set to 0 to skip bootstrap.
        seed: Deterministic random seed for the bootstrap resampler.
        paired: If True, resamples paired differences d_i = Y_cand,i - Y_base,i (requires
            matched trajectory seeds / common random numbers and equal trajectory counts).
        interval_method: Bootstrap CI construction strategy ("percentile", "basic", "normal",
            "studentized", or "bca").
        estimand: Formal statistical estimand being estimated (defaults to P_model mean delta).
        objectives: Optional sequence of ObjectiveSpec or dict of {metric_name: direction}
            specifying multi-objective optimization criteria.
        objective: Optional user-supplied evaluation function mapping a scenario's metric means
            dict[str, float] to a scalar score. If omitted, no ranking or winner is declared.

    Returns:
        ScenarioComparison containing uncertainty distributions, deltas, bootstrap CIs,
        Pareto frontier analysis, and optional rankings.
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
            scn_metrics[m_name] = summarize_distribution(
                values,
                estimand=f"E_{{P_model}}[{m_name}]",
            )
        results_by_scenario[run.scenario.name] = scn_metrics

    # Compute deltas and bootstrap intervals against baseline
    deltas: dict[str, dict[str, dict[str, float]]] = {}
    bootstrap_deltas: dict[str, dict[str, BootstrapDelta]] = {}
    base_metrics = results_by_scenario[baseline_name]

    for cand_idx, cand in enumerate(candidates):
        cand_name = cand.scenario.name
        cand_deltas: dict[str, dict[str, float]] = {}
        cand_boot: dict[str, BootstrapDelta] = {}

        # Derive candidate-specific bootstrap seed if a parent seed was provided
        boot_seed = (seed + cand_idx * 1000) if seed is not None else None

        for m_name in metric_names:
            base_mean = base_metrics[m_name].mean
            cand_mean = results_by_scenario[cand_name][m_name].mean
            abs_delta = cand_mean - base_mean
            pct_delta = (abs_delta / base_mean * 100.0) if base_mean != 0.0 else 0.0
            cand_deltas[m_name] = {
                "absolute_delta": float(abs_delta),
                "percent_delta": float(pct_delta),
            }

            base_values = [t.final_metric(m_name) for t in baseline.trajectories]
            cand_values = [t.final_metric(m_name) for t in cand.trajectories]
            cand_boot[m_name] = compute_bootstrap_delta(
                baseline_values=base_values,
                candidate_values=cand_values,
                confidence_level=confidence_level,
                n_resamples=n_bootstrap,
                seed=boot_seed,
                paired=paired,
                interval_method=interval_method,
                estimand=f"E_{{P_model}}[{m_name}_candidate] - E_{{P_model}}[{m_name}_baseline]",
            )

        deltas[cand_name] = cand_deltas
        bootstrap_deltas[cand_name] = cand_boot

    # Multi-objective Pareto analysis
    pareto_frontier: ParetoFrontier | None = None
    if objectives is not None:
        parsed_objectives: list[ObjectiveSpec] = []
        if isinstance(objectives, dict):
            for m_name, d_val in objectives.items():
                direction = (
                    d_val
                    if isinstance(d_val, ObjectiveDirection)
                    else ObjectiveDirection(str(d_val).lower())
                )
                parsed_objectives.append(ObjectiveSpec(metric_name=m_name, direction=direction))
        else:
            parsed_objectives = list(objectives)

        scores_by_scenario: dict[str, dict[str, float]] = {}
        for run in all_runs:
            scores_by_scenario[run.scenario.name] = {
                m_name: results_by_scenario[run.scenario.name][m_name].mean
                for m_name in results_by_scenario[run.scenario.name]
            }

        pareto_frontier = compute_pareto_frontier(
            scores=scores_by_scenario,
            objectives=parsed_objectives,
        )

    # User objective ranking (only performed if user provides an objective callable)
    rankings: list[tuple[str, float]] | None = None
    if objective is not None:
        scenario_scores: list[tuple[str, float]] = []
        for run in all_runs:
            scn_name = run.scenario.name
            means = {
                m: results_by_scenario[scn_name][m].mean for m in results_by_scenario[scn_name]
            }
            score = float(objective(means))
            scenario_scores.append((scn_name, score))
        # Sort descending by score
        scenario_scores.sort(key=lambda item: item[1], reverse=True)
        rankings = scenario_scores

    return ScenarioComparison(
        baseline_name=baseline_name,
        results_by_scenario=results_by_scenario,
        deltas=deltas,
        bootstrap_deltas=bootstrap_deltas,
        pareto_frontier=pareto_frontier,
        rankings=rankings,
        estimand=estimand,
    )
