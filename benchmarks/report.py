"""Machine-readable and human-readable reporting for scientific benchmark suites.

Produces structured JSON and comparative tables without automated 'winner' declarations,
enforcing epistemic humility and provenance tracking.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.protocol import BenchmarkResult


class BenchmarkSuiteReport(BaseModel):
    """Aggregated report of multiple scientific benchmark family runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite_name: str = Field(default="EWM_Scientific_Benchmark_Suite")
    timestamp: str = Field(description="ISO-8601 UTC timestamp of execution.")
    results: list[BenchmarkResult] = Field(default_factory=list)
    epistemic_disclaimer: str = Field(
        default=(
            "======================================================================\n"
            "EPISTEMIC DISCLAIMER: SCIENTIFIC BENCHMARK INSTRUMENT\n"
            "----------------------------------------------------------------------\n"
            "These benchmarks evaluate models against artificial, synthetic worlds.\n"
            "Results reflect properties of specific synthetic setups only, NOT\n"
            "empirical real-world superiority. The engine forbids auto-declaring\n"
            "a 'winning' model without user-specified utility and risk preferences.\n"
            "======================================================================"
        )
    )

    def to_json(self, indent: int = 2) -> str:
        """Export report as machine-readable JSON string."""
        return json.dumps(self.model_dump(), indent=indent)

    def save_json(self, path: Path | str) -> None:
        """Save report to a JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def to_text_table(self) -> str:
        """Generate a formatted human-readable ASCII comparison table."""
        lines = [
            self.epistemic_disclaimer,
            "",
            f"Benchmark Suite: {self.suite_name} (Executed: {self.timestamp})",
            "=" * 92,
            f"{'Benchmark Family':<22} | {'Model Name':<28} | {'Invariants':<10} | {'Key Metric / Gap':<22}",
            "-" * 92,
        ]

        for b_res in self.results:
            family_str = b_res.family.value
            for idx, m_eval in enumerate(b_res.models_evaluated):
                f_col = family_str if idx == 0 else ""
                inv_status = (
                    "PASS" if m_eval.invariants_passed else f"FAIL ({m_eval.invariant_violations})"
                )

                # Select most informative metric
                if "interventional_gap" in m_eval.metrics:
                    metric_str = f"Gap: {m_eval.metrics['interventional_gap']:+.4f} ({m_eval.metrics.get('gap_ratio', 1.0):.1f}x)"
                elif "rule_shift_degradation" in m_eval.metrics:
                    metric_str = f"PostMAE: {m_eval.metrics.get('post_shift_mae', 0.0):.4f} ({m_eval.metrics.get('degradation_ratio', 1.0):.1f}x)"
                elif "stress_violation_rate" in m_eval.metrics:
                    metric_str = f"ViolRate: {m_eval.metrics['stress_violation_rate'] * 100:.1f}%"
                elif "drift_rate" in m_eval.metrics:
                    metric_str = f"Drift: {m_eval.metrics['drift_rate']:+.4f}/step"
                elif "final_unserved_demand" in m_eval.metrics:
                    metric_str = f"Unserved: {m_eval.metrics['final_unserved_demand']:.1f}"
                else:
                    metric_str = f"Divergence: {m_eval.divergence_or_gap:.4f}"

                lines.append(
                    f"{f_col:<22} | {m_eval.model_name:<28} | {inv_status:<10} | {metric_str:<22}"
                )
            lines.append("-" * 92)

        lines.extend(
            [
                "",
                "Key Scientific Phenomena Surfaced:",
            ]
        )
        for b_res in self.results:
            lines.append(f"  [{b_res.family.value}] {b_res.phenomenon_surfaced}")

        return "\n".join(lines)

    def to_markdown_table(self) -> str:
        """Generate a GitHub-flavored Markdown table of benchmark results."""
        lines = [
            f"### Scientific Benchmark Results: `{self.suite_name}`",
            "",
            "> **Epistemic Note:** Results reflect synthetic research probes on artificial environments. "
            "No automated winner is declared.",
            "",
            "| Benchmark Family | Model Name | Invariants | Primary Metric / Gap |",
            "|:-----------------|:-----------|:-----------|:---------------------|",
        ]

        for b_res in self.results:
            family_str = b_res.family.value
            for idx, m_eval in enumerate(b_res.models_evaluated):
                f_col = f"**{family_str}**" if idx == 0 else '""'
                inv_status = (
                    "PASS" if m_eval.invariants_passed else f"FAIL ({m_eval.invariant_violations})"
                )

                if "interventional_gap" in m_eval.metrics:
                    metric_str = f"Gap: {m_eval.metrics['interventional_gap']:+.4f}"
                elif "rule_shift_degradation" in m_eval.metrics:
                    metric_str = f"Degradation: {m_eval.metrics.get('degradation_ratio', 1.0):.2f}x"
                elif "stress_violation_rate" in m_eval.metrics:
                    metric_str = (
                        f"Stress Violations: {m_eval.metrics['stress_violation_rate'] * 100:.1f}%"
                    )
                elif "drift_rate" in m_eval.metrics:
                    metric_str = f"Drift: {m_eval.metrics['drift_rate']:+.4f}/step"
                elif "final_unserved_demand" in m_eval.metrics:
                    metric_str = f"Unserved Demand: {m_eval.metrics['final_unserved_demand']:.1f}"
                else:
                    metric_str = f"{m_eval.divergence_or_gap:.4f}"

                lines.append(f"| {f_col} | `{m_eval.model_name}` | {inv_status} | {metric_str} |")

        return "\n".join(lines)
