"""Structured reporting for decision-focused scenario discovery and multi-robustness analysis.

Conforms to Track T7: Reporting & Visualization (Renderer-neutral, pure Python/Rich optional).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import normalize_canonical_value
from ewm_engine.decision.discovery import ScenarioDiscoveryResult
from ewm_engine.decision.efe import ExpectedFreeEnergyScore
from ewm_engine.decision.robustness import DecisionComparisonResult


class DecisionReport(BaseModel):
    """Structured decision analysis report synthesizing vulnerability discovery and multi-robustness rankings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(default="EWM Decision Analysis Report", description="Report title.")
    world_name: str = Field(description="Name of the evaluated world model.")
    uncertainty_space_name: str = Field(
        description="Identifier of the sampled parameter uncertainty space."
    )
    sample_count: int = Field(
        description="Number of uncertainty scenarios evaluated per candidate."
    )
    candidates: tuple[str, ...] = Field(
        description="List of candidate actions or strategies evaluated."
    )
    robustness: DecisionComparisonResult = Field(
        description="Multi-metric robustness evaluation and disagreement audit."
    )
    vulnerabilities: dict[str, ScenarioDiscoveryResult] = Field(
        description="Scenario discovery vulnerability results indexed by candidate strategy."
    )
    efe_scores: dict[str, ExpectedFreeEnergyScore] | None = Field(
        default=None,
        description="Optional active inference Expected Free Energy scores.",
    )
    abstention_decision: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured ValidityLimitAbstentionDecision ('DO_NOT_USE_ME_HERE' vs 'WITHIN_DECLARED_VALIDITY_LIMITS').",
    )
    fingerprint: str = Field(
        description="Canonical SHA-256 fingerprint of the complete decision report."
    )

    def summary_table(self) -> str:
        """Render a clean text table comparing candidate decisions across all robustness metrics."""
        lines: list[str] = []
        lines.append("=" * 95)
        lines.append(f" DECISION ANALYSIS REPORT: {self.title.upper()}")
        lines.append(
            f" World: {self.world_name} | Uncertainty Space: {self.uncertainty_space_name} (N={self.sample_count})"
        )
        lines.append("=" * 95)

        # 1. Status & Consensus
        if self.robustness.refusal_to_rank:
            lines.append("\n [!] EPISTEMIC STATUS: REFUSAL TO RANK (CRITERIA DISAGREEMENT)")
            lines.append(
                "     Under deep uncertainty, different robustness metrics favor different actions."
            )
            lines.append(f"     {self.robustness.refusal_rationale}")
        else:
            lines.append(f"\n [*] DOMINANT ACTION: '{self.robustness.dominant_action}'")
            lines.append(
                "     All evaluated robustness criteria unanimously agree on this candidate."
            )

        # 2. Metric Winners Table
        lines.append("\n--- METRIC ADJUDICATION ---")
        lines.append(
            f"{'Robustness Metric':<28} | {'Winning Candidate':<25} | {'Decision Posture'}"
        )
        lines.append("-" * 95)
        lines.append(
            f"{'Min-Max Regret':<28} | {self.robustness.metric_winners.get('min_max_regret', 'N/A'):<25} | Worst-case regret minimization (Savage)"
        )
        lines.append(
            f"{'CVaR(90) Regret':<28} | {self.robustness.metric_winners.get('cvar_regret_90', 'N/A'):<25} | Tail regret minimization (Rockafellar)"
        )
        lines.append(
            f"{'Satisficing Rate':<28} | {self.robustness.metric_winners.get('satisficing', 'N/A'):<25} | Domain target threshold meeting (Starr)"
        )
        lines.append(
            f"{'Expected Regret (Laplace)':<28} | {self.robustness.metric_winners.get('mean_regret', 'N/A'):<25} | Neutral average regret across futures"
        )

        # 3. Detailed Scores Table
        lines.append("\n--- DETAILED CANDIDATE ROBUSTNESS SCORES ---")
        header = f"{'Candidate Action':<20} | {'Max Regret':<12} | {'CVaR(90)':<12} | {'Satisficing':<12} | {'Mean Regret':<12} | {'Mean Outcome':<12}"
        lines.append(header)
        lines.append("-" * len(header))
        for act_id in self.candidates:
            s = self.robustness.scores.get(act_id)
            if s:
                lines.append(
                    f"{act_id:<20} | {s.max_regret:<12.2f} | {s.cvar_regret_90:<12.2f} | "
                    f"{s.satisficing_rate * 100.0:<11.1f}% | {s.mean_regret:<12.2f} | {s.mean_outcome:<12.2f}"
                )

        # 4. Vulnerability Discovery (PRIM / CART)
        lines.append("\n--- SCENARIO DISCOVERY: VULNERABILITY BOXES (WHERE ACTIONS BREAK) ---")
        for act_id, disc in self.vulnerabilities.items():
            lines.append(
                f"Candidate: '{act_id}' (Total Failures: {disc.total_failures}/{disc.total_samples}):"
            )
            if disc.primary_box and disc.total_failures > 0:
                b = disc.primary_box
                lines.append(f"  * Primary Vulnerability Box ({b.box_id}):")
                lines.append(f"    - Rule: {b.format_rule()}")
                lines.append(
                    f"    - Coverage (Recall): {b.coverage * 100.0:.1f}% of failures occur in this subspace"
                )
                lines.append(
                    f"    - Density (Purity):  {b.density * 100.0:.1f}% failure rate inside box"
                )
                lines.append(f"    - Significance:      quasi-p = {b.quasi_p_value:.4e}")
            else:
                lines.append(
                    "  * No significant vulnerability subspace identified (robust across evaluated uncertainty)."
                )

        # 5. Optional Active Inference EFE Scores
        if self.efe_scores:
            lines.append("\n--- ACTIVE INFERENCE: EXPECTED FREE ENERGY (EFE) ---")
            efe_header = f"{'Candidate Action':<20} | {'Total EFE':<12} | {'Pragmatic (Ext)':<16} | {'Epistemic (Info)':<16} | {'Weight (Beta)'}"
            lines.append(efe_header)
            lines.append("-" * len(efe_header))
            for act_id in self.candidates:
                e = self.efe_scores.get(act_id)
                if e:
                    lines.append(
                        f"{act_id:<20} | {e.total_efe:<12.3f} | {e.pragmatic_value:<16.3f} | "
                        f"{e.epistemic_value:<16.3f} | {e.weight_epistemic:.2f}"
                    )

        lines.append("=" * 95)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render report in GitHub Flavored Markdown."""
        md: list[str] = []
        md.append(f"# {self.title}")
        md.append(
            f"**World:** `{self.world_name}` | **Uncertainty Space:** `{self.uncertainty_space_name}` | **Scenarios:** `{self.sample_count}`\n"
        )

        if self.robustness.refusal_to_rank:
            md.append("> [!WARNING]")
            md.append("> **EPISTEMIC REFUSAL TO RANK: METRICS DISAGREE**")
            md.append(f"> {self.robustness.refusal_rationale}\n")
        else:
            md.append("> [!NOTE]")
            md.append(
                f"> **UNANIMOUS WINNER:** Candidate `{self.robustness.dominant_action}` is favored across all criteria.\n"
            )

        md.append("## Robustness Comparison\n")
        md.append(
            "| Candidate | Max Regret | CVaR(90) Regret | Satisficing Rate | Mean Regret | Mean Outcome |"
        )
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for act_id in self.candidates:
            s = self.robustness.scores.get(act_id)
            if s:
                md.append(
                    f"| `{act_id}` | {s.max_regret:.2f} | {s.cvar_regret_90:.2f} | {s.satisficing_rate * 100.0:.1f}% | {s.mean_regret:.2f} | {s.mean_outcome:.2f} |"
                )

        md.append("\n## Scenario Discovery (Where Strategies Break)\n")
        for act_id, disc in self.vulnerabilities.items():
            md.append(f"### Strategy: `{act_id}`")
            if disc.primary_box and disc.total_failures > 0:
                b = disc.primary_box
                md.append(f"- **Vulnerability Predicate:** `{b.format_rule()}`")
                md.append(
                    f"- **Coverage:** {b.coverage * 100.0:.1f}% | **Density:** {b.density * 100.0:.1f}% | **Quasi-p:** `{b.quasi_p_value:.3e}`"
                )
            else:
                md.append("- *No vulnerability box identified.*")

        return "\n".join(md)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to JSON string."""
        return json.dumps(normalize_canonical_value(self.model_dump()), indent=indent)

    def _repr_html_(self) -> str:
        """Render a rich, styled HTML card representation for interactive notebooks."""
        short_fp = self.fingerprint[:12] if self.fingerprint else "unknown"
        if self.robustness.refusal_to_rank:
            status_badge = "<span style='background: #fef2f2; color: #991b1b; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px;'>Refusal to Rank (Disagreement)</span>"
        else:
            status_badge = f"<span style='background: #f0fdf4; color: #166534; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px;'>Consensus: {self.robustness.dominant_action}</span>"

        cand_rows: list[str] = []
        for act_id in self.candidates:
            s = self.robustness.scores.get(act_id)
            if s:
                cand_rows.append(
                    f"<tr>"
                    f"<td style='padding: 6px 12px; font-weight: 600;'><code>{act_id}</code></td>"
                    f"<td style='padding: 6px 12px; text-align: right; font-family: monospace;'>{s.max_regret:.2f}</td>"
                    f"<td style='padding: 6px 12px; text-align: right; font-family: monospace;'>{s.cvar_regret_90:.2f}</td>"
                    f"<td style='padding: 6px 12px; text-align: right; font-family: monospace;'>{s.satisficing_rate * 100.0:.1f}%</td>"
                    f"<td style='padding: 6px 12px; text-align: right; font-family: monospace;'>{s.mean_outcome:.2f}</td>"
                    f"</tr>"
                )

        table_html = (
            f"<table style='width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px;'>"
            f"<thead><tr style='border-bottom: 2px solid #cbd5e1; text-align: left; color: #475569; font-size: 11px; text-transform: uppercase;'>"
            f"<th style='padding: 6px 12px;'>Candidate</th><th style='padding: 6px 12px; text-align: right;'>Max Regret</th>"
            f"<th style='padding: 6px 12px; text-align: right;'>CVaR(90)</th><th style='padding: 6px 12px; text-align: right;'>Satisficing</th>"
            f"<th style='padding: 6px 12px; text-align: right;'>Mean Outcome</th></tr></thead>"
            f"<tbody>{''.join(cand_rows)}</tbody></table>"
        )

        return (
            f"<div style='font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; "
            f"border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px; background: #ffffff; "
            f"box-shadow: 0 1px 3px rgba(0,0,0,0.05); max-width: 720px; margin: 8px 0; color: #1e293b;'>"
            f"<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f1f5f9; padding-bottom: 10px;'>"
            f"<div>"
            f"<span style='font-weight: 700; font-size: 15px; color: #0f172a;'>DecisionReport</span>"
            f"<span style='background: #e0e7ff; color: #4338ca; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 12px; margin-left: 8px;'>{self.world_name}</span>"
            f"</div>"
            f"{status_badge}"
            f"</div>"
            f"<div style='display: flex; gap: 16px; margin: 10px 0; font-size: 12px; color: #64748b;'>"
            f"<div><strong>Uncertainty Space:</strong> {self.uncertainty_space_name}</div>"
            f"<div><strong>Samples:</strong> N={self.sample_count}</div>"
            f"<div><strong>Fingerprint:</strong> <code>{short_fp}…</code></div>"
            f"</div>"
            f"{table_html}"
            f"</div>"
        )
