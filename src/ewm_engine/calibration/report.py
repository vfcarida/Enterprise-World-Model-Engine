"""Reporting models and export formats for simulation calibration and certified coverage."""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.calibration.certification import CoverageCertificate
from ewm_engine.calibration.sbi import PosteriorDistribution
from ewm_engine.provenance.evidence import EvidenceLevel


class CalibrationReport(BaseModel):
    """Audited, reproducible report of Bayesian simulation calibration and certified coverage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    calibration_id: str = Field(description="Unique identifier for this calibration execution.")
    spec_name: str = Field(description="Name of the calibrated WorldSpec.")
    timestamp: float = Field(
        default_factory=time.time, description="Unix timestamp of calibration."
    )
    target_data_hash: str = Field(
        description="Cryptographic SHA-256 hash of observed target series."
    )
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.PREDICTIVE,
        description="Epistemic tier. Strictly defaults to PREDICTIVE (calibrated != causally identified).",
    )
    priors: list[dict[str, Any]] = Field(description="Declared prior bounds and distributions.")
    posterior: PosteriorDistribution = Field(
        description="Recovered parameter posterior distribution p(theta | x_obs)."
    )
    certificate: CoverageCertificate = Field(
        description="Mandatory coverage certificate (TARP, Posterior-SBC, misspecification)."
    )
    posterior_predictive: dict[str, Any] = Field(
        description="Predictive trajectories (mean, 90% credible band) vs observed target series."
    )
    misspecification_diagnostic: dict[str, Any] = Field(
        description="Diagnostic metrics evaluating simulator generative support consistency."
    )
    abstention_decision: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured ValidityLimitAbstentionDecision ('DO_NOT_USE_ME_HERE' when uncertified or misspecified).",
    )
    run_manifest: dict[str, Any] = Field(
        default_factory=dict, description="Deterministic run manifest for exact reproduction."
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_console_table(self) -> str:
        """Alias for summary_table."""
        return self.summary_table()

    def summary_table(self) -> str:
        """Render human-readable console table of calibrated parameters and certification status."""
        lines: list[str] = []
        lines.append(f"Calibration Report: {self.calibration_id} (Spec: {self.spec_name})")
        lines.append("=" * 72)

        color_marker = "[OK]" if self.certificate.is_certified else "[REFUSED]"
        lines.append(f"Coverage Status:       {color_marker} {self.certificate.status}")
        lines.append(
            f"Decision Usable:       {'YES' if self.certificate.decision_usable else 'NO (Refused for decision use)'}"
        )
        lines.append(f"TARP Max Discrepancy:  {self.certificate.tarp_max_discrepancy:.4f}")
        lines.append(f"TARP KS p-value:       {self.certificate.tarp_ks_pvalue:.4f}")
        lines.append(f"Posterior-SBC p-value: {self.certificate.sbc_rank_ks_pvalue:.4f}")
        lines.append(
            f"Misspecification:      {self.certificate.misspecification_score:.2f} std ({'DETECTED' if self.certificate.is_misspecified else 'LOW'})"
        )

        if self.certificate.refusal_reason:
            lines.append(f"\nRefusal Rationale:\n  {self.certificate.refusal_reason}")

        lines.append("\nCalibrated Parameters (Posterior Distribution):")
        lines.append("-" * 72)
        lines.append(f"{'Parameter':<24} {'Mean':<10} {'Std':<10} {'5% CI':<10} {'95% CI':<10}")
        lines.append("-" * 72)

        for p_name in self.posterior.parameter_names:
            m = self.posterior.mean.get(p_name, 0.0)
            s = self.posterior.std.get(p_name, 0.0)
            ci = self.posterior.credible_intervals_90.get(p_name, (0.0, 0.0))
            lines.append(f"{p_name:<24} {m:<10.3f} {s:<10.3f} {ci[0]:<10.3f} {ci[1]:<10.3f}")

        lines.append("-" * 72)
        lines.append(
            f"Epistemic Note: {self.evidence_level.upper()} tier. Calibrated != identified."
        )
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Render report as formatted GitHub-flavored Markdown."""
        md: list[str] = []
        md.append(f"# Calibration Report: {self.calibration_id}")
        md.append(f"\n- **WorldSpec:** `{self.spec_name}`")
        md.append(f"- **Data Hash:** `{self.target_data_hash[:16]}...`")
        md.append(f"- **Status:** `{self.certificate.status}`")
        md.append(f"- **Decision Usable:** `{self.certificate.decision_usable}`")
        md.append(f"- **Evidence Level:** `{self.evidence_level}` (Observational calibrated)")

        if self.certificate.refusal_reason:
            md.append(
                f"\n> [!CAUTION]\n> **Refused for Decision Use:** {self.certificate.refusal_reason}\n"
            )

        md.append("\n## 1. Coverage Certification (TARP & Posterior-SBC)")
        md.append("\n| Metric | Value | Threshold | Verdict |")
        md.append("| :--- | :--- | :--- | :--- |")
        disc_th = self.certificate.max_discrepancy_threshold
        ks_th = self.certificate.ks_pvalue_threshold
        md.append(
            f"| TARP Max Discrepancy | `{self.certificate.tarp_max_discrepancy:.4f}` | `{disc_th:.4f}` | {'PASS' if self.certificate.tarp_max_discrepancy <= disc_th else 'FAIL'} |"
        )
        md.append(
            f"| TARP KS Test p-value | `{self.certificate.tarp_ks_pvalue:.4f}` | `{ks_th:.4f}` | {'PASS' if self.certificate.tarp_ks_pvalue >= ks_th else 'FAIL'} |"
        )
        md.append(
            f"| SBC Rank Uniformity p | `{self.certificate.sbc_rank_ks_pvalue:.4f}` | `{ks_th:.4f}` | {'PASS' if self.certificate.sbc_rank_ks_pvalue >= ks_th else 'FAIL'} |"
        )
        md.append(
            f"| Misspecification Score | `{self.certificate.misspecification_score:.2f} std` | `3.50 std` | {'PASS' if not self.certificate.is_misspecified else 'FAIL'} |"
        )

        md.append("\n## 2. Posterior Distribution over Parameters")
        md.append("\n| Parameter | Posterior Mean | Std | Median | 90% Credible Interval |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for p_name in self.posterior.parameter_names:
            m = self.posterior.mean.get(p_name, 0.0)
            s = self.posterior.std.get(p_name, 0.0)
            med = self.posterior.median.get(p_name, 0.0)
            ci = self.posterior.credible_intervals_90.get(p_name, (0.0, 0.0))
            md.append(
                f"| `{p_name}` | {m:.3f} | {s:.3f} | {med:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |"
            )

        md.append("\n## 3. Epistemic Integrity")
        md.append("\n> [!NOTE]")
        md.append(
            "> **Calibrated $\\neq$ Causally Identified:** A calibrated simulation accurately reproduces observed time series data under declared assumptions, but does not identify unconfounded counterfactual effects without exogenous variation or structural backdoor admissibility (Structural Causal Model)."
        )

        return "\n".join(md)
