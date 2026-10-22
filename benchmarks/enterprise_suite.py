"""Enterprise World-Model Evaluation & Failure-Mode Benchmark Suite Runner (ADR-054).

Executes the twelve independent benchmark families (A through L) across the four
mandatory baseline paradigms (`rules-only`, `learned-only`, `static hybrid`,
`hybrid + runtime context discovery`), enforces the statistical and shift protocols,
verifies zero train/eval data leakage, and auto-generates machine-readable JSON,
Markdown tables, and vector SVG figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.families.enterprise_families import build_all_enterprise_benchmarks  # noqa: E402
from benchmarks.protocol import (  # noqa: E402
    BENCHMARK_SPEC_VERSION,
    METRIC_DEFINITIONS,
    METRIC_IMPLEMENTATION_VERSION,
    BaselineParadigm,
    EnterpriseBenchmarkFamilyResult,
    MetricDefinition,
    MetricDimension,
)


class EnterpriseBenchmarkSuiteReport(BaseModel):
    """Machine-readable report for the 12-family Enterprise World-Model Benchmark Suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite_name: str = Field(default="EWM_Enterprise_World_Model_Benchmark_Suite")
    benchmark_spec_version: str = Field(default=BENCHMARK_SPEC_VERSION)
    metric_implementation_version: str = Field(default=METRIC_IMPLEMENTATION_VERSION)
    master_seed: int = Field(description="Master seed governing all 12 benchmark families.")
    timestamp: str = Field(description="ISO-8601 UTC timestamp of suite execution.")
    suite_fingerprint: str = Field(
        description="Cryptographic SHA-256 fingerprint sealing all 12 family outputs and datasets."
    )
    metric_definitions: dict[MetricDimension, MetricDefinition] = Field(
        default_factory=lambda: dict(METRIC_DEFINITIONS),
        description="Formal mathematical definitions for the 13 orthogonal evaluation dimensions.",
    )
    families: tuple[EnterpriseBenchmarkFamilyResult, ...] = Field(
        description="Independent evaluation results for benchmark families A through L."
    )
    epistemic_disclaimer: str = Field(
        default=(
            "====================================================================================\n"
            "EPISTEMIC & SCIENTIFIC BENCHMARK DISCLAIMER (ADR-054)\n"
            "------------------------------------------------------------------------------------\n"
            "1. PURPOSE: This suite is a scientific measurement instrument designed to expose\n"
            "   failure modes across symbolic, learned, static-hybrid, and context-aware hybrid\n"
            "   world models on controlled synthetic enterprise scenarios.\n"
            "2. NO COLLAPSED SCORE: The 13 metric dimensions are orthogonal and are NEVER\n"
            "   collapsed into a single scalar 'winner' score.\n"
            "3. HONEST LIMITATIONS: Even `hybrid + runtime context discovery` incurs transient\n"
            "   post-shift error during audit telemetry lag (Family D), bounded filtering lag\n"
            "   under partial observability (Family E), raw prediction failure on unmodeled OOD\n"
            "   physics requiring abstention (Family K), and higher compute latency/memory.\n"
            "===================================================================================="
        )
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize the complete suite report to deterministic JSON."""
        return json.dumps(self.model_dump(mode="json"), indent=indent, sort_keys=True)

    def save_json(self, path: Path | str) -> None:
        """Write machine-readable JSON artifact to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(indent=2) + "\n", encoding="utf-8")

    def to_text_table(self) -> str:
        """Render a comprehensive human-readable ASCII summary table for console inspection."""
        lines: list[str] = [
            self.epistemic_disclaimer,
            "",
            f"Suite: {self.suite_name} | Spec v{self.benchmark_spec_version} | Seed: {self.master_seed}",
            f"Suite Fingerprint: {self.suite_fingerprint}",
            "=" * 118,
            f"{'Fam':<4} | {'Family Name':<32} | {'Baseline Paradigm':<34} | {'Primary Dimension':<24} | {'Value (95% CI / det)':<18}",
            "-" * 118,
        ]
        for fam in self.families:
            p_dim = fam.primary_dimensions[0]
            for idx, b_rec in enumerate(fam.baselines):
                f_code = fam.family_code if idx == 0 else ""
                f_name = fam.family.value if idx == 0 else ""
                stat = b_rec.dimension_summaries[p_dim]
                val_str = stat.format_compact().replace("±", "+/-")
                lines.append(
                    f"{f_code:<4} | {f_name:<32} | {b_rec.paradigm.value:<34} | {p_dim.value:<24} | {val_str:<18}"
                )
            lines.append("-" * 118)

        lines.append("")
        lines.append("Leakage Verification Summary:")
        all_leak_ok = all(f.leakage_certificate.passed for f in self.families)
        lines.append(
            f"  Hard Train/Eval Temporal, Seed & Config Boundaries: {'PASS (12/12 families verified)' if all_leak_ok else 'FAIL'}"
        )
        return "\n".join(lines)

    def generate_markdown_primary_table(self) -> str:
        """Auto-generate the Family A-L x 4 Baselines Markdown table from the artifact."""
        lines = [
            "| Code | Benchmark Family | Oracle | Primary Metric | `rules-only` | `learned-only` | `static hybrid` | `hybrid + runtime context discovery` |",
            "|:----:|:-----------------|:------:|:---------------|:------------:|:--------------:|:---------------:|:------------------------------------:|",
        ]
        for fam in self.families:
            oracle_tag = (
                "Exact (`det`)"
                if fam.is_deterministic_oracle
                else f"Stochastic ($M={len(fam.reproducibility.evaluated_seeds)}$)"
            )
            p_dim = fam.primary_dimensions[0]
            by_p = {
                b.paradigm: b.dimension_summaries[p_dim].format_compact() for b in fam.baselines
            }
            lines.append(
                f"| **{fam.family_code}** | `{fam.family.value}` | {oracle_tag} | `{p_dim.value}` | "
                f"{by_p[BaselineParadigm.RULES_ONLY]} | "
                f"{by_p[BaselineParadigm.LEARNED_ONLY]} | "
                f"{by_p[BaselineParadigm.STATIC_HYBRID]} | "
                f"**{by_p[BaselineParadigm.CONTEXTUAL_HYBRID]}** |"
            )
        return "\n".join(lines)

    def generate_markdown_dimensions_table(self) -> str:
        """Auto-generate the 13 Orthogonal Metric Dimensions profile table across all 4 baselines."""
        lines = [
            "| Metric Dimension | Direction | Unit | `rules-only` | `learned-only` | `static hybrid` | `hybrid + runtime context discovery` |",
            "|:-----------------|:---------:|:-----|:------------:|:--------------:|:---------------:|:------------------------------------:|",
        ]
        for dim in MetricDimension:
            m_def = self.metric_definitions[dim]
            dir_str = "Higher ↑" if m_def.direction == "higher_is_better" else "Lower ↓"
            means_by_p: dict[BaselineParadigm, float] = {}
            for p in BaselineParadigm:
                vals = [
                    b.dimension_summaries[dim].mean
                    for f in self.families
                    for b in f.baselines
                    if b.paradigm == p
                ]
                means_by_p[p] = sum(vals) / max(1, len(vals))

            lines.append(
                f"| `{dim.value}` | {dir_str} | {m_def.unit} | "
                f"{means_by_p[BaselineParadigm.RULES_ONLY]:.4f} | "
                f"{means_by_p[BaselineParadigm.LEARNED_ONLY]:.4f} | "
                f"{means_by_p[BaselineParadigm.STATIC_HYBRID]:.4f} | "
                f"{means_by_p[BaselineParadigm.CONTEXTUAL_HYBRID]:.4f} |"
            )
        return "\n".join(lines)

    def generate_markdown_shift_table(self) -> str:
        """Auto-generate the Shift Protocol table for configuration and deployment shift families."""
        lines = [
            "| Family | Baseline Paradigm | Train Context | Eval Context | Audit Delay | Pre-Shift MAE | Peak Shift MAE | Steady Post-Shift MAE | Steps to Recovery | Recovered in Budget |",
            "|:-------|:------------------|:-------------:|:------------:|:-----------:|:-------------:|:--------------:|:---------------------:|:-----------------:|:-------------------:|",
        ]
        for fam in self.families:
            for b in fam.baselines:
                if b.shift_protocol is not None:
                    sp = b.shift_protocol
                    rec_str = (
                        str(sp.steps_until_recovery)
                        if sp.steps_until_recovery is not None
                        else "Unrecovered (∞)"
                    )
                    bud_str = "PASS" if sp.recovered_within_budget else "FAIL"
                    lines.append(
                        f"| **{fam.family_code}** (`{fam.family.value}`) | `{b.paradigm.value}` | "
                        f"`{sp.training_context_version}` | `{sp.evaluation_context_version}` | "
                        f"{sp.audit_discovery_delay_steps} steps | {sp.pre_shift_mae:.3f} | "
                        f"{sp.peak_post_shift_mae:.3f} | {sp.steady_state_post_shift_mae:.3f} | "
                        f"{rec_str} | {bud_str} |"
                    )
        return "\n".join(lines)

    def generate_markdown_failure_modes_table(self) -> str:
        """Auto-generate the Failure Mode Exposure table across families A-L."""
        lines = [
            "| Family | `rules-only` Failure Mode | `learned-only` Failure Mode | `static hybrid` Failure Mode | `hybrid + runtime context discovery` Limitation / Behavior |",
            "|:------:|:--------------------------|:----------------------------|:-----------------------------|:-----------------------------------------------------------|",
        ]
        for fam in self.families:
            fm_map: dict[BaselineParadigm, str] = {}
            for b in fam.baselines:
                fm_map[b.paradigm] = (
                    "; ".join(b.failure_modes_exposed)
                    if b.failure_modes_exposed
                    else "Optimal in-regime behavior (0 violations)."
                )
            lines.append(
                f"| **{fam.family_code}** | {fm_map[BaselineParadigm.RULES_ONLY]} | "
                f"{fm_map[BaselineParadigm.LEARNED_ONLY]} | "
                f"{fm_map[BaselineParadigm.STATIC_HYBRID]} | "
                f"{fm_map[BaselineParadigm.CONTEXTUAL_HYBRID]} |"
            )
        return "\n".join(lines)

    def generate_markdown_reproducibility_table(self) -> str:
        """Auto-generate the Reproducibility & Non-Leakage Certificate table."""
        lines = [
            "| Code | Family | Train Seed | Eval Seeds | State Overlap | Temporal Gate | Config Gate | Dataset SHA-256 (prefix) | Output SHA-256 (prefix) |",
            "|:----:|:-------|:----------:|:----------:|:-------------:|:-------------:|:-----------:|:------------------------:|:-----------------------:|",
        ]
        for fam in self.families:
            lc = fam.leakage_certificate
            rp = fam.reproducibility
            seeds_str = ",".join(str(s)[:6] for s in rp.evaluated_seeds[:2])
            if len(rp.evaluated_seeds) > 2:
                seeds_str += f",+{len(rp.evaluated_seeds) - 2}"
            lines.append(
                f"| **{fam.family_code}** | `{fam.family.value}` | `{lc.train_partition_seed}` | "
                f"`{seeds_str}` | {lc.state_fingerprint_overlap_count} | "
                f"{'PASS' if lc.temporal_boundary_verified else 'FAIL'} | "
                f"{'PASS' if lc.config_boundary_verified else 'FAIL'} | "
                f"`{rp.dataset_scenario_fingerprint[:12]}` | `{rp.output_fingerprint[:12]}` |"
            )
        return "\n".join(lines)

    def generate_svg_heatmap(self) -> str:
        """Auto-generate an accessible vector SVG heatmap of the 12 families x 4 baselines."""
        width = 880
        row_h = 34
        header_h = 90
        height = header_h + len(self.families) * row_h + 45

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title-hm desc-hm">',
            '  <title id="title-hm">Enterprise World-Model Benchmark Suite (Families A-L) Primary Metric Comparison</title>',
            '  <desc id="desc-hm">Comparative matrix across 12 independent benchmark families and 4 baseline paradigms generated directly from enterprise_benchmark_suite.json.</desc>',
            '  <rect width="100%" height="100%" fill="#0f172a" rx="8"/>',
            '  <text x="24" y="34" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="16" font-weight="bold">Enterprise World-Model Evaluation Suite (A-L): Primary Metric Profile</text>',
            f'  <text x="24" y="54" fill="#94a3b8" font-family="monospace" font-size="11">Spec v{self.benchmark_spec_version} | Seed={self.master_seed} | Suite SHA-256={self.suite_fingerprint[:16]}...</text>',
        ]

        col_x = [310, 445, 580, 715]
        col_labels = ["rules-only", "learned-only", "static hybrid", "contextual hybrid"]
        for cx, lbl in zip(col_x, col_labels, strict=True):
            svg_lines.append(
                f'  <text x="{cx + 60}" y="80" fill="#cbd5e1" font-family="system-ui, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">{lbl}</text>'
            )

        for row_idx, fam in enumerate(self.families):
            y = header_h + row_idx * row_h
            p_dim = fam.primary_dimensions[0]
            direction = self.metric_definitions[p_dim].direction
            svg_lines.append(
                f'  <text x="24" y="{y + 21}" fill="#e2e8f0" font-family="monospace" font-size="11">[{fam.family_code}] {fam.family.value[:26]}</text>'
            )

            for c_idx, b in enumerate(fam.baselines):
                cx = col_x[c_idx]
                val = b.dimension_summaries[p_dim].mean
                # Normalize color intensity for visual inspection
                if direction == "higher_is_better":
                    norm_score = max(0.0, min(1.0, val))
                else:
                    norm_score = 1.0 / (1.0 + max(0.0, val) / 5.0)

                if norm_score >= 0.80:
                    bg = "#065f46"
                    fg = "#ecfdf5"
                elif norm_score >= 0.50:
                    bg = "#1e3a8a"
                    fg = "#eff6ff"
                else:
                    bg = "#7f1d1d"
                    fg = "#fef2f2"

                svg_lines.append(
                    f'  <rect x="{cx}" y="{y + 4}" width="124" height="{row_h - 6}" fill="{bg}" rx="4"/>'
                )
                svg_lines.append(
                    f'  <text x="{cx + 62}" y="{y + 21}" fill="{fg}" font-family="monospace" font-size="11" text-anchor="middle">{val:.3f}</text>'
                )

        svg_lines.append(
            f'  <text x="24" y="{height - 14}" fill="#64748b" font-family="system-ui, sans-serif" font-size="10">Auto-generated from docs/results/artifacts/enterprise_benchmark_suite.json — Green: strong in-dimension performance; Red: exposed failure mode.</text>'
        )
        svg_lines.append("</svg>")
        return "\n".join(svg_lines)

    def generate_svg_shift_recovery(self) -> str:
        """Auto-generate an accessible vector SVG chart showing Family D deployment-shift error & recovery."""
        fam_d = next(
            (f for f in self.families if f.family_code == "D"),
            self.families[0],
        )
        width = 760
        height = 300
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title-sr desc-sr">',
            '  <title id="title-sr">Family D Deployment Shift Error and Audit-Delayed Recovery</title>',
            '  <desc id="desc-sr">Bar comparison of pre-shift MAE, peak post-shift MAE during audit telemetry delay, and steady-state post-shift MAE across the four baselines.</desc>',
            '  <rect width="100%" height="100%" fill="#0f172a" rx="8"/>',
            '  <text x="24" y="34" fill="#f8fafc" font-family="system-ui, sans-serif" font-size="15" font-weight="bold">Family D (DeploymentShift): Pre-Shift vs. Peak Transient vs. Post-Recovery MAE</text>',
            '  <text x="24" y="54" fill="#94a3b8" font-family="monospace" font-size="11">Unseen reserve_ratio (0.10 -> 0.35) &amp; expedite_fee ($2 -> $14) shift at step 8 with 2-step audit telemetry lag</text>',
        ]

        base_y = 235
        scale = 3.2
        for idx, b in enumerate(fam_d.baselines):
            sp = b.shift_protocol
            if sp is None:
                continue
            gx = 55 + idx * 170
            pre_h = min(140.0, sp.pre_shift_mae * scale)
            peak_h = min(140.0, sp.peak_post_shift_mae * scale)
            steady_h = min(140.0, sp.steady_state_post_shift_mae * scale)

            svg_lines.append(
                f'  <rect x="{gx}" y="{base_y - pre_h:.1f}" width="32" height="{max(2.0, pre_h):.1f}" fill="#38bdf8" rx="2"/>'
            )
            svg_lines.append(
                f'  <rect x="{gx + 38}" y="{base_y - peak_h:.1f}" width="32" height="{max(2.0, peak_h):.1f}" fill="#f59e0b" rx="2"/>'
            )
            svg_lines.append(
                f'  <rect x="{gx + 76}" y="{base_y - steady_h:.1f}" width="32" height="{max(2.0, steady_h):.1f}" fill="#ef4444" rx="2"/>'
            )
            svg_lines.append(
                f'  <text x="{gx + 54}" y="{base_y + 20}" fill="#e2e8f0" font-family="system-ui, sans-serif" font-size="11" font-weight="bold" text-anchor="middle">{b.paradigm.value[:18]}</text>'
            )
            rec_txt = (
                f"Rec: {sp.steps_until_recovery} steps"
                if sp.steps_until_recovery is not None
                else "Unrecovered"
            )
            svg_lines.append(
                f'  <text x="{gx + 54}" y="{base_y + 36}" fill="#94a3b8" font-family="monospace" font-size="10" text-anchor="middle">{rec_txt}</text>'
            )

        svg_lines.append(
            '  <text x="24" y="286" fill="#cbd5e1" font-family="system-ui, sans-serif" font-size="11">Legend: [Blue] Pre-Shift MAE | [Amber] Peak Post-Shift MAE (during 2-step audit lag) | [Red] Steady-State Post-Shift MAE</text>'
        )
        svg_lines.append("</svg>")
        return "\n".join(svg_lines)


def run_enterprise_benchmark_suite(
    seed: int = 42,
    fast: bool = False,
    families: Sequence[str] | None = None,
    timestamp: str | None = None,
) -> EnterpriseBenchmarkSuiteReport:
    """Execute the 12-family Enterprise World-Model Benchmark Suite (A through L)."""
    all_benchmarks = build_all_enterprise_benchmarks(master_seed=seed, fast=fast)
    if families:
        wanted = {f.strip().upper() for f in families}
        selected = [
            b
            for b in all_benchmarks
            if b.family_code.upper() in wanted
            or b.family.value.upper() in wanted
            or any(w in b.family.value.upper() for w in wanted)
        ]
    else:
        selected = all_benchmarks

    results: list[EnterpriseBenchmarkFamilyResult] = []
    h = hashlib.sha256()
    h.update(f"{BENCHMARK_SPEC_VERSION}:{METRIC_IMPLEMENTATION_VERSION}:{seed}".encode())

    fixed_ts = timestamp or datetime.now(UTC).isoformat()
    for bench in selected:
        res = bench.run_enterprise()
        if timestamp is not None:
            res = res.model_copy(
                update={
                    "reproducibility": res.reproducibility.model_copy(
                        update={"timestamp": fixed_ts}
                    )
                }
            )
        h.update(res.reproducibility.dataset_scenario_fingerprint.encode("utf-8"))
        h.update(res.reproducibility.output_fingerprint.encode("utf-8"))
        results.append(res)

    return EnterpriseBenchmarkSuiteReport(
        master_seed=seed,
        timestamp=fixed_ts,
        suite_fingerprint=h.hexdigest(),
        families=tuple(results),
    )


def render_enterprise_benchmark_markdown_doc(report: EnterpriseBenchmarkSuiteReport) -> str:
    """Generate the complete `docs/research/enterprise-benchmark-suite.md` document from `report`."""
    primary_table = report.generate_markdown_primary_table()
    dims_table = report.generate_markdown_dimensions_table()
    shift_table = report.generate_markdown_shift_table()
    failure_table = report.generate_markdown_failure_modes_table()
    repro_table = report.generate_markdown_reproducibility_table()

    return rf"""# Enterprise World-Model Evaluation & Failure-Mode Suite (A-L)

!!! info "Scientifically Defensible Failure-Mode Instrument — ADR-054"
    This benchmark suite answers: ***"When is an enterprise world model actually useful, correct, calibrated, robust, and decision-relevant?"***
    Every table and figure on this page is **automatically generated** from the committed machine-readable artifact [`docs/results/artifacts/enterprise_benchmark_suite.json`](../results/artifacts/enterprise_benchmark_suite.json) (`suite_fingerprint = {report.suite_fingerprint}`). Numbers are never copied by hand.

---

## 1. Design Philosophy & Research Lineage

Standard machine-learning evaluation reduces world modeling to a single observational one-step accuracy score. Recent research across enterprise workflow simulation, causal representation learning, and physical world models—including *World of Workflows*, *CascadeBench* (*Do Enterprise Systems Need Learned World Models?*), *Vending-Bench 2*, *Vending-Bench Arena*, Vafa et al. ([arXiv:2406.03689](https://arxiv.org/abs/2406.03689), [arXiv:2507.06952](https://arxiv.org/abs/2507.06952)), *Physics-IQ* ([arXiv:2501.09038](https://arxiv.org/abs/2501.09038)), and *RWM-U* ([arXiv:2504.16680](https://arxiv.org/abs/2504.16680))—shows that observational MSE conceals catastrophic operational failures.

**Anti-Optimization Mandate**: This benchmark is **not** tuned to make EWM Engine look artificially perfect. It explicitly exposes where each paradigm fails—including where `hybrid + runtime context discovery` incurs transient errors during audit-telemetry delays (Family D), bounded filtering lag under partial observability (Family E), raw state-prediction breakdown on unmodeled OOD structural physics requiring abstention (Family K), and higher compute latency/memory overhead.

```mermaid
flowchart TD
    accTitle: Enterprise World-Model Evaluation Matrix (Families A-L and Baselines)
    accDescr: Diagram showing how the 12 independent benchmark families evaluate the 4 baseline paradigms across 11 task types and 13 orthogonal metric dimensions without collapsing into a single score.
    Gen["Isolated Partition Generators (Train Seed != Eval Seeds)"] --> Guard["DataLeakageCertificate (Temporal, State-Hash & Config Gates)"]
    Guard --> Fam["12 Independent Benchmark Families (A through L)"]
    Fam --> B1["1. rules-only (Symbolic K_0)"]
    Fam --> B2["2. learned-only (Empirical Regressor)"]
    Fam --> B3["3. static hybrid (Rules + Projected Residual, Fixed K_0)"]
    Fam --> B4["4. hybrid + runtime context discovery (Dynamic K_t + Belief + Abstention)"]
    B1 & B2 & B3 & B4 --> Dim["13 Orthogonal Metric Dimensions (Never Collapsed)"]
```

---

## 2. Benchmark Card & Dataset/Scenario Cards (Families A-L)

### 2.1 Suite Benchmark Card

- **Suite Identifier**: `{report.suite_name}` (Spec `v{report.benchmark_spec_version}`, Metrics `v{report.metric_implementation_version}`)
- **Governing ADR**: [`ADR-054: Enterprise World-Model Evaluation & Failure-Mode Benchmark Suite`](../adr/ADR-054-enterprise-world-model-evaluation-benchmark-suite.md)
- **Master Seed**: `{report.master_seed}` (cryptographically partitioned via `np.random.SeedSequence` with domain tags `0xE03_1001` for training and `0xE03_2002` for evaluation)
- **Eleven Task Types Evaluated**: `next_state_prediction`, `action_effect_prediction`, `constraint_prediction`, `workflow_side_effect_prediction`, `audit_trace_reconstruction`, `configuration_context_inference`, `counterfactual_query`, `policy_ranking`, `robust_action_selection`, `model_abstention`, `confidence_calibration`
- **Intended Use**: Comparative diagnostic auditing of enterprise world-model architectures across transition fidelity, constraint preservation, causal selectivity, calibration, shift recovery, and decision regret.
- **Out-of-Scope Use**: Claiming real-world deployment readiness or empirical domain superiority from synthetic benchmark scores without domain-specific backtesting and causal identification.

### 2.2 Independent Family Scenario Cards (A through L)

| Family | Scenario Card & Data-Generating Mechanism | Task Types | Primary Dimensions |
| :---: | :--- | :--- | :--- |
| **A** | **`A_DirectDynamics`**: Deterministic inventory transfer with quadratic valve-friction loss above flow threshold $\tau = 10$. Exact analytical oracle ($\le 10^{{-12}}$). | `next_state_prediction`, `action_effect_prediction` | `TRANSITION_FIDELITY`, `CONSTRAINT_PRESERVATION`, `TRACE_PROVENANCE_COMPLETENESS` |
| **B** | **`B_HiddenWorkflowCascades`**: *World of Workflows* probe. Expedited orders ($q > 25$) trigger compliance hold backlogs, throttling dispatch and incurring SLA cash penalties never excited in routine training logs ($q \le 22$). | `workflow_side_effect_prediction`, `action_effect_prediction`, `audit_trace_reconstruction` | `TRANSITION_FIDELITY`, `CAUSAL_CORRECTNESS`, `TRACE_PROVENANCE_COMPLETENESS` |
| **C** | **`C_RuntimeConfiguration`**: Cross-tenant deployment (`tenant_us_standard` vs. `tenant_eu_strict`) sharing nominal schemas but differing in `ContextSnapshot` $K_t$ (22% VAT withholding + $15 dual-control fee). | `configuration_context_inference`, `next_state_prediction`, `workflow_side_effect_prediction` | `SHIFT_ROBUSTNESS`, `TRANSITION_FIDELITY`, `TRACE_PROVENANCE_COMPLETENESS` |
| **D** | **`D_DeploymentShift`**: Mid-rollout shift at step 8 (`reserve_ratio` $0.10 \to 0.35$, `expedite_fee` $\$2 \to \$14$) with a **2-step runtime audit telemetry delay**. | `configuration_context_inference`, `next_state_prediction`, `audit_trace_reconstruction` | `SHIFT_ROBUSTNESS`, `TIME_TO_RECOVERY_AFTER_SHIFT`, `TRANSITION_FIDELITY` |
| **E** | **`E_PartialObservability`**: `ObservationFilter` withholds latent `supplier_congestion` and secret rules while exposing noisy `audit_fill_shortfall` telemetry. | `next_state_prediction`, `audit_trace_reconstruction`, `confidence_calibration` | `TRANSITION_FIDELITY`, `CALIBRATION`, `TRACE_PROVENANCE_COMPLETENESS` |
| **F** | **`F_LongHorizonBusinessOperation`**: *Vending-Bench 2* spirit. Multi-step inventory, cash, demand, 3% per-step spoilage, and mid-horizon supplier wholesale price shift. | `next_state_prediction`, `workflow_side_effect_prediction`, `constraint_prediction` | `LONG_HORIZON_ERROR_GROWTH`, `TRANSITION_FIDELITY`, `CONSTRAINT_PRESERVATION` |
| **G** | **`G_MultiAgentEconomy`**: *Vending-Bench Arena* spirit. Buyer, Primary Supplier, and Arbitrage Desk clear scarce goods under 3-way conservation ($\sum \Delta R = 0$) and scarcity diversion. | `action_effect_prediction`, `workflow_side_effect_prediction`, `policy_ranking` | `CONSTRAINT_PRESERVATION`, `TRANSITION_FIDELITY`, `DECISION_REGRET` |
| **H** | **`H_CausalCounterfactualCorrectness`**: Synthetic SCM testing abducted-noise twin counterfactuals, non-descendant invariance (`macro_index`), and fail-closed refusal under latent confounding $U_{{\text{{macro}}}}$. | `counterfactual_query`, `action_effect_prediction`, `model_abstention` | `CAUSAL_CORRECTNESS`, `ABSTENTION_QUALITY`, `TRANSITION_FIDELITY` |
| **I** | **`I_ConservationConstraintStress`**: Adversarial overdraft transfers attempting negative stock and phantom mass fabrication across closed vaults ($V_A + V_B = 500$). Exact oracle ($\le 10^{{-12}}$). | `constraint_prediction`, `next_state_prediction` | `CONSTRAINT_PRESERVATION`, `TRANSITION_FIDELITY`, `TRACE_PROVENANCE_COMPLETENESS` |
| **J** | **`J_UncertaintyCalibration`**: Heteroscedastic load- and context-scaled Gaussian simulator testing 90% interval coverage, coverage error, and proper scoring (CRPS). | `confidence_calibration`, `next_state_prediction` | `CALIBRATION`, `TRANSITION_FIDELITY`, `SHIFT_ROBUSTNESS` |
| **K** | **`K_OODRegimeShift`**: Severe out-of-support load ($[180, 300]$ vs. $[10, 50]$) with an **unmodeled quadratic grid collapse** absent from `ContextSnapshot`, testing epistemic abstention. | `model_abstention`, `next_state_prediction`, `confidence_calibration` | `ABSTENTION_QUALITY`, `TRANSITION_FIDELITY`, `DECISION_REGRET` |
| **L** | **`L_DecisionQuality`**: Decouples prediction error from policy regret $U^*(\pi^*) - U^*(\hat{{\pi}})$ under confounded observational history vs. additive fixed-overhead offset. | `policy_ranking`, `robust_action_selection`, `counterfactual_query` | `DECISION_REGRET`, `TRANSITION_FIDELITY`, `CAUSAL_CORRECTNESS` |

---

## 3. Baseline Paradigms Evaluated

1. **`rules-only` (`RulesOnlyStructuralModel`)**: First-principles symbolic rules configured with compile-time context $K_0$, hard constraint enforcement, zero learned residual, and deterministic point predictions.
2. **`learned-only` (`LearnedPolynomialRidgeModel`)**: Empirical polynomial ridge regressor fitted strictly on training partition transitions $(S_t, A_t) \mapsto \Delta S_t$ without structural conservation projections, causal graph constraints, or runtime context $K_t$.
3. **`static hybrid` (`StaticHybridProjectedModel`)**: Symbolic rules + feasible-set projected learned residual (`InfeasibleResidualPolicy.PROJECT`), locked to compile-time context $K_0$ without runtime audit discovery or belief updating.
4. **`hybrid + runtime context discovery` (`ContextualHybridDiscoveryModel`)**: Symbolic rules + feasible-set projected learned residual + `AuditMetadataContextDiscovery` evolving $K_t$ from runtime audit telemetry + `ExponentialSmoothingBeliefUpdater` for partial observability + `CompetenceEstimator` for OOD abstention.

---

## 4. Comparative Results Across Families A-L (Auto-Generated)

![Enterprise Benchmark Suite Primary Metric Heatmap](../results/artifacts/enterprise_benchmark_heatmap.svg)

{primary_table}

> **Statistical Protocol Note**: Entries marked `(det)` denote deterministic exact-oracle evaluations where statistical significance is not applicable and exact/tolerance correctness is reported. Entries with `±` report the sample mean and 95% non-parametric bootstrap confidence half-width across independent seeds spawned via `np.random.SeedSequence`, with Holm-Bonferroni step-down adjusted $p$-values and paired Cohen's $d_z$ recorded in the JSON artifact.

---

## 5. Thirteen Orthogonal Metric Dimensions Profile (Auto-Generated)

The thirteen evaluation dimensions are formally defined in `benchmarks/protocol.py` (`METRIC_DEFINITIONS`) and are **never collapsed** into a single aggregate score:

{dims_table}

---

## 6. Deployment & Configuration Shift Protocol Analysis (Auto-Generated)

![Family D Deployment Shift Error and Recovery](../results/artifacts/enterprise_shift_recovery.svg)

{shift_table}

---

## 7. Surfaced Failure Modes Across All Baselines (Auto-Generated)

{failure_table}

---

## 8. Data Non-Leakage & Reproducibility Certificates (Auto-Generated)

{repro_table}

---

## 9. Limitations & Threats to Validity

1. **Internal Validity**:
   - **Threat**: Procedural generator seed coupling or accidental overlap between training and evaluation states could inflate learned baseline scores.
   - **Mitigation**: `derive_partition_seeds` uses distinct cryptographic domain tags (`0xE03_1001` vs. `0xE03_2002`), and `verify_no_data_leakage` enforces zero ontic state-fingerprint overlap, strict temporal precedence ($\max t_{{\text{{train}}}} < \min t_{{\text{{eval}}}}$), and held-out context version isolation across all 12 families.
2. **Construct Validity**:
   - **Threat**: Collapsing accuracy, safety, calibration, and latency into a single scalar score masks critical operational trade-offs (e.g., `learned-only` is fast and fits smooth curves, but violates conservation in Family G/I and incurs high decision regret in Family L).
   - **Mitigation**: All 13 `MetricDimension`s are reported separately, and Family L explicitly separates predictive MAE from decision regret $U^*(\pi^*) - U^*(\hat{{\pi}})$.
3. **External Validity**:
   - **Threat**: Synthetic enterprise scenarios are stylized abstractions of production ERP/SCM workflows; when an unseen physical/structural change falls outside the schema of `ContextSnapshot` (as demonstrated in **Family K**), runtime context discovery alone cannot predict the new trajectory.
   - **Mitigation**: Family K explicitly tests and documents this boundary—showing that `hybrid + runtime context discovery` suffers high raw prediction error on unmodeled OOD physics and must rely on `CompetenceEstimator` abstention rather than claiming omniscient extrapolation.
4. **Statistical Conclusion Validity**:
   - **Threat**: Manufacturing $p$-values on deterministic simulators or ignoring multiple comparisons across baselines.
   - **Mitigation**: Deterministic oracle families (A, I) set `statistical_significance_applicable = False` and verify numerical tolerances ($\le 10^{{-12}}$), while stochastic families separate Monte Carlo standard error ($\text{{SE}}_{{\text{{MC}}}}$) from internal model uncertainty spread and apply Holm-Bonferroni adjustments across baseline comparisons.

---

## 10. Reproduction Instructions

```bash
# Run the full 12-family Enterprise Benchmark Suite and regenerate JSON + SVG + Markdown artifacts
python -m benchmarks.enterprise_suite --seed 42 --update-artifacts

# Run fast smoke configuration (< 0.5s) for CI verification
python -m benchmarks.enterprise_suite --seed 42 --fast

# Run specific families (e.g., Hidden Workflow Cascades, Deployment Shift, Decision Quality)
python -m benchmarks.enterprise_suite --family B D L

# Execute the self-validation test suite verifying oracles, metrics, repeatability, and non-leakage
pytest tests/benchmark/test_enterprise_benchmark_suite.py -q
```
"""


def generate_enterprise_benchmark_artifacts(
    report: EnterpriseBenchmarkSuiteReport,
    repo_root: Path | None = None,
) -> dict[str, Path]:
    """Write machine-readable JSON, vector SVGs, and Markdown documentation from `report`."""
    root = repo_root or REPO_ROOT
    artifacts_dir = root / "docs" / "results" / "artifacts"
    research_dir = root / "docs" / "research"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    research_dir.mkdir(parents=True, exist_ok=True)

    json_path = artifacts_dir / "enterprise_benchmark_suite.json"
    heatmap_path = artifacts_dir / "enterprise_benchmark_heatmap.svg"
    shift_svg_path = artifacts_dir / "enterprise_shift_recovery.svg"
    doc_path = research_dir / "enterprise-benchmark-suite.md"

    report.save_json(json_path)
    heatmap_path.write_text(report.generate_svg_heatmap() + "\n", encoding="utf-8")
    shift_svg_path.write_text(report.generate_svg_shift_recovery() + "\n", encoding="utf-8")
    doc_path.write_text(render_enterprise_benchmark_markdown_doc(report), encoding="utf-8")

    return {
        "json": json_path,
        "heatmap_svg": heatmap_path,
        "shift_svg": shift_svg_path,
        "markdown_doc": doc_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EWM Engine 12-Family Enterprise World-Model Evaluation Suite (ADR-054)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Master seed (default: 42)")
    parser.add_argument(
        "--fast", action="store_true", help="Run fast smoke-test horizon/seed configuration"
    )
    parser.add_argument(
        "--family",
        nargs="*",
        default=None,
        help="Optional filter of family codes (A..L) or names",
    )
    parser.add_argument("--json", type=str, default=None, help="Optional path to save JSON report")
    parser.add_argument(
        "--update-artifacts",
        action="store_true",
        help="Regenerate committed JSON, SVG figures, and Markdown docs in docs/",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default="2026-10-21T00:00:00+00:00",
        help="Deterministic ISO-8601 timestamp for reproducible artifact generation",
    )

    args = parser.parse_args()
    report = run_enterprise_benchmark_suite(
        seed=args.seed,
        fast=args.fast,
        families=args.family,
        timestamp=args.timestamp,
    )
    print(report.to_text_table())

    if args.json:
        report.save_json(args.json)
        print(f"\nSaved JSON report to: {args.json}")

    if args.update_artifacts:
        paths = generate_enterprise_benchmark_artifacts(report)
        print("\nUpdated Enterprise Benchmark Suite artifacts:")
        for k, p in paths.items():
            print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()
