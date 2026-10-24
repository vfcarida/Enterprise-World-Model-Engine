#!/usr/bin/env python3
"""Reproducible runner for Fenced Experimental Lines (Composition & SWE Domain)."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ewm_engine.experimental.domain_composition import run_domain_composition_experiment
from ewm_engine.experimental.swe_ecosystem import run_swe_ecosystem_experiment

ARTIFACT_PATH = ROOT / "docs" / "results" / "artifacts" / "experimental_composition_and_swe.json"


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--write-artifact", action="store_true", default=True)
    args = parser.parse_args()

    comp_report = run_domain_composition_experiment(seed=args.seed, n_seeds=args.n_seeds)
    swe_report = run_swe_ecosystem_experiment(seed=args.seed, n_seeds=args.n_seeds)

    payload = {
        "schema_version": "1.0.0",
        "experiment_suite_id": "EXP-COMPOSITION-SWE-v1",
        "namespace": "ewm_engine.experimental",
        "evidence_tier": "PROPERTY_VERIFIED",
        "runtime_environment": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.system(),
            "git_commit": _git_commit(),
        },
        "line1_domain_composition": comp_report.to_dict(),
        "line2_swe_ecosystem": swe_report.to_dict(),
    }

    if args.write_artifact:
        ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("=" * 104)
    print("FENCED EXPERIMENTAL LINES: DOMAIN COMPOSITION & SOFTWARE-ENGINEERING ECOSYSTEM")
    print(
        f"Env: Python {platform.python_version()} | NumPy {np.__version__} | "
        f"OS: {platform.system()} | Commit: {_git_commit()} | Seed: {args.seed} (M={args.n_seeds})"
    )
    print("=" * 104)
    print("\n[LINE 1] Domain-Model Composition (Fraud/Customer Submodel -> Call-Center Submodel)")
    print(f"  Epistemic Status           : {comp_report.epistemic_status}")
    print(
        f"  Spec Components            : Monolithic={comp_report.monolithic_spec_components} vs "
        f"Composed={comp_report.composed_spec_components} "
        f"(+{comp_report.specification_overhead_pct:.1f}% overhead)"
    )
    print(
        f"  Error Propagation          : +{comp_report.error_propagation_audit['upstream_call_rate_relative_bias'] * 100:.1f}% "
        f"upstream call-rate bias -> +{comp_report.error_propagation_audit['downstream_wait_time_relative_error'] * 100:.1f}% "
        f"downstream wait error ({comp_report.error_propagation_audit['nonlinear_boundary_amplification_ratio']:.2f}x amplification)"
    )
    print(
        f"  Assumption Gate            : Correlational do() blocked={comp_report.assumption_incompatibility_audit['correlational_upstream_blocked_for_intervention']} | "
        f"Unit mismatch blocked={comp_report.assumption_incompatibility_audit['cohort_unit_mismatch_blocked']} "
        f"(Ungated silent corruption=${comp_report.assumption_incompatibility_audit['ungated_cohort_mismatch_silent_corruption_cost_usd']:,.2f})"
    )
    print(
        f"  Temporal Desync (Cadence=3): Utility Loss=${comp_report.temporal_desync_audit['desynchronized_utility_loss_usd']:,.2f} | "
        f"Extra SLA Breach Steps={comp_report.temporal_desync_audit['desynchronized_extra_sla_breach_steps']}"
    )
    print(
        f"  Uncertainty Accumulation   : Coupled 90% width={comp_report.uncertainty_accumulation_audit['coupled_90pct_wait_interval_width_sec']:.2f}s vs "
        f"Naive sum width={comp_report.uncertainty_accumulation_audit['naive_boundary_sum_interval_width_sec']:.2f}s "
        f"(+{comp_report.uncertainty_accumulation_audit['naive_interval_width_inflation_pct']:.1f}% inflation)"
    )
    for nf in comp_report.negative_findings:
        print(f"  Negative Finding [{nf['finding_id']}]    : {nf['title']}")

    print("\n[LINE 2] Software-Engineering Ecosystem (6-Component Service/DB/Cache/Lib Topology)")
    print(f"  Epistemic Status           : {swe_report.epistemic_status}")
    print(f"  Static Dependents(auth_lib): {swe_report.static_dependents_of_auth_lib}")
    print(f"  True Dynamic Impacted      : {swe_report.true_dynamic_impacted_under_burst_rollout}")
    print(
        f"  Bitemporal Guard (as_of=5) : Withheld {swe_report.bitemporal_withheld_records_at_t5} future records"
    )
    print("-" * 104)
    print(
        f"{'Rung':<42} | {'Impact F1':>9} | {'Lat MAE':>8} | {'Test Rec':>8} | {'RC Prec':>7} | {'Deploy Regret ($)':>17}"
    )
    print("-" * 104)
    for r in swe_report.ladder_results:
        print(
            f"{r.rung_id:<42} | {r.change_impact_f1:>9.4f} | {r.change_impact_latency_mae_ms:>8.2f} | "
            f"{r.test_selection_fault_recall:>8.2f} | {r.root_cause_precision_at_1:>7.2f} | "
            f"{r.deployment_decision_regret_usd:>17,.2f}"
        )
    print("-" * 104)
    for nf in swe_report.negative_findings_and_transfer_limits:
        print(f"  Negative Finding [{nf['finding_id']}]  : {nf['title']}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
