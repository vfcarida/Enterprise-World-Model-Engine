#!/usr/bin/env python3
"""Reproducible runner and artifact generator for the Flagship Fraud -> Call-Center Cascade.

Usage:
    python scripts/run_fraud_cascade_flagship.py           # Full deterministic run + write artifact
    python scripts/run_fraud_cascade_flagship.py --smoke   # Fast seeded CI smoke verification
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ewm_engine.core.bitemporal import BitemporalLeakageError
from ewm_engine.flagship.fraud_cascade import (
    FLAGSHIP_EPISTEMIC_DISCLAIMER,
    FLAGSHIP_EPISTEMIC_LABEL,
    FLAGSHIP_SCENARIO_ID,
    FraudCascadeEnvironment,
    FraudCascadeParameters,
    run_four_capability_queries,
    verify_known_mechanisms_recovery,
)

DEFAULT_ARTIFACT_PATH = REPO_ROOT / "docs" / "results" / "artifacts" / "fraud_cascade_flagship.json"


def _git_head_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def _lockfile_sha256() -> str:
    lock_path = REPO_ROOT / "uv.lock"
    if not lock_path.exists():
        return "missing"
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def build_flagship_artifact(
    *,
    master_seed: int = 42,
    train_seed: int = 101,
    eval_seed: int = 202,
    horizon: int = 20,
) -> dict[str, Any]:
    """Execute the flagship fraud-cascade scenario and assemble the canonical JSON artifact."""
    params = FraudCascadeParameters()
    env = FraudCascadeEnvironment(params)

    # 1. Verify ground-truth vs. recovered mechanism checks
    mechanism_checks = verify_known_mechanisms_recovery(params, seed=master_seed, horizon=40)

    # 2. Execute the 4 vision queries + DataLeakageCertificate
    four_queries = run_four_capability_queries(
        params,
        train_seed=train_seed,
        eval_seed=eval_seed,
        horizon=horizon,
    )

    # 3. Verify bitemporal leakage guard explicitly on a rollout ledger
    _records, ledger, _ontic, _obs = env.rollout(horizon=horizon, seed=master_seed)
    pit_t5 = ledger.as_of(5.0)
    leakage_guard_triggered = False
    blocked_error_message = ""
    try:
        # Attempt to read step-5 confirmed chargeback settlement (knowledge_time = 9.0) at decision_time = 5.0
        ledger.read_record_at("fraud_final_step_5", decision_time=5.0)
    except BitemporalLeakageError as exc:
        leakage_guard_triggered = True
        blocked_error_message = str(exc)

    # Compute deterministic output fingerprint over core scientific results
    core_payload = {
        "scenario_id": FLAGSHIP_SCENARIO_ID,
        "parameters_fingerprint": params.fingerprint,
        "master_seed": master_seed,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "horizon": horizon,
        "mechanism_checks": mechanism_checks,
        "four_queries": four_queries,
    }
    output_fp = hashlib.sha256(
        json.dumps(core_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return {
        "scenario_id": FLAGSHIP_SCENARIO_ID,
        "epistemic_label": FLAGSHIP_EPISTEMIC_LABEL,
        "disclaimer": FLAGSHIP_EPISTEMIC_DISCLAIMER,
        "environment_manifest": {
            "parameters": params.model_dump(mode="json"),
            "parameters_fingerprint": params.fingerprint,
            "initial_context_fingerprint": env.initial_context.fingerprint,
            "observable_resources": sorted(env.observation_spec.observable_resource_ids or ()),
            "hidden_ontic_resources": [
                "latent_churn_propensity",
                "instantaneous_fp_declines",
                "instantaneous_missed_fraud",
            ],
            "cascade_layers": [
                {
                    "layer": 1,
                    "name": "Direct Authorization Transition",
                    "evidence_nature": "business_rule_mechanism",
                    "description": "Context K_t auth_threshold tau governs p_catch(tau) and p_fp(tau).",
                },
                {
                    "layer": 2,
                    "name": "Triggered Fraud Workflow",
                    "evidence_nature": "business_rule_mechanism",
                    "description": "Step-up SMS rescue (65%) and automated account locks (45% of net FP declines).",
                },
                {
                    "layer": 3,
                    "name": "Secondary Customer Reactions & Call Queue",
                    "evidence_nature": "experimentally_supported",
                    "description": "Retries (35%), channel switch (20%), contact-center call surge, and queue congestion.",
                },
                {
                    "layer": 4,
                    "name": "Downstream Constraints, SLA Penalties & Conservation",
                    "evidence_nature": "known_functional",
                    "description": "Customer pool conservation (N_0=10,000), queue flow conservation, SLA breach penalty, and cash ledger.",
                },
            ],
        },
        "reproducibility": {
            "git_commit": _git_head_commit(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "uv_lock_sha256": _lockfile_sha256(),
            "master_seed": master_seed,
            "train_seed": train_seed,
            "eval_seed": eval_seed,
            "horizon": horizon,
            "scenario_fingerprint": params.fingerprint,
            "output_fingerprint": output_fp,
        },
        "bitemporal_audit": {
            "decision_time_inspected": 5.0,
            "visible_records_at_t5": len(pit_t5.visible_records),
            "withheld_future_knowledge_at_t5": pit_t5.withheld_future_knowledge_count,
            "withheld_future_events_at_t5": pit_t5.withheld_future_event_count,
            "leakage_guard_triggered_on_future_read": leakage_guard_triggered,
            "blocked_leakage_attempts": ledger.blocked_leakage_attempts,
            "sample_blocked_error": blocked_error_message,
        },
        "mechanism_recovery_checks": mechanism_checks,
        "four_queries": four_queries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Flagship Fraud -> Declines -> Call-Center Cascade scenario."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast seeded smoke check and verify artifact determinism without overwriting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Output path for the canonical JSON artifact.",
    )
    args = parser.parse_args()

    horizon = 20
    artifact = build_flagship_artifact(horizon=horizon)

    if not artifact["mechanism_recovery_checks"]["all_passed"]:
        print("[FAIL] Mechanism recovery checks failed!", file=sys.stderr)
        return 1
    if not artifact["four_queries"]["leakage_certificate"]["passed"]:
        print("[FAIL] DataLeakageCertificate check failed!", file=sys.stderr)
        return 1
    if not artifact["bitemporal_audit"]["leakage_guard_triggered_on_future_read"]:
        print("[FAIL] Bitemporal leakage guard did not trigger!", file=sys.stderr)
        return 1

    if args.smoke:
        if args.output.exists():
            committed = json.loads(args.output.read_text(encoding="utf-8"))
            committed_fp = committed.get("reproducibility", {}).get("output_fingerprint")
            actual_fp = artifact["reproducibility"]["output_fingerprint"]
            if committed_fp != actual_fp:
                print(
                    f"[FAIL] Flagship artifact fingerprint mismatch: committed={committed_fp} vs actual={actual_fp}",
                    file=sys.stderr,
                )
                return 1
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    plan = artifact["four_queries"]["planning"]
    print(f"[OK] Flagship scenario '{FLAGSHIP_SCENARIO_ID}' verified.")
    print(f"     Epistemic Label      : {artifact['epistemic_label']}")
    print(
        f"     Runtime Env          : Python {artifact['reproducibility']['python_version']} | NumPy {artifact['reproducibility']['numpy_version']} | Commit {artifact['reproducibility']['git_commit'][:12]}"
    )
    print(f"     Scenario Fingerprint : {artifact['reproducibility']['scenario_fingerprint']}")
    print(f"     Output Fingerprint   : {artifact['reproducibility']['output_fingerprint']}")
    print(
        f"     Mechanism Recovery   : PASS ({len(artifact['mechanism_recovery_checks']['checks'])}/6 checks within tolerance)"
    )
    print(
        f"     Bitemporal Guard     : PASS (withheld={artifact['bitemporal_audit']['withheld_future_knowledge_at_t5']} future-knowledge records at t=5.0)"
    )
    print(
        f"     Leakage Certificate  : PASS (overlap={artifact['four_queries']['leakage_certificate']['state_fingerprint_overlap_count']})"
    )
    print(
        f"     Planning Regret      : Myopic={plan['myopic_decision_regret']} ({plan['myopic_selected_policy']}) vs. Cascade-EWM={plan['cascade_ewm_decision_regret']} ({plan['cascade_ewm_selected_policy']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
