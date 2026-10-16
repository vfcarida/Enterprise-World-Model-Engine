#!/usr/bin/env python3
"""Standalone Conformance Test Runner for Enterprise World Model Engine.

Evaluates simulation engines against the EWM Conformance Suite.
Adopts the JSON Schema Test Suite operational model:
- Conformance is defined as: exact bit-level reproduction of expected SHA-256 fingerprints.
- Executes mandatory 'latest/' cases and optional/package cases.
- Supports negative mutation testing to verify that divergent fingerprints fail loudly.

Usage:
    python conformance/runner.py
    python conformance/runner.py --category deterministic
    python conformance/runner.py --test-negative
    python conformance/runner.py --report-json conformance_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from conformance.reference_worlds import (  # noqa: E402
    create_closed_form_stochastic_world,
    create_constraint_stress_world,
    create_seed_trivial_world,
)
from ewm_engine.simulation.scenario import Scenario  # noqa: E402
from ewm_engine.simulation.trajectory import SimulationResult  # noqa: E402

CONFORMANCE_DIR = Path(__file__).resolve().parent
TESTS_DIR = CONFORMANCE_DIR / "tests"


WORLD_FACTORIES = {
    "seed_trivial": create_seed_trivial_world,
    "closed_form_stochastic": create_closed_form_stochastic_world,
    "constraint_stress": create_constraint_stress_world,
}


def load_test_cases(target_dir: Path, category_filter: str | None = None) -> list[dict[str, Any]]:
    """Load all JSON test fixtures from directory tree, optionally filtered by category."""
    cases: list[dict[str, Any]] = []
    for json_file in sorted(target_dir.rglob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
                data["_file_path"] = str(json_file.relative_to(CONFORMANCE_DIR))
                if category_filter and data.get("category") != category_filter:
                    continue
                cases.append(data)
        except Exception as exc:
            print(f"[ERROR] Failed to load fixture {json_file}: {exc}", file=sys.stderr)
    return cases


def run_conformance_case(case: dict[str, Any], *, mutate_expected: bool = False) -> dict[str, Any]:
    """Execute a single conformance case against the reference world model."""
    name = case.get("name", "Unnamed Case")
    category = case.get("category", "unknown")
    ref_world_key = case.get("reference_world")

    # If it's a proposal or non-executable fixture, skip execution
    if not ref_world_key or ref_world_key not in WORLD_FACTORIES:
        return {
            "name": name,
            "category": category,
            "status": "skipped",
            "reason": f"Non-executable or abstract proposal: {case.get('_file_path')}",
            "passed": True,
        }

    factory = WORLD_FACTORIES[ref_world_key]
    world, default_scen = factory()

    # Build scenario from case or use default
    scen_data = case.get("scenario")
    if scen_data:
        scenario = Scenario.model_validate(scen_data)
    else:
        scenario = default_scen

    start_t = time.perf_counter()
    result: SimulationResult = world.simulate(scenario=scenario)
    elapsed = time.perf_counter() - start_t

    actual_fp = result.fingerprint
    actual_branches = [t.final_state.fingerprint for t in result.trajectories]
    actual_status = result.trajectories[0].status.value if result.trajectories else "unknown"
    actual_violations = result.trajectories[0].total_violations if result.trajectories else 0

    expected_fp = case.get("expected_fingerprint", "")
    if mutate_expected:
        # Deliberately perturb expected fingerprint to test negative failure detection
        expected_fp = "0000000000000000000000000000000000000000000000000000000000000000"

    expected_branches = case.get("expected_branch_fingerprints")
    expected_status = case.get("expected_status")
    expected_violations = case.get("expected_total_violations")

    # Check assertions
    fp_match = actual_fp == expected_fp if expected_fp else True
    branches_match = actual_branches == expected_branches if expected_branches else True
    status_match = actual_status == expected_status if expected_status else True
    violations_match = (
        actual_violations == expected_violations if expected_violations is not None else True
    )

    passed = fp_match and branches_match and status_match and violations_match

    return {
        "name": name,
        "category": category,
        "file": case.get("_file_path"),
        "passed": passed,
        "status": "pass" if passed else "fail",
        "elapsed_seconds": elapsed,
        "actual_fingerprint": actual_fp,
        "expected_fingerprint": expected_fp,
        "fp_match": fp_match,
        "branches_match": branches_match,
        "status_match": status_match,
        "violations_match": violations_match,
    }


def run_suite(
    suite_dir: Path = TESTS_DIR / "latest",
    category_filter: str | None = None,
    test_negative: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """Run suite of conformance cases and return results and overall pass flag."""
    cases = load_test_cases(suite_dir, category_filter=category_filter)
    results: list[dict[str, Any]] = []
    all_passed = True

    for c in cases:
        res = run_conformance_case(c, mutate_expected=test_negative)
        results.append(res)
        if not res["passed"]:
            all_passed = False

    return results, all_passed


def main() -> int:
    parser = argparse.ArgumentParser(description="EWM Conformance Test Runner")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by category (deterministic, stochastic, constraints)",
    )
    parser.add_argument(
        "--suite-dir", type=str, default=None, help="Path to specific test suite directory"
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all suites (latest + optional + proposals)"
    )
    parser.add_argument(
        "--test-negative",
        action="store_true",
        help="Run negative mutation test (asserts failures are caught)",
    )
    parser.add_argument(
        "--report-json", type=str, default=None, help="Save JSON test report to file"
    )
    args = parser.parse_args()

    if args.suite_dir:
        suite_dir = Path(args.suite_dir)
    elif args.all:
        suite_dir = TESTS_DIR
    else:
        suite_dir = TESTS_DIR / "latest"
    print("=" * 70)
    print("Enterprise World Model Engine - Conformance Suite Runner")
    print(f"Target Directory: {suite_dir}")
    if args.category:
        print(f"Category Filter:  {args.category}")
    if args.test_negative:
        print("Mode:             NEGATIVE MUTATION TEST (asserting divergence is caught)")
    print("=" * 70)

    results, all_passed = run_suite(
        suite_dir=suite_dir,
        category_filter=args.category,
        test_negative=args.test_negative,
    )

    for r in results:
        badge = "[PASS]" if r["passed"] else "[FAIL]"
        if r.get("status") == "skipped":
            badge = "[SKIP]"
        print(f"{badge:6s} {r['name']:<35} ({r.get('category', 'unknown')})")
        if not r["passed"]:
            print(f"       Expected FP: {r.get('expected_fingerprint')}")
            print(f"       Actual FP:   {r.get('actual_fingerprint')}")

    print("=" * 70)
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"Summary: {passed_count}/{total} cases passed.")

    if args.report_json:
        report_path = Path(args.report_json)
        report_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": total,
            "passed": passed_count,
            "all_passed": all_passed,
            "results": results,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"Report written to: {report_path}")

    if args.test_negative:
        # In negative mutation test, we EXPECT failure when fingerprints diverge!
        if not all_passed:
            print(
                "[SUCCESS] Negative mutation test passed: divergent fingerprints were properly caught."
            )
            return 0
        else:
            print(
                "[FAILURE] Negative mutation test failed: divergent fingerprints went undetected!",
                file=sys.stderr,
            )
            return 1

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
