"""Coverage verification script enforcing spec quality gates.

Enforces:
- Overall test coverage >= 85.0%
- Core areas test coverage >= 90.0% for:
  - ewm_engine.core
  - ewm_engine.simulation
  - ewm_engine.constraints
  - ewm_engine.provenance
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OVERALL_THRESHOLD = 85.0
CORE_AREA_THRESHOLDS = {
    "core": 90.0,
    "simulation": 90.0,
    "constraints": 90.0,
    "provenance": 90.0,
}


def ensure_coverage_data(force: bool = False) -> Path:
    """Ensure .coverage data file exists or run pytest to generate it."""
    cov_file = REPO_ROOT / ".coverage"
    if force or "--fresh" in sys.argv or "--force" in sys.argv or not cov_file.exists():
        if cov_file.exists():
            try:
                cov_file.unlink()
            except OSError:
                pass
        print("Generating fresh coverage data via pytest --cov=ewm_engine...")
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "--cov=ewm_engine",
            "-q",
        ]
        res = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if res.returncode != 0:
            print("ERROR: Test run failed during coverage generation.", file=sys.stderr)
            sys.exit(res.returncode)
    return cov_file


def generate_coverage_json() -> dict:
    """Generate and parse JSON report from coverage data."""
    json_path = REPO_ROOT / "coverage.json"
    cmd = [
        sys.executable,
        "-m",
        "coverage",
        "json",
        "-o",
        str(json_path),
    ]
    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if res.returncode != 0:
        print(f"ERROR: Failed to generate coverage JSON:\n{res.stderr}", file=sys.stderr)
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Clean up temporary json report
    try:
        json_path.unlink()
    except OSError:
        pass

    return data


def main() -> int:
    ensure_coverage_data()
    data = generate_coverage_json()

    totals = data.get("totals", {})
    overall_percent = float(totals.get("percent_covered", 0.0))

    files = data.get("files", {})

    area_stats: dict[str, dict[str, float | int]] = {}
    for area in CORE_AREA_THRESHOLDS:
        stmts = 0
        covered = 0
        for filepath, file_data in files.items():
            norm_path = filepath.replace("\\", "/")
            if f"/ewm_engine/{area}/" in norm_path:
                summary = file_data.get("summary", {})
                num_stmts = summary.get("num_statements", 0)
                cov_lines = summary.get("covered_lines", 0)
                num_branches = summary.get("num_branches", 0)
                cov_branches = summary.get("covered_branches", 0)
                missing_branches = summary.get("missing_branches", 0)

                total_elements = num_stmts + cov_branches + missing_branches
                total_covered = cov_lines + cov_branches

                stmts += total_elements
                covered += total_covered

        percent = (covered / stmts * 100.0) if stmts > 0 else 0.0
        area_stats[area] = {
            "percent": percent,
            "covered": covered,
            "total": stmts,
        }

    print("\n" + "=" * 70)
    print("EWM ENGINE TEST COVERAGE QUALITY GATE CHECK")
    print("=" * 70)
    print(f"{'Target Area':<20} | {'Required':<10} | {'Actual':<10} | {'Status'}")
    print("-" * 70)

    failed = False

    # Check overall
    overall_status = "PASS" if overall_percent >= OVERALL_THRESHOLD else "FAIL"
    if overall_status == "FAIL":
        failed = True
    print(
        f"{'TOTAL (Overall)':<20} | {OVERALL_THRESHOLD:>8.1f}% | {overall_percent:>8.2f}% | {overall_status}"
    )

    # Check each core area
    for area, threshold in CORE_AREA_THRESHOLDS.items():
        stats = area_stats[area]
        act_pct = float(stats["percent"])
        status = "PASS" if act_pct >= threshold else "FAIL"
        if status == "FAIL":
            failed = True
        area_name = f"core/{area}"
        print(f"{area_name:<20} | {threshold:>8.1f}% | {act_pct:>8.2f}% | {status}")

    print("=" * 70)

    if failed:
        print("\nERROR: Quality gate check failed. One or more coverage thresholds were breached.\n", file=sys.stderr)
        return 1

    print("\nSUCCESS: All overall and core-area coverage quality gates passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
