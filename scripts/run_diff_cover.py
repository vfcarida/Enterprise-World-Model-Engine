"""Diff-cover patch coverage gate enforcing >=90% branch/statement coverage on changed lines.

In accordance with R03 Task 8:
- Turn on branch coverage (branch = true in pyproject.toml).
- Enforces patch coverage >= 90% on changed lines via diff-cover.
- Keeps core/overall floors as a secondary check.
- Principles:
  "Test coverage is a useful tool for finding untested code; it is of little
  use as a guide to code quality." — Martin Fowler
  Coverage is an invariant lower bound (a floor), never the primary testing goal.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_coverage_xml() -> Path:
    """Ensure coverage.xml exists, generating from .coverage if needed."""
    xml_path = REPO_ROOT / "coverage.xml"
    cov_file = REPO_ROOT / ".coverage"

    if not cov_file.exists():
        print("[*] Generating fresh coverage data with branch coverage...")
        res = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=ewm_engine",
                "--cov-branch",
                "-q",
                "tests/unit",
            ],
            cwd=REPO_ROOT,
        )
        if res.returncode != 0:
            print("[!] Pytest execution failed during coverage collection.", file=sys.stderr)
            sys.exit(res.returncode)

    print("[*] Exporting coverage.xml...")
    res = subprocess.run(
        [sys.executable, "-m", "coverage", "xml", "-o", str(xml_path)], cwd=REPO_ROOT
    )
    if res.returncode != 0:
        print("[!] coverage xml generation failed.", file=sys.stderr)
        sys.exit(res.returncode)

    return xml_path


def run_diff_cover(compare_branch: str, fail_under: float) -> int:
    """Execute diff-cover on coverage.xml against the compare branch."""
    xml_path = ensure_coverage_xml()

    print(f"\n=== EWM Engine Patch Coverage Gate (diff-cover >= {fail_under}%) ===")
    print(f"Comparing against branch: {compare_branch}")
    print(
        "Principle: 'Test coverage is a useful tool for finding untested code, "
        "not a proxy for test suite quality.' — Fowler / Google Testing Blog\n"
    )

    cmd = [
        sys.executable,
        "-m",
        "diff_cover.diff_cover_tool",
        str(xml_path),
        f"--compare-branch={compare_branch}",
        f"--fail-under={fail_under}",
    ]

    try:
        res = subprocess.run(cmd, cwd=REPO_ROOT)
        return res.returncode
    except Exception as exc:
        print(f"[!] diff-cover execution error: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="EWM Engine Patch Coverage Gate (diff-cover)")
    parser.add_argument(
        "--compare-branch",
        default="HEAD~1",
        help="Git branch or reference to compare against (e.g. origin/main, HEAD~1)",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=90.0,
        help="Minimum patch coverage percentage required on changed lines",
    )
    args = parser.parse_args()

    return run_diff_cover(args.compare_branch, args.fail_under)


if __name__ == "__main__":
    sys.exit(main())
