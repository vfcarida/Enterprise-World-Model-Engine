#!/usr/bin/env python3
"""Run Airspeed Velocity (ASV) performance history suite for EWM Engine.

Executes ASV benchmarks across commits, generates static HTML performance dashboard,
and produces human-readable markdown summaries.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_cmd(cmd: list[str]) -> int:
    """Execute command and stream output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ASV benchmark suite.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run each benchmark once for quick sanity checking.",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        help="Filter benchmarks by regex pattern.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish static HTML dashboard after running.",
    )
    parser.add_argument(
        "--environment",
        "-E",
        type=str,
        default="existing",
        help="Environment specification for ASV (default: existing).",
    )
    args = parser.parse_args()

    # 1. Initialize machine config non-interactively if needed
    run_cmd([sys.executable, "-m", "asv", "machine", "--yes"])

    # 2. Check benchmarks
    check_code = run_cmd([sys.executable, "-m", "asv", "check", "-E", args.environment])
    if check_code != 0:
        print("ASV benchmark check failed!", file=sys.stderr)
        return check_code

    # 3. Run benchmarks
    run_cmd_args = [
        sys.executable,
        "-m",
        "asv",
        "run",
        "-E",
        args.environment,
        "--show-stderr",
    ]
    if args.quick:
        run_cmd_args.append("--quick")
    if args.bench:
        run_cmd_args.extend(["--bench", args.bench])

    run_code = run_cmd(run_cmd_args)
    if run_code != 0:
        print("ASV run completed with non-zero exit code.", file=sys.stderr)

    # 4. Publish static HTML if requested
    if args.publish:
        pub_code = run_cmd([sys.executable, "-m", "asv", "publish"])
        if pub_code == 0:
            print("ASV static HTML dashboard published to .asv/html/")

    return run_code


if __name__ == "__main__":
    sys.exit(main())
