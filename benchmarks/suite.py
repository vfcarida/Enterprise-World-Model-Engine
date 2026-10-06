"""Scientific benchmark suite runner for EWM Engine.

Executes the five canonical benchmark families, produces machine-readable reports,
and prints human-readable comparative tables with provenance fingerprints.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.families.constraint_stress import ConstraintStressBenchmark  # noqa: E402
from benchmarks.families.intervention_shift import InterventionShiftBenchmark  # noqa: E402
from benchmarks.families.long_horizon import LongHorizonBenchmark  # noqa: E402
from benchmarks.families.multi_agent_cascade import MultiAgentCascadeBenchmark  # noqa: E402
from benchmarks.families.rule_shift import RuleShiftBenchmark  # noqa: E402
from benchmarks.protocol import BenchmarkResult  # noqa: E402
from benchmarks.report import BenchmarkSuiteReport  # noqa: E402


def run_scientific_benchmark_suite(
    seed: int = 42,
    fast: bool = False,
    families: list[str] | None = None,
    include_neural: bool = True,
) -> BenchmarkSuiteReport:
    """Run the scientific benchmark families and return an aggregated report."""
    results: list[BenchmarkResult] = []

    # Scale sample count and horizons if fast mode requested
    n_samples = 20 if fast else 60
    horizon_long = 15 if fast else 40
    horizon_cascade = 6 if fast else 8

    selected = {
        f.lower()
        for f in (families or ["intervention", "rule", "constraint", "horizon", "cascade"])
    }

    # 1. InterventionShift
    if any(k in selected for k in ("intervention", "interventionshift")):
        bench_interv = InterventionShiftBenchmark(
            seed=seed,
            n_in_samples=n_samples,
            n_out_samples=n_samples,
            include_neural=include_neural,
        )
        results.append(bench_interv.run())

    # 2. RuleShift
    if any(k in selected for k in ("rule", "ruleshift")):
        bench_rule = RuleShiftBenchmark(
            seed=seed,
            n_samples_per_regime=n_samples,
            include_neural=include_neural,
        )
        results.append(bench_rule.run())

    # 3. ConstraintStress
    if any(k in selected for k in ("constraint", "constraintstress")):
        bench_stress = ConstraintStressBenchmark(
            seed=seed,
            n_samples=n_samples,
        )
        results.append(bench_stress.run())

    # 4. LongHorizon
    if any(k in selected for k in ("horizon", "longhorizon")):
        bench_horizon = LongHorizonBenchmark(
            seed=seed,
            horizon=horizon_long,
        )
        results.append(bench_horizon.run())

    # 5. MultiAgentCascade
    if any(k in selected for k in ("cascade", "multiagentcascade")):
        bench_cascade = MultiAgentCascadeBenchmark(
            seed=seed,
            horizon=horizon_cascade,
        )
        results.append(bench_cascade.run())

    return BenchmarkSuiteReport(
        suite_name="EWM_Scientific_Benchmark_Suite",
        timestamp=datetime.now(UTC).isoformat(),
        results=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EWM Engine Scientific Benchmark Suite (Research Instrument)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument(
        "--fast", action="store_true", help="Run fast configurations for CI / smoke testing"
    )
    parser.add_argument(
        "--json", type=str, default=None, help="Optional output path for JSON report"
    )
    parser.add_argument(
        "--no-neural",
        action="store_true",
        help="Exclude neural baseline dynamics even if PyTorch is installed",
    )
    parser.add_argument(
        "--family",
        nargs="*",
        default=None,
        help="Filter specific benchmark families to run (intervention, rule, constraint, horizon, cascade)",
    )

    args = parser.parse_args()

    report = run_scientific_benchmark_suite(
        seed=args.seed,
        fast=args.fast,
        families=args.family,
        include_neural=not args.no_neural,
    )
    print(report.to_text_table())

    if args.json:
        report.save_json(args.json)
        print(f"\nMachine-readable JSON report saved to: {args.json}")


if __name__ == "__main__":
    main()
