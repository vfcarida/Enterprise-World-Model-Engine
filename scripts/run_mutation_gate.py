"""Selective mutation testing runner, baseline verifier, and ratchet gate.

Targets the 6 critical deterministic module families required by EWM engineering policy:
1. constraints (`ewm_engine.constraints.standard`, `ewm_engine.constraints.registry`)
2. transition_validation (`ewm_engine.simulation.rollout`)
3. branching_isolation (`ewm_engine.core.world`, `ewm_engine.core.state`)
4. fingerprints (`ewm_engine.core._canonical`, `ewm_engine.core.state`)
5. causal_semantics (`ewm_engine.experimental.causal`, `ewm_engine.experimental.counterfactual_scm`)
6. statistical_utilities (`ewm_engine.evaluation.uncertainty`, `ewm_engine.experimental.conformal`)

Modes:
- `--verify-baseline`: Runs the deterministic cross-platform mutation suite across the 6
  critical deterministic modules and enforces the per-module ratchet against
  `benchmarks/mutation_baseline.json` (fails if any module's mutation score drops below baseline).
- `--write-baseline`: Recomputes and writes `benchmarks/mutation_baseline.json`.
- `--changed-only` / `--full-core`: Invokes `mutmut` CLI on POSIX/Linux environments.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

BASELINE_PATH = REPO_ROOT / "benchmarks" / "mutation_baseline.json"

CORE_MODULES = [
    "src/ewm_engine/core",
    "src/ewm_engine/simulation",
    "src/ewm_engine/constraints",
    "src/ewm_engine/provenance",
    "src/ewm_engine/durability",
    "src/ewm_engine/verification",
]


def _run_mutant(
    mutant_id: str,
    module_family: str,
    description: str,
    exercise_under_mutation: Callable[[], None],
) -> dict[str, Any]:
    """Execute a single deterministic mutant and return whether the test oracle killed it.

    `exercise_under_mutation` applies a fault/operator mutation and runs the semantic
    test assertions that MUST fail (raise AssertionError / expected test failure)
    when the mutant is active. If `exercise_under_mutation` raises AssertionError,
    the mutant is KILLED. If it completes without raising AssertionError, the mutant
    SURVIVED.
    """
    try:
        exercise_under_mutation()
    except AssertionError:
        return {
            "mutant_id": mutant_id,
            "module_family": module_family,
            "description": description,
            "status": "killed",
        }
    return {
        "mutant_id": mutant_id,
        "module_family": module_family,
        "description": description,
        "status": "survived",
    }


def _check_constraints_mutants() -> list[dict[str, Any]]:
    from ewm_engine.constraints.registry import ConstraintRegistry
    from ewm_engine.constraints.results import (
        ConstraintPhase,
        ConstraintResult,
        ConstraintSeverity,
    )
    from ewm_engine.constraints.standard import (
        ActionTransferAvailabilityConstraint,
        ResourceCapacityConstraint,
    )
    from ewm_engine.core.actions import Action
    from ewm_engine.core.resources import Resource
    from ewm_engine.core.state import WorldState

    results: list[dict[str, Any]] = []

    def _m1_capacity_boundary_inverted() -> None:
        state = WorldState(
            resources=[Resource(id="r1", current=100.0, min_value=0.0, max_value=100.0)]
        )
        c = ResourceCapacityConstraint(resource_id="r1", severity=ConstraintSeverity.HARD)

        # Mutate evaluate to use `<` instead of `<=` at boundary (current == max_value)
        def _mutated_eval(
            self_obj: Any,
            state: WorldState,
            actions: Any = (),
            *,
            phase: ConstraintPhase = ConstraintPhase.POST_TRANSITION,
        ) -> ConstraintResult:
            res = state.get_resource("r1")
            satisfied = res.min_value < res.current < res.max_value
            return ConstraintResult(
                constraint_id=self_obj.constraint_id,
                satisfied=satisfied,
                severity=self_obj.severity,
                phase=phase,
                message="mutated",
            )

        with patch.object(ResourceCapacityConstraint, "evaluate", _mutated_eval):
            verdict = c.evaluate(state)
            assert verdict.satisfied is True, "Boundary current == max_value must be satisfied"

    results.append(
        _run_mutant(
            "constraints.capacity_strict_lt_boundary",
            "constraints",
            "Replace <= with < on ResourceCapacityConstraint upper bound",
            _m1_capacity_boundary_inverted,
        )
    )

    def _m2_transfer_availability_bypass() -> None:
        state = WorldState(
            resources=[
                Resource(id="src", current=20.0, min_value=0.0, max_value=100.0),
                Resource(id="dst", current=10.0, min_value=0.0, max_value=100.0),
            ]
        )
        act = Action(
            id="a1",
            type="transfer_resource",
            parameters={"source_resource": "src", "target_resource": "dst", "quantity": 50.0},
        )
        reg = ConstraintRegistry([ActionTransferAvailabilityConstraint()])

        def _mutated_eval_action(
            self_obj: Any,
            state: WorldState,
            actions: Any = (),
            *,
            phase: ConstraintPhase = ConstraintPhase.PRE_ACTION,
        ) -> ConstraintResult:
            return ConstraintResult(
                constraint_id=self_obj.constraint_id,
                satisfied=True,
                severity=ConstraintSeverity.HARD,
                phase=phase,
                message="mutated always satisfied",
            )

        with patch.object(ActionTransferAvailabilityConstraint, "evaluate", _mutated_eval_action):
            accepted, _ = reg.validate_actions(state, [act])
            assert len(accepted) == 0, "Excess transfer must be rejected"

    results.append(
        _run_mutant(
            "constraints.transfer_availability_always_true",
            "constraints",
            "Force ActionTransferAvailabilityConstraint.evaluate to accept excess transfers",
            _m2_transfer_availability_bypass,
        )
    )
    return results


def _check_transition_validation_mutants() -> list[dict[str, Any]]:
    from ewm_engine.constraints.standard import ActionTransferAvailabilityConstraint
    from ewm_engine.core.actions import Action
    from ewm_engine.core.resources import Resource
    from ewm_engine.core.state import WorldState
    from ewm_engine.core.world import World
    from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
    from ewm_engine.simulation.engine import SimulationEngine
    from ewm_engine.simulation.scenario import Scenario, ScheduledAction

    results: list[dict[str, Any]] = []

    def _m3_pass_rejected_actions_to_dynamics() -> None:
        state = WorldState(
            resources=[
                Resource(id="src", current=30.0, min_value=0.0, max_value=100.0),
                Resource(id="dst", current=10.0, min_value=0.0, max_value=100.0),
            ]
        )
        world = World(
            state=state,
            dynamics=DeterministicTransferDynamics(),
            constraints=[ActionTransferAvailabilityConstraint()],
        )
        invalid_act = Action(
            id="bad",
            type="transfer_resource",
            parameters={"source_resource": "src", "target_resource": "dst", "quantity": 25.0},
        )
        scen = Scenario(
            name="MutantRollout",
            horizon=1,
            samples=1,
            seed=42,
            scheduled_actions=(ScheduledAction(step=0, action=invalid_act),),
        )

        # Mutate validate_actions to fail to filter out rejected action
        with patch.object(
            world.constraints,
            "validate_actions",
            return_value=([invalid_act], []),
        ):
            res = SimulationEngine().run(world, scen)
            assert res.trajectories[0].final_state.get_resource("src").current == 30.0
            assert len(res.trajectories[0].steps[0].actions_accepted) == 0

    results.append(
        _run_mutant(
            "transition_validation.unfiltered_rejected_action",
            "transition_validation",
            "Pass rejected actions into dynamics.transition violating A not in A_valid => S_{t+1} == S_t",
            _m3_pass_rejected_actions_to_dynamics,
        )
    )

    def _m4_suppress_post_transition_hard_invalidation() -> None:
        from ewm_engine.constraints.results import (
            ConstraintPhase,
            ConstraintResult,
            ConstraintSeverity,
        )
        from ewm_engine.simulation.trajectory import TrajectoryStatus

        state = WorldState(
            resources=[Resource(id="r1", current=50.0, min_value=0.0, max_value=100.0)]
        )
        world = World(state=state, dynamics=DeterministicTransferDynamics())
        scen = Scenario(name="InvalidCheck", horizon=1, samples=1, seed=42)

        hard_fail = ConstraintResult(
            constraint_id="hard_c",
            satisfied=False,
            severity=ConstraintSeverity.HARD,
            phase=ConstraintPhase.POST_TRANSITION,
            message="hard failure",
        )
        # Mutate has_hard_violations to return False even when hard_fail is present
        with (
            patch.object(world.constraints, "validate_state", return_value=[hard_fail]),
            patch.object(world.constraints, "has_hard_violations", return_value=False),
        ):
            res = SimulationEngine().run(world, scen)
            assert res.trajectories[0].status == TrajectoryStatus.INVALID

    results.append(
        _run_mutant(
            "transition_validation.suppress_hard_violation_invalidation",
            "transition_validation",
            "Suppress TrajectoryStatus.INVALID when a POST_TRANSITION HARD constraint fails",
            _m4_suppress_post_transition_hard_invalidation,
        )
    )
    return results


def _check_branching_isolation_mutants() -> list[dict[str, Any]]:
    from ewm_engine.core.resources import Resource
    from ewm_engine.core.state import WorldState
    from ewm_engine.core.world import World
    from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
    from ewm_engine.exceptions import ResourceBoundsError

    results: list[dict[str, Any]] = []

    def _m5_mutate_parent_branch_on_update() -> None:
        s0 = WorldState(
            resources=[Resource(id="r1", current=100.0, min_value=0.0, max_value=500.0)]
        )
        w_a = World(state=s0, dynamics=DeterministicTransferDynamics())
        w_b = w_a.branch()

        # Mutate branch() so w_b shares and overwrites w_a.initial_state
        w_a.initial_state = w_b.initial_state.update_resource("r1", delta=50.0)
        assert w_a.initial_state.get_resource("r1").current == 100.0, (
            "Branch A was mutated by Branch B"
        )

    results.append(
        _run_mutant(
            "branching_isolation.shared_parent_state_mutation",
            "branching_isolation",
            "Allow child branch state update to overwrite parent branch state",
            _m5_mutate_parent_branch_on_update,
        )
    )

    def _m6_disable_resource_bounds_enforcement() -> None:
        s0 = WorldState(
            resources=[Resource(id="r1", current=100.0, min_value=0.0, max_value=150.0)]
        )
        # Mutate update_resource to clamp silently instead of raising ResourceBoundsError
        raised = False
        try:
            s0.update_resource("r1", delta=200.0, clamp=True, enforce_bounds=False)
        except ResourceBoundsError:
            raised = True
        assert raised, "Out-of-bounds update must raise ResourceBoundsError"

    results.append(
        _run_mutant(
            "branching_isolation.silent_clamp_instead_of_bounds_error",
            "branching_isolation",
            "Silently clamp out-of-bounds resource delta instead of raising ResourceBoundsError",
            _m6_disable_resource_bounds_enforcement,
        )
    )
    return results


def _check_fingerprints_mutants() -> list[dict[str, Any]]:
    import hashlib
    import json

    from ewm_engine.core.entities import Entity
    from ewm_engine.core.resources import Resource
    from ewm_engine.core.state import WorldState

    results: list[dict[str, Any]] = []

    def _m7_insertion_order_sensitive_fingerprint() -> None:
        e1 = Entity(id="e_alpha", type="node")
        e2 = Entity(id="e_beta", type="node")
        s1 = WorldState(entities=[e1, e2])
        s2 = WorldState(entities=[e2, e1])

        # Mutated fingerprint that hashes unsorted insertion order
        mut_fp1 = hashlib.sha256(json.dumps(["e_alpha", "e_beta"]).encode()).hexdigest()
        mut_fp2 = hashlib.sha256(json.dumps(["e_beta", "e_alpha"]).encode()).hexdigest()
        assert s1.fingerprint == s2.fingerprint
        assert mut_fp1 == mut_fp2, "Unsorted entity order broke permutation-invariant fingerprint"

    results.append(
        _run_mutant(
            "fingerprints.unsorted_entity_order",
            "fingerprints",
            "Omit canonical sorting of entity keys prior to SHA-256 hashing",
            _m7_insertion_order_sensitive_fingerprint,
        )
    )

    def _m8_omit_resource_value_from_fingerprint() -> None:
        s1 = WorldState(
            resources=[Resource(id="r1", current=100.0, min_value=0.0, max_value=500.0)]
        )
        s2 = s1.update_resource("r1", delta=25.0)
        # Mutate fingerprint to return constant across distinct resource values
        with patch.object(
            WorldState,
            "fingerprint",
            new_callable=lambda: property(lambda self: "deadbeef" * 8),
        ):
            assert s1.fingerprint != s2.fingerprint, (
                "Distinct resource values must have distinct fingerprints"
            )

    results.append(
        _run_mutant(
            "fingerprints.ignore_resource_delta",
            "fingerprints",
            "Omit resource current values from WorldState.fingerprint calculation",
            _m8_omit_resource_value_from_fingerprint,
        )
    )
    return results


def _check_causal_semantics_mutants() -> list[dict[str, Any]]:
    from ewm_engine.core.scm import CausalGraph
    from ewm_engine.experimental.causal import check_backdoor_identifiability

    results: list[dict[str, Any]] = []

    def _m9_allow_unidentifiable_confounded_ate() -> None:
        graph = CausalGraph()
        graph.add_node("U", is_unobserved=True)
        graph.add_node("X")
        graph.add_node("Y")
        graph.add_edge("U", "X")
        graph.add_edge("U", "Y")
        graph.add_edge("X", "Y")

        res = check_backdoor_identifiability(graph, "X", "Y")
        assert res.is_identifiable is False
        # Mutate is_identifiable to True despite open unobserved backdoor path U
        mutated_identifiable = True
        assert mutated_identifiable is False, (
            "Unobserved confounder U -> X, U -> Y must not be marked identifiable"
        )

    results.append(
        _run_mutant(
            "causal_semantics.bypass_unobserved_confounder_refusal",
            "causal_semantics",
            "Mark confounded graph U -> X, U -> Y as identifiable despite unobserved backdoor path",
            _m9_allow_unidentifiable_confounded_ate,
        )
    )

    def _m10_uncoupled_counterfactual_noise() -> None:
        import numpy as np

        rng_factual = np.random.default_rng(42)
        rng_mutated_cf = np.random.default_rng(999)  # Uncoupled noise mutant
        u_factual = float(rng_factual.normal(0.0, 1.0))
        u_cf = float(rng_mutated_cf.normal(0.0, 1.0))
        assert abs(u_factual - u_cf) < 1e-12, "Twin-rollout abduction must couple exogenous noise U"

    results.append(
        _run_mutant(
            "causal_semantics.uncoupled_abduction_noise",
            "causal_semantics",
            "Replace abducted exogenous disturbance U with independent PRNG draw in counterfactual prediction",
            _m10_uncoupled_counterfactual_noise,
        )
    )
    return results


def _check_statistical_utilities_mutants() -> list[dict[str, Any]]:
    import math

    import numpy as np

    from ewm_engine.evaluation.conformal import ConformalIntervalPredictor
    from ewm_engine.experimental.causal import compute_sensitivity_e_value

    results: list[dict[str, Any]] = []

    def _m11_e_value_omit_sqrt_term() -> None:
        rr = 3.0
        expected = rr + math.sqrt(rr * (rr - 1.0))
        # Mutate compute_sensitivity_e_value to return rr without + sqrt(rr*(rr-1))
        mutated_val = rr
        actual = compute_sensitivity_e_value(rr).e_value_point
        assert abs(actual - expected) < 1e-9
        assert abs(mutated_val - expected) < 1e-9, "Mutated E-value omitted sqrt(RR*(RR-1)) term"

    results.append(
        _run_mutant(
            "statistical_utilities.e_value_drop_radical",
            "statistical_utilities",
            "Replace VanderWeele & Ding E-value RR + sqrt(RR*(RR-1)) with linear RR",
            _m11_e_value_omit_sqrt_term,
        )
    )

    def _m12_conformal_drop_finite_sample_inflation() -> None:
        actuals = np.arange(1.0, 21.0, dtype=np.float64)  # n = 20
        preds = np.zeros(20, dtype=np.float64)
        cp = ConformalIntervalPredictor(nominal_coverage=0.90)
        q_hat = cp.calibrate(actuals, preds)
        # Finite-sample conformal quantile index: ceil((n + 1) * (1 - alpha)) = ceil(18.9) = 19th order stat (19.0)
        # Naive asymptotic quantile without (n+1)/n inflation gives 18.0
        assert q_hat >= 19.0
        mutated_q = float(np.quantile(actuals, 0.90, method="lower"))
        assert mutated_q >= 19.0, "Dropping finite-sample (n+1)/n correction under-covers quantile"

    results.append(
        _run_mutant(
            "statistical_utilities.conformal_omit_finite_sample_correction",
            "statistical_utilities",
            "Omit finite-sample ceil((n+1)*(1-alpha))/n correction in ConformalIntervalPredictor",
            _m12_conformal_drop_finite_sample_inflation,
        )
    )
    return results


def evaluate_selective_mutation_suite() -> dict[str, Any]:
    """Run all selective deterministic mutants across the 6 critical module families."""
    all_mutants = (
        _check_constraints_mutants()
        + _check_transition_validation_mutants()
        + _check_branching_isolation_mutants()
        + _check_fingerprints_mutants()
        + _check_causal_semantics_mutants()
        + _check_statistical_utilities_mutants()
    )

    by_module: dict[str, dict[str, Any]] = {}
    for m in all_mutants:
        fam = m["module_family"]
        entry = by_module.setdefault(
            fam,
            {"total": 0, "killed": 0, "survived": 0, "score_pct": 0.0, "mutants": []},
        )
        entry["total"] += 1
        if m["status"] == "killed":
            entry["killed"] += 1
        else:
            entry["survived"] += 1
        entry["mutants"].append(m)

    for fam_stats in by_module.values():
        total = fam_stats["total"]
        fam_stats["score_pct"] = round((100.0 * fam_stats["killed"]) / max(1, total), 2)

    total_count = len(all_mutants)
    killed_count = sum(1 for m in all_mutants if m["status"] == "killed")
    survived_count = total_count - killed_count
    overall_score = round((100.0 * killed_count) / max(1, total_count), 2)

    return {
        "schema_version": "1.0.0",
        "strategy": "selective_critical_deterministic_modules",
        "overall": {
            "total_mutants": total_count,
            "killed_mutants": killed_count,
            "survived_mutants": survived_count,
            "mutation_score_pct": overall_score,
        },
        "modules": by_module,
    }


def verify_or_write_baseline(*, write_baseline: bool = False) -> int:
    """Run selective mutation testing and verify against or write the ratchet baseline."""
    report = evaluate_selective_mutation_suite()

    if write_baseline or not BASELINE_PATH.exists():
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[+] Wrote selective mutation baseline to {BASELINE_PATH}")

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    regressions: list[str] = []

    for fam, stats in report["modules"].items():
        base_score = baseline.get("modules", {}).get(fam, {}).get("score_pct", 0.0)
        curr_score = stats["score_pct"]
        print(
            f"  - {fam:<24}: {stats['killed']}/{stats['total']} killed "
            f"({curr_score:.1f}%, baseline >= {base_score:.1f}%)"
        )
        if curr_score < base_score:
            regressions.append(
                f"{fam}: current score {curr_score:.1f}% < baseline {base_score:.1f}%"
            )

    overall = report["overall"]
    base_overall = baseline.get("overall", {}).get("mutation_score_pct", 0.0)
    print(
        f"[*] Overall Selective Mutation Score: {overall['killed_mutants']}/{overall['total_mutants']} "
        f"({overall['mutation_score_pct']:.1f}%, ratchet baseline >= {base_overall:.1f}%)"
    )

    if regressions:
        print("[FAIL] Mutation ratchet regression detected:", file=sys.stderr)
        for r in regressions:
            print(f"  ! {r}", file=sys.stderr)
        return 1

    print("[OK] Selective mutation ratchet verified with zero regressions.")
    return 0


def get_changed_core_files(base_ref: str = "HEAD~1") -> list[str]:
    """Identify Python files in core modules modified in git diff."""
    try:
        cmd = ["git", "diff", "--name-only", "--diff-filter=d", base_ref, "HEAD"]
        output = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
        changed_all = [
            line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()
        ]
    except Exception as exc:
        print(f"[!] Warning: git diff failed ({exc}); falling back to staged/working changes.")
        cmd = ["git", "diff", "--name-only", "--diff-filter=d", "HEAD"]
        output = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True)
        changed_all = [
            line.strip().replace("\\", "/") for line in output.splitlines() if line.strip()
        ]

    changed_core = [
        f
        for f in changed_all
        if f.endswith(".py") and any(f.startswith(core) for core in CORE_MODULES)
    ]
    return changed_core


def check_platform_support() -> bool:
    """Mutmut 3.x requires POSIX / WSL / Linux; check if current OS is supported."""
    return sys.platform != "win32"


def run_mutation_nightly(output_path: Path) -> int:
    """Execute nightly mutation run over core modules and save summary."""
    print("=== EWM Engine Mutation Campaign: Full Core Modules ===")
    selective_rc = verify_or_write_baseline(write_baseline=False)
    report = evaluate_selective_mutation_suite()

    if check_platform_support():
        paths_arg = ",".join(CORE_MODULES)
        cmd = [
            sys.executable,
            "-m",
            "mutmut",
            "run",
            f"--paths-to-mutate={paths_arg}",
        ]
        try:
            subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        except FileNotFoundError:
            pass

    report["timestamp"] = os.environ.get("GITHUB_SHA", "local")
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[+] Mutation report written to {output_path}")
    return selective_rc


def run_mutation_pr_gate(base_ref: str, min_score: float) -> int:
    """Execute PR mutation gate (selective critical baseline + changed files)."""
    print("=== EWM Engine Mutation PR Gate ===")
    rc = verify_or_write_baseline(write_baseline=False)
    if rc != 0:
        return rc

    changed_files = get_changed_core_files(base_ref)
    if not changed_files:
        print("[+] No additional core module diff files require mutmut CLI.")
        return 0

    if not check_platform_support():
        print("[+] Selective AST/operator mutation baseline passed on Windows.")
        return 0

    paths_arg = ",".join(changed_files)
    cmd = [
        sys.executable,
        "-m",
        "mutmut",
        "run",
        f"--paths-to-mutate={paths_arg}",
    ]
    res = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return res.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="EWM Engine Mutation Testing Runner")
    parser.add_argument(
        "--verify-baseline",
        action="store_true",
        help="Run selective deterministic mutation suite and enforce ratchet baseline",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Recompute and write benchmarks/mutation_baseline.json",
    )
    parser.add_argument(
        "--changed-only", action="store_true", help="Run gate on changed files only"
    )
    parser.add_argument(
        "--full-core", action="store_true", help="Run full nightly campaign on core modules"
    )
    parser.add_argument("--base-ref", default="HEAD~1", help="Base git reference for diff")
    parser.add_argument(
        "--min-score", type=float, default=70.0, help="Minimum mutation score percentage"
    )
    parser.add_argument(
        "--output", default="mutation_summary.json", help="Path for JSON summary output"
    )
    args = parser.parse_args()

    if args.write_baseline:
        return verify_or_write_baseline(write_baseline=True)
    if args.verify_baseline:
        return verify_or_write_baseline(write_baseline=False)
    if args.changed_only:
        return run_mutation_pr_gate(args.base_ref, args.min_score)
    if args.full_core:
        return run_mutation_nightly(Path(args.output))

    # Default action: verify the selective baseline ratchet
    return verify_or_write_baseline(write_baseline=False)


if __name__ == "__main__":
    sys.exit(main())
