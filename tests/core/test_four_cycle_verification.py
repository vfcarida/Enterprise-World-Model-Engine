"""Four-Cycle Iterative Verification Suite (`EWM-009`).

Codifies the four independent verification perspectives required by `EWM-009`:
1. Cycle 1 — Contract & Integration: End-to-end Finance and SWE domain workflows and core contracts.
2. Cycle 2 — Adversarial & Failure-Oriented: Malformed inputs (`NaN`/`Inf` timestamps, negative steps,
   cyclic SWE dependencies, `NaN`/`sNaN`/`+Inf`/`float` financial values, unbalanced postings, and
   massive 1D `1e160` and multivariate 2D `[1e200, 1e200]` OOD neural inputs), verifying safe failure
   and `AbstainException` / constraint rejection (`C2-01`..`C2-04`).
3. Cycle 3 — Scientific & Independent Oracle: Independent `fractions.Fraction` exact rational ledger
   oracle across 200 steps (`0.00` drift), structural DAG zero-temporal-leakage oracle, and fair
   `N=30` benchmark evaluation against `RandomAgent` with identical exogenous demand shocks (`C3-01`).
4. Cycle 4 — Clean-Environment & Publication: Isolated temporary directory execution of the README
   quickstart, package metadata verification, and repository hygiene sweep.
"""

from __future__ import annotations

import importlib.metadata
import math
import subprocess
import sys
from decimal import ROUND_HALF_EVEN, Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ewm_engine.causal import CausalGraph, TransitionEngine
from ewm_engine.core.actions import Action
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import Entity, WorldState
from ewm_engine.core.time import (
    BranchManager,
    EventLog,
    TemporalMonotonicityError,
    collect_mutable_container_ids,
)
from ewm_engine.domains.finance import (
    ZERO_DECIMAL,
    AccountCategory,
    FinancialState,
    JournalEntry,
    JournalLine,
    LedgerAccount,
    build_corporate_treasury_engine,
    build_journal_action,
    run_liquidity_drain_stress_test,
)
from ewm_engine.domains.swe import (
    CircularDependencyError,
    CodeFileNode,
    SWEGraphCapacityError,
    SWEState,
    build_swe_action,
    build_swe_agent_engine,
    create_default_swe_repository_state,
    run_concurrent_agent_intervention_scenario,
)
from ewm_engine.domains.swe.state import _FallbackDiGraph
from ewm_engine.dynamics.calibration import (
    AbstainException,
    MahalanobisOODAndDriftDetector,
    Predictor,
)
from ewm_engine.dynamics.learned import TransitionDataset, TransitionSample
from ewm_engine.evaluation.evaluator import (
    MIN_SIGNIFICANCE_SEEDS,
    EnvironmentSpec,
    EWM_Evaluator,
)
from ewm_engine.exceptions import InvalidWorldStateError

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_cycle_1_contract_and_domain_integration() -> None:
    """Cycle 1: Execute end-to-end Finance and SWE domain workflows and core state/router contracts."""
    # 1. End-to-End Finance Domain Workflow
    fin_report = run_liquidity_drain_stress_test()
    assert fin_report.all_steps_balanced is True
    assert fin_report.max_abs_discrepancy_across_all_branches == ZERO_DECIMAL
    assert ("market_stress", "deposit_outflow_amount") in fin_report.severed_edges_on_stress_branch

    fin_engine, fin_state = build_corporate_treasury_engine()
    fin_world = fin_state.to_world_state()
    valid_journal = build_journal_action(
        "c1_settle_deposits",
        debits={"demand_deposits": Decimal("25000.00")},
        credits={"cash": Decimal("25000.00")},
        description="Settle institutional deposit withdrawal",
    )
    fin_outcome = fin_engine.step(state=fin_world, action=valid_journal)
    assert not fin_outcome.pre_rule_rejected
    assert not fin_outcome.post_rule_rejected
    next_fin = FinancialState.from_world_state(fin_outcome.next_state)
    assert next_fin.is_balanced
    assert next_fin.accounting_discrepancy == ZERO_DECIMAL
    assert next_fin.meets_liquidity_constraints

    # 2. End-to-End SWE Domain Workflow
    swe_trace = run_concurrent_agent_intervention_scenario()
    assert swe_trace.cyclic_injection_rejected is True
    assert "circular dependency" in swe_trace.cyclic_rejection_reason.lower()
    assert len(swe_trace.branch_agent_a_merges_first.merged_prs) >= 1
    assert len(swe_trace.branch_agent_b_modifies_first.stale_observations) >= 1


def test_cycle_2_adversarial_and_failure_oriented_fault_injection() -> None:
    """Cycle 2: Inject malformed/adversarial inputs across Time, Finance, SWE, and Learned OOD boundaries."""
    # 1. C2-03: Negative time steps and non-finite (NaN / +Inf / -Inf) timestamps
    with pytest.raises(ValidationError):
        WorldState(step=-1, timestamp=0.0)

    for non_finite_ts in (float("nan"), float("inf"), float("-inf")):
        bad_ts_state = WorldState(step=0, timestamp=non_finite_ts)
        with pytest.raises(InvalidWorldStateError):
            _ = bad_ts_state.state_hash
        with pytest.raises(TemporalMonotonicityError):
            EventLog(bad_ts_state)

    base_state = WorldState(
        step=0,
        timestamp=0.0,
        entities={"e1": Entity(id="e1", type="service")},
        resources={"r1": Resource(id="r1", current=50.0, min_value=-1e300, max_value=1e300)},
    )
    with pytest.raises(ValueError):
        base_state.advance_time(delta_t=-0.5, steps=1)
    with pytest.raises(ValueError):
        base_state.advance_time(delta_t=1.0, steps=0)
    for non_finite_dt in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            base_state.advance_time(delta_t=non_finite_dt, steps=1)

    log = EventLog(base_state)
    with pytest.raises(TemporalMonotonicityError):
        log.append(base_state.model_copy(update={"step": 0, "timestamp": 1.0}))
    with pytest.raises(TemporalMonotonicityError):
        log.append(base_state.model_copy(update={"step": 1, "timestamp": -1.0}))
    with pytest.raises(TemporalMonotonicityError):
        log.append(base_state.model_copy(update={"step": 1, "timestamp": float("nan")}))

    # 2. C2-04: Malformed Financial inputs (float, NaN, sNaN, +Infinity, negative postings, $0.01 imbalance)
    for bad_val in (100.0, float("nan"), float("inf")):
        with pytest.warns(DeprecationWarning), pytest.raises((TypeError, ValidationError)):
            LedgerAccount(
                account_id="cash",
                name="Cash",
                category=AccountCategory.ASSET,
                balance=bad_val,  # type: ignore[arg-type]
            )

    for non_finite_dec in (
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        "NaN",
        "Infinity",
    ):
        with pytest.raises((ValueError, ValidationError)):
            LedgerAccount(
                account_id="cash",
                name="Cash",
                category=AccountCategory.ASSET,
                balance=Decimal("100.00"),
                min_balance=non_finite_dec,  # type: ignore[arg-type]
            )

    # Confirm Decimal("-Infinity") is still accepted as an unbounded lower floor (None)
    unbounded_acc = LedgerAccount(
        account_id="cash",
        name="Cash",
        category=AccountCategory.ASSET,
        balance=Decimal("100.00"),
        min_balance=Decimal("-Infinity"),
    )
    assert unbounded_acc.min_balance is None

    with pytest.raises((ValueError, ValidationError)):
        JournalLine(account_id="cash", debit=Decimal("-10.00"), credit=ZERO_DECIMAL)

    fin_engine, fin_state = build_corporate_treasury_engine()
    fin_world = fin_state.to_world_state()
    imbalanced_act = build_journal_action(
        "adv_imbalance_1_cent",
        debits={"demand_deposits": Decimal("100.01")},
        credits={"cash": Decimal("100.00")},
    )
    imbalanced_outcome = fin_engine.step(state=fin_world, action=imbalanced_act)
    assert imbalanced_outcome.pre_rule_rejected is True
    rejected_fin = FinancialState.from_world_state(imbalanced_outcome.next_state)
    assert rejected_fin.is_balanced
    assert rejected_fin.accounting_discrepancy == ZERO_DECIMAL

    # 3. Adversarial SWE inputs (self-loop, multi-node cycle, capacity overflow)
    with pytest.raises(CircularDependencyError):
        SWEState(
            files={"a.py": CodeFileNode(file_path="a.py")},
            dependencies={"a.py": ("a.py",)},
        )

    with pytest.raises(SWEGraphCapacityError):
        SWEState(
            files={f"f_{i}.py": CodeFileNode(file_path=f"f_{i}.py") for i in range(15)},
            dependencies={},
            max_graph_nodes=10,
        )

    swe_engine = build_swe_agent_engine(raise_on_violation=False)
    swe_world = create_default_swe_repository_state().to_world_state()
    cycle_action = build_swe_action(
        "adv_cycle_edge",
        "ADD_DEPENDENCY",
        parameters={"source_file": "src/core/schema.py", "target_file": "src/api/OrderController.py"},
    )
    cycle_outcome = swe_engine.step(state=swe_world, action=cycle_action)
    assert cycle_outcome.pre_rule_rejected is True

    # 4. C2-01 & C2-02: Massive 1D (1e160) and 2D ([1e200, 1e200]) OOD Neural Inputs
    samples = [
        TransitionSample(
            state=WorldState(
                step=0,
                timestamp=0.0,
                resources={
                    "r1": Resource(id="r1", current=v, min_value=-1e300, max_value=1e300)
                },
            ),
            actions=(),
            next_state=WorldState(
                step=1,
                timestamp=1.0,
                resources={
                    "r1": Resource(
                        id="r1", current=v + 1.0, min_value=-1e300, max_value=1e300
                    )
                },
            ),
        )
        for v in (10.0, 11.0, 9.0, 10.5)
    ]
    calibrated_pred = Predictor(lambda s, a: {"r1": 1.0}, ood_threshold=5.0, threshold=10.0)
    calibrated_pred.calibrate(TransitionDataset(samples))

    ood_1d_state = WorldState(
        step=0,
        timestamp=0.0,
        resources={"r1": Resource(id="r1", current=1e160, min_value=-1e300, max_value=1e300)},
    )
    with pytest.raises(AbstainException) as exc_1d:
        calibrated_pred.predict(ood_1d_state)
    assert exc_1d.value.ood_score > 5.0

    # Multivariate 2D OOD ([1e200, 1e200]) with correlated in-distribution features
    det_2d = MahalanobisOODAndDriftDetector(feature_keys=("r1", "r2"), ood_threshold=5.0)
    det_2d.fit(np.array([[1.0, 1.0], [2.0, 2.1], [3.0, 2.9], [4.0, 4.2]]))
    pred_2d = Predictor(
        lambda s, a: {"r1": 1.0, "r2": 1.0},
        ood_detector=det_2d,
        ood_threshold=5.0,
        threshold=10.0,
    )
    ood_2d_state = WorldState(
        step=0,
        timestamp=0.0,
        resources={
            "r1": Resource(id="r1", current=1e200, min_value=-1e300, max_value=1e300),
            "r2": Resource(id="r2", current=1e200, min_value=-1e300, max_value=1e300),
        },
    )
    with pytest.raises(AbstainException) as exc_2d:
        pred_2d.predict(ood_2d_state)
    assert exc_2d.value.ood_score > 1e190 or math.isinf(exc_2d.value.ood_score)


def test_cycle_3_scientific_and_independent_oracle() -> None:
    """Cycle 3: Verify exact rational ledger oracle, zero temporal/causal leakage, and fair RandomAgent benchmark."""
    # 1. Independent Exact Rational Oracle (fractions.Fraction) vs FinancialState across 200 steps
    fin_state = FinancialState(
        step=0,
        timestamp=Decimal("0.00"),
        min_total_liquidity=Decimal("100000.00"),
        accounts={
            "operating_cash": LedgerAccount(
                account_id="operating_cash",
                name="Operating Cash",
                category=AccountCategory.ASSET,
                balance=Decimal("1000000.0000"),
                min_balance=Decimal("100000.0000"),
                is_liquid=True,
            ),
            "treasury_reserve": LedgerAccount(
                account_id="treasury_reserve",
                name="Treasury Reserve",
                category=AccountCategory.ASSET,
                balance=Decimal("500000.0000"),
                min_balance=Decimal("0.0000"),
                is_liquid=True,
            ),
            "senior_debt": LedgerAccount(
                account_id="senior_debt",
                name="Senior Debt",
                category=AccountCategory.LIABILITY,
                balance=Decimal("600000.0000"),
            ),
            "retained_earnings": LedgerAccount(
                account_id="retained_earnings",
                name="Retained Earnings",
                category=AccountCategory.EQUITY,
                balance=Decimal("900000.0000"),
            ),
        },
    )

    oracle_balances: dict[str, Fraction] = {
        aid: Fraction(str(acc.balance)) for aid, acc in fin_state.accounts.items()
    }
    float_balances: dict[str, float] = {
        aid: float(acc.balance) for aid, acc in fin_state.accounts.items()
    }

    primes = (3, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    quantum = Decimal("0.0001")

    for step_idx in range(1, 201):
        p = primes[step_idx % len(primes)]
        raw_dec = (Decimal(1000 + step_idx * 17) / Decimal(p)).quantize(
            quantum, rounding=ROUND_HALF_EVEN
        )
        raw_frac = Fraction(str(raw_dec))
        raw_flt = (1000.0 + step_idx * 17.0) / float(p)

        entry = JournalEntry(
            entry_id=f"oracle_step_{step_idx}",
            step=step_idx,
            lines=(
                JournalLine(account_id="treasury_reserve", debit=raw_dec, credit=ZERO_DECIMAL),
                JournalLine(account_id="operating_cash", debit=ZERO_DECIMAL, credit=raw_dec),
            ),
        )
        fin_state = fin_state.apply_journal_entry(entry)
        oracle_balances["treasury_reserve"] += raw_frac
        oracle_balances["operating_cash"] -= raw_frac
        float_balances["treasury_reserve"] += raw_flt
        float_balances["operating_cash"] -= raw_flt

        # Exact conservation and exact agreement with the independent Fraction oracle at every step
        assert fin_state.is_balanced
        assert fin_state.accounting_discrepancy == ZERO_DECIMAL
        for aid, acc in fin_state.accounts.items():
            assert Fraction(str(acc.balance)) == oracle_balances[aid]

    # 2. Structural Graph & Temporal Branch Isolation Oracle (Zero Temporal/Cross-Branch Leakage)
    scm_state = WorldState(
        resources={
            "macro_index": Resource(id="macro_index", current=100.0, min_value=0.0, max_value=1000.0),
            "credit_spread": Resource(id="credit_spread", current=150.0, min_value=0.0, max_value=1000.0),
            "cash_flow": Resource(id="cash_flow", current=700.0, min_value=0.0, max_value=2000.0),
        }
    )
    graph = CausalGraph()
    for node in ("macro_index", "credit_spread", "cash_flow"):
        graph.add_node(node)
    graph.add_edge("macro_index", "credit_spread")
    graph.add_edge("credit_spread", "cash_flow")

    scm_engine = TransitionEngine(
        initial_state=scm_state,
        causal_graph=graph,
        structural_equations={
            "macro_index": lambda parents, actions, u: 100.0,
            "credit_spread": lambda parents, actions, u: parents.get("macro_index", 0.0) * 1.5,
            "cash_flow": lambda parents, actions, u: 1000.0 - parents.get("credit_spread", 0.0) * 2.0,
        },
    )
    mutilated = scm_engine.branch("do_spread").do(
        Action(id="fix_spread", type="do", parameters={"target": "credit_spread", "value": 50.0})
    )
    assert ("macro_index", "credit_spread") not in mutilated.causal_graph.edges
    assert ("credit_spread", "cash_flow") in mutilated.causal_graph.edges
    assert ("macro_index", "credit_spread") in scm_engine.causal_graph.edges

    bm = BranchManager(fin_state.to_world_state(), root_branch_id="main", full_canonical_hashes=True)
    bm.spawn_branch("counterfactual_a", parent_branch_id="main", at_step=fin_state.step)
    bm.spawn_branch("counterfactual_b", parent_branch_id="main", at_step=fin_state.step)
    assert collect_mutable_container_ids(
        bm.get_head_state("counterfactual_a")
    ).isdisjoint(collect_mutable_container_ids(bm.get_head_state("counterfactual_b")))

    # 3. Fair Benchmark Suite Comparison against RandomAgent (N=30 seeds, identical exogenous shocks)
    evaluator = EWM_Evaluator(EnvironmentSpec(horizon_steps=10))
    report = evaluator.evaluate(n_seeds=MIN_SIGNIFICANCE_SEEDS, master_seed=42)
    assert report.significance_claim_permitted is True

    ewm_summary = report.agent_summaries["EWMPolicyAgent"]
    rnd_summary = report.agent_summaries["RandomAgent"]
    assert ewm_summary.constraint_violation_rate.mean == 0.0
    assert ewm_summary.predictive_accuracy.mean > rnd_summary.predictive_accuracy.mean
    assert ewm_summary.cumulative_reward.mean > rnd_summary.cumulative_reward.mean

    vs_random = {
        comp.metric_name: comp
        for comp in report.pairwise_comparisons
        if comp.reference_agent == "EWMPolicyAgent" and comp.baseline_agent == "RandomAgent"
    }
    assert vs_random["predictive_accuracy"].is_statistically_significant is True
    assert vs_random["predictive_accuracy"].probability_of_improvement == 1.0
    assert vs_random["constraint_violation_rate"].is_statistically_significant is True


def test_cycle_4_clean_environment_quickstart_and_hygiene_sweep(tmp_path: Path) -> None:
    """Cycle 4: Execute README quickstart in an isolated temporary directory and sweep for artifact residue."""
    # 1. Verify installed package metadata and pure-Python SWE DAG fallback (C4-01)
    dist_version = importlib.metadata.version("ewm-engine")
    assert dist_version == "1.0.0"

    fallback_dag = _FallbackDiGraph()
    fallback_dag.add_node("a.py", revision=2)
    fallback_dag.add_edge("a.py", "b.py")
    assert fallback_dag.number_of_nodes() == 2
    assert fallback_dag.number_of_edges() == 1
    assert fallback_dag.has_edge("a.py", "b.py")
    reconstructed = SWEState.from_networkx_dag(fallback_dag)
    assert reconstructed.files["a.py"].revision == 2
    assert reconstructed.dependencies["a.py"] == ("b.py",)

    # 2. Execute README quickstart script with cwd set to an isolated clean temporary directory
    (tmp_path / "README.md").write_text(
        (REPO_ROOT / "README.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    readme_runner = REPO_ROOT / "scripts" / "test_readme_snippets.py"
    proc = subprocess.run(
        [sys.executable, str(readme_runner)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert (
        proc.returncode == 0
    ), f"README quickstart failed in clean tmp_path:\n{proc.stderr}\n{proc.stdout}"
    assert "All README Python code snippets executed successfully!" in proc.stdout

    # 3. Sweep repository for leftover development artifacts (.bak, .tmp, .orig, .swp)
    forbidden_globs = ("*.bak", "*.tmp", "*.orig", "*.swp", "*~")
    leftover_files: list[str] = []
    for pattern in forbidden_globs:
        for match in REPO_ROOT.rglob(pattern):
            rel_parts = match.relative_to(REPO_ROOT).parts
            if any(
                p.startswith(".venv") or p.startswith(".git") or p == "__pycache__"
                for p in rel_parts
            ):
                continue
            leftover_files.append(str(match.relative_to(REPO_ROOT)))

    assert not leftover_files, f"Leftover scratch/backup files found: {leftover_files}"
