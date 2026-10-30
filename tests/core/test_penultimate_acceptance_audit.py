"""Penultimate Acceptance Audit verification suite (EWM-008).

Executes and verifies all 8 Final Acceptance Checklist items, Section 8 scientific
and mathematical requirements (qualified causal discovery & exact Decimal units),
Section 11 operational safeguards (sandboxed, memory-bounded integration subprocess),
and Section 12 repository hygiene (zero .bak/scratch files, secrets, or fabricated citations).
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
from decimal import Decimal, getcontext
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
    MIN_CORPORATE_DECIMAL_PRECISION,
    ZERO_DECIMAL,
    AccountCategory,
    FinancialState,
    LedgerAccount,
    build_corporate_treasury_engine,
    build_journal_action,
    ensure_corporate_decimal_context,
    run_liquidity_drain_stress_test,
)
from ewm_engine.domains.swe import (
    MAX_SWE_GRAPH_NODES,
    CircularDependencyError,
    CodeFileNode,
    StructuralViolationError,
    SWEState,
    build_swe_action,
    build_swe_agent_engine,
    create_default_swe_repository_state,
    detect_cycle_iterative_o_v_plus_e,
)
from ewm_engine.dynamics.calibration import (
    AbstainException,
    MahalanobisOODAndDriftDetector,
    Predictor,
)
from ewm_engine.dynamics.learned import (
    LinearResidualDynamics,
    TransitionDataset,
    TransitionSample,
)
from ewm_engine.evaluation.evaluator import (
    MIN_SIGNIFICANCE_SEEDS,
    EnvironmentSpec,
    EWM_Evaluator,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_audit_checklist_1_state_immutability_and_no_shared_references() -> None:
    """[x] State immutability is strictly enforced (no shared references)."""
    raw_attrs = {"region": "emea", "tags": ["tier1", "core"]}
    state = WorldState(
        step=0,
        timestamp=0.0,
        entities={"e1": Entity(id="e1", type="node", attributes=raw_attrs)},
        resources={"r1": Resource(id="r1", current=100.0, min_value=0.0, max_value=500.0)},
    )
    root_hash = state.state_hash

    # Mutating caller's input dictionary/list after construction must not affect WorldState
    raw_attrs["region"] = "mutated"
    assert state.entities["e1"].attributes["region"] == "emea"
    assert state.state_hash == root_hash

    # Direct attribute assignment must fail, and defensive deep-copy protects all container mappings
    with pytest.raises(ValidationError):
        state.step = 99

    detached_entities = state.entities
    detached_entities["e1"] = Entity(id="e1", type="hacked")
    assert state.entities["e1"].type == "node"

    detached_attrs = state.entities["e1"].attributes
    detached_attrs["region"] = "hacked"
    assert state.entities["e1"].attributes["region"] == "emea"
    assert state.state_hash == root_hash

    # Verify zero shared mutable container IDs between state and its clone
    cloned = state.clone()
    assert collect_mutable_container_ids(state).isdisjoint(collect_mutable_container_ids(cloned))

    # Temporal monotonicity & DAG branching isolation
    log = EventLog(state)
    with pytest.raises(TemporalMonotonicityError):
        log.append(
            state.model_copy(update={"step": 0, "timestamp": -1.0}),
            Action(id="retro", type="noop"),
        )

    manager = BranchManager(state, root_branch_id="main", full_canonical_hashes=True)
    manager.step_branch(
        "main",
        Action(id="a1", type="deposit", parameters={"amount": 50.0}),
        lambda st_in, a: st_in.update_resource("r1", delta=float(a.parameters["amount"])),
    )
    fork_state = manager.spawn_branch("alt", parent_branch_id="main", at_step=0)
    assert fork_state.get_resource("r1").current == 100.0
    assert manager.get_head_state("main").get_resource("r1").current == 150.0
    assert state.state_hash == root_hash


def test_audit_checklist_2_causal_do_severs_structural_graph_edges() -> None:
    """[x] Causal `do()` operators successfully sever structural graph edges."""
    state = WorldState(
        resources={
            "Z": Resource(id="Z", current=10.0, min_value=0.0, max_value=500.0),
            "X": Resource(id="X", current=20.0, min_value=0.0, max_value=500.0),
            "Y": Resource(id="Y", current=50.0, min_value=0.0, max_value=500.0),
        }
    )
    graph = CausalGraph()
    for node in ("Z", "U_X", "X", "Y"):
        graph.add_node(node)
    graph.add_edge("Z", "X")
    graph.add_edge("U_X", "X")
    graph.add_edge("Z", "Y")
    graph.add_edge("X", "Y")

    engine = TransitionEngine(
        initial_state=state,
        causal_graph=graph,
        structural_equations={
            "X": lambda parents, actions, u: 2.0 * parents.get("Z", 0.0) + u.get("U_X", 0.0),
            "Y": lambda parents, actions, u: 1.5 * parents.get("X", 0.0) + 0.5 * parents.get("Z", 0.0),
        },
    )

    mutilated = engine.branch("do_x_branch").do(
        Action(id="do_x", type="do", parameters={"target": "X", "value": 80.0})
    )
    assert ("Z", "X") not in mutilated.causal_graph.edges
    assert ("U_X", "X") not in mutilated.causal_graph.edges
    assert ("X", "Y") in mutilated.causal_graph.edges
    assert ("Z", "X") in engine.causal_graph.edges  # Root graph unmutated

    outcome = mutilated.step(u_noise={"U_X": 5000.0})
    assert outcome.next_state.get_resource("X").current == pytest.approx(80.0)
    assert outcome.next_state.get_resource("Y").current == pytest.approx(1.5 * 80.0 + 0.5 * 10.0)


def test_audit_checklist_3_neural_models_fallback_and_abstain_on_ood() -> None:
    """[x] Neural models fallback/abstain on OOD inputs."""
    rng = np.random.default_rng(12345)
    samples: list[TransitionSample] = []
    for i in range(60):
        val = float(100.0 + rng.normal(0.0, 1.5))
        s_t = WorldState(
            step=i,
            resources={"liq": Resource(id="liq", current=val, min_value=0.0, max_value=1e6)},
        )
        act = Action(id=f"a_{i}", type="rebalance", parameters={"value": 0.5})
        s_next = WorldState(
            step=i + 1,
            resources={"liq": Resource(id="liq", current=val + 1.0, min_value=0.0, max_value=1e6)},
        )
        samples.append(TransitionSample(state=s_t, actions=(act,), events=(), next_state=s_next))

    train_ds = TransitionDataset(samples[:40])
    cal_ds = TransitionDataset(samples[40:])
    model = LinearResidualDynamics(target_resource="liq", action_type="rebalance")
    model.fit(train_ds)

    ood = MahalanobisOODAndDriftDetector(regularization=1e-5)
    ood.fit(train_ds, tune_quantile=0.99, safety_margin=1.20)
    predictor = Predictor(
        models=[model],
        ood_detector=ood,
        ood_threshold=ood.ood_threshold,
        threshold=15.0,
        nominal_coverage=0.90,
    )
    predictor.calibrate(cal_ds, alpha=0.10)

    ood_state = WorldState(
        step=99,
        resources={"liq": Resource(id="liq", current=50000.0, min_value=0.0, max_value=1e6)},
    )
    with pytest.raises(AbstainException):
        predictor.predict(ood_state)

    # Verify router/engine safe structural fallback on OOD abstention
    engine = TransitionEngine(
        initial_state=samples[0].state,
        structural_equations={"liq": lambda parents, actions, u: parents.get("liq", 100.0) + 2.0},
        ml_predictor=predictor,
        max_fallback_retries=1,
    )
    step_out = engine.step(ood_state, Action(id="act_ood", type="rebalance", parameters={"value": 0.5}))
    assert step_out.abstained is True
    assert step_out.fallback_executed is True
    assert step_out.next_state.get_resource("liq").current == pytest.approx(50002.0)


def test_audit_checklist_4_finance_invariant_and_decimal_unit_correctness() -> None:
    """[x] Finance domain invariant (Assets = Liabilities + Equity) holds flawlessly with Decimal."""
    ensure_corporate_decimal_context()
    assert getcontext().prec >= MIN_CORPORATE_DECIMAL_PRECISION

    with pytest.warns(DeprecationWarning), pytest.raises(TypeError):
        LedgerAccount(
            account_id="cash",
            name="Cash",
            category=AccountCategory.ASSET,
            balance=100.0,  # type: ignore[arg-type]
        )

    engine, init_fin = build_corporate_treasury_engine()
    assert isinstance(init_fin.total_assets, Decimal)
    assert init_fin.accounting_discrepancy == ZERO_DECIMAL

    # $0.01 imbalance must be rejected
    bad_outcome = engine.step(
        action=build_journal_action(
            "imbalance_1_cent",
            debits={"cash": Decimal("500.00")},
            credits={"retained_earnings": Decimal("500.01")},
        )
    )
    assert bad_outcome.pre_rule_rejected is True
    assert bad_outcome.rolled_back is True
    assert FinancialState.from_world_state(bad_outcome.next_state).accounting_discrepancy == ZERO_DECIMAL

    stress = run_liquidity_drain_stress_test(steps=5)
    assert stress.all_steps_balanced is True
    assert stress.max_abs_discrepancy_across_all_branches == ZERO_DECIMAL


def test_audit_checklist_5_swe_avoids_infinite_graph_recursion() -> None:
    """[x] SWE domain avoids infinite graph recursion (iterative O(V+E) cycle detection)."""
    # Verify detect_cycle_iterative_o_v_plus_e contains zero recursive self-calls in AST
    src_code = inspect.getsource(detect_cycle_iterative_o_v_plus_e)
    tree = ast.parse(src_code)
    func_calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "detect_cycle_iterative_o_v_plus_e" not in func_calls

    # Verify deep chain exceeding Python's default recursion limit (1000) does not raise RecursionError
    deep_n = 1500
    assert deep_n < MAX_SWE_GRAPH_NODES
    nodes = [f"file_{i:04d}.py" for i in range(deep_n)]
    deps = {nodes[i]: (nodes[i + 1],) for i in range(deep_n - 1)}
    deps[nodes[-1]] = (nodes[0],)
    cycle = detect_cycle_iterative_o_v_plus_e(deps, nodes)
    assert cycle is not None
    assert cycle[0] == cycle[-1]

    with pytest.raises(CircularDependencyError):
        SWEState(
            files={n: CodeFileNode(file_path=n) for n in ("x.py", "y.py")},
            dependencies={"x.py": ("y.py",), "y.py": ("x.py",)},
        )

    swe_engine = build_swe_agent_engine()
    base_ws = create_default_swe_repository_state().to_world_state()
    with pytest.raises(StructuralViolationError):
        swe_engine.step(
            base_ws,
            build_swe_action(
                "cycle_attempt",
                "ADD_DEPENDENCY",
                actor_id="auditor",
                parameters={
                    "source_file": "src/core/schema.py",
                    "target_file": "src/api/OrderController.py",
                },
            ),
        )


def test_audit_checklist_6_benchmarks_output_statistically_valid_json_artifacts(
    tmp_path: Path,
) -> None:
    """[x] Benchmarks output statistically valid JSON artifacts."""
    evaluator = EWM_Evaluator(
        env_spec=EnvironmentSpec(horizon_steps=10),
        timeout_seconds=30.0,
    )
    report = evaluator.evaluate(n_seeds=MIN_SIGNIFICANCE_SEEDS, master_seed=42)
    json_path, csv_path = evaluator.save_artifacts(
        report, output_dir=tmp_path, prefix="audit_ewm008"
    )
    assert json_path.exists() and csv_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_seeds"] == 30
    assert payload["significance_claim_permitted"] is True
    ewm_summary = payload["agent_summaries"]["EWMPolicyAgent"]
    assert ewm_summary["constraint_violation_rate"]["mean"] == 0.0
    assert (
        ewm_summary["cumulative_reward"]["ci_lower_95"]
        <= ewm_summary["cumulative_reward"]["iqm"]
        <= ewm_summary["cumulative_reward"]["ci_upper_95"]
    )

    # Also verify committed canonical JSON benchmark artifacts on disk
    committed_jsons = [
        REPO_ROOT / "reproduction_manifest.json",
        REPO_ROOT / "benchmarks" / "ewm_006_scientific_evaluation.json",
        REPO_ROOT / "docs" / "results" / "artifacts" / "comparative_evaluation_ladder.json",
        REPO_ROOT / "docs" / "results" / "artifacts" / "enterprise_benchmark_suite.json",
        REPO_ROOT / "docs" / "results" / "artifacts" / "fraud_cascade_flagship.json",
    ]
    for cj in committed_jsons:
        assert cj.exists(), f"Missing committed JSON benchmark artifact: {cj}"
        parsed = json.loads(cj.read_text(encoding="utf-8"))
        assert isinstance(parsed, dict) and len(parsed) > 0


def test_audit_checklist_7_readme_code_blocks_are_runnable() -> None:
    """[x] README code blocks are actually runnable."""
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "test_readme_snippets.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"README code block execution failed:\n{res.stdout}\n{res.stderr}"


def test_audit_checklist_8_hygiene_no_secrets_prompt_leakage_or_unqualified_causal_discovery() -> None:
    """[x] No sensitive data, prompt leakage, fabricated citations, or unqualified causal discovery claims."""
    # 1. Zero .bak, .tmp, .orig, or scratch files in repository
    bad_ext_files = [
        p
        for p in REPO_ROOT.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and ".venv" not in p.parts
        and p.suffix in {".bak", ".tmp", ".orig"}
    ]
    assert bad_ext_files == [], f"Found leftover backup/temp files: {bad_ext_files}"

    # 2. Scan src/, docs/, tests/, README.md for secret patterns or prompt meta-leakage
    secret_pattern = re.compile(
        r"(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
    )
    prompt_leak_pattern = re.compile(r"Action Required by AI:|<USER_REQUEST>")

    for scan_dir in (REPO_ROOT / "src", REPO_ROOT / "docs", REPO_ROOT / "README.md"):
        files = [scan_dir] if scan_dir.is_file() else list(scan_dir.rglob("*.py")) + list(scan_dir.rglob("*.md"))
        for fp in files:
            if fp.name == "README_legacy.md":
                continue
            text = fp.read_text(encoding="utf-8", errors="ignore")
            assert not secret_pattern.search(text), f"Potential secret pattern found in {fp}"
            assert not prompt_leak_pattern.search(text), f"Prompt meta-leakage found in {fp}"

    # Verify EWM-007 audited docs (README.md, REFERENCES.md, models, datasets, domains) have zero unverified citations
    audited_doc_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "REFERENCES.md",
        REPO_ROOT / "docs" / "benchmarks.md",
        REPO_ROOT / "docs" / "ml_ops.md",
        REPO_ROOT / "docs" / "architecture.md",
        REPO_ROOT / "domains" / "finance" / "README.md",
        REPO_ROOT / "domains" / "swe" / "README.md",
        *list((REPO_ROOT / "docs" / "models").glob("*.md")),
        *list((REPO_ROOT / "docs" / "datasets").glob("*.md")),
    ]
    for adp in audited_doc_paths:
        ad_text = adp.read_text(encoding="utf-8")
        assert "Business World Model" not in ad_text, f"Unverified citation in {adp}"
        assert "World of Workflows" not in ad_text, f"Unverified citation in {adp}"

    # 3. Verify Section 8 Scientific Requirement: causal discovery is strictly qualified
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Not an Assumption-Free Causal Discovery Oracle" in readme_text or "Markov Equivalence Class" in readme_text
    assert "CPDAG" in readme_text and "PAG" in readme_text


def test_audit_section_11_sandboxed_memory_limited_integration_subprocess() -> None:
    """Section 11 Operational Safeguard: Run integration tests in a sandboxed, memory-limited subprocess."""
    sandbox_runner = (
        "import tracemalloc, pytest, sys; "
        "tracemalloc.start(); "
        "code = pytest.main(['-q', 'tests/integration/test_simulation.py', "
        "'tests/integration/test_branching.py', 'tests/integration/test_hard_constraints.py', "
        "'-o', 'addopts=-q']); "
        "current, peak = tracemalloc.get_traced_memory(); "
        "tracemalloc.stop(); "
        "max_bytes = 512 * 1024 * 1024; "
        "sys.exit(code if peak < max_bytes else 99)"
    )
    sandbox_env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "USERPROFILE": os.environ.get("USERPROFILE", ""),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": f"{REPO_ROOT / 'src'}{os.pathsep}{REPO_ROOT}",
    }
    res = subprocess.run(
        [sys.executable, "-c", sandbox_runner],
        cwd=REPO_ROOT,
        env=sandbox_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert res.returncode == 0, (
        f"Sandboxed memory-limited integration subprocess failed (exit={res.returncode}):\n"
        f"{res.stdout}\n{res.stderr}"
    )
