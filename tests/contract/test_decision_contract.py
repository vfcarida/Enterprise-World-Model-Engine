"""Contract tests for decision-focused scenario discovery and multi-robustness analysis."""

from __future__ import annotations

import json
from pathlib import Path

from ewm_engine.cli.main import main
from ewm_engine.core.actions import Action
from ewm_engine.core.spec import WorldFactory, WorldSpec
from ewm_engine.decision.engine import DecisionEngine
from ewm_engine.decision.report import DecisionReport
from ewm_engine.experimentation.params import ParameterDef, ParameterSpace


def _get_warehouse_spec_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "examples" / "minimal_warehouse" / "spec.yaml"


def test_decision_engine_fingerprint_equivalence_contract() -> None:
    """Invariant 5: Identical decision analysis inputs must yield bit-identical fingerprints."""
    spec = WorldSpec.from_yaml(_get_warehouse_spec_path())
    world = WorldFactory().create_world(spec)

    space = ParameterSpace(
        name="ContractUncertainty",
        parameters=(
            ParameterDef(
                name="warehouse_a_initial",
                type="continuous",
                bounds=(50.0, 150.0),
                default=100.0,
                target="resource",
                target_id="warehouse_a",
            ),
            ParameterDef(
                name="warehouse_b_initial",
                type="continuous",
                bounds=(10.0, 80.0),
                default=20.0,
                target="resource",
                target_id="warehouse_b",
            ),
        ),
    )

    candidates = {
        "status_quo": None,
        "policy_transfer": Action(
            id="transfer_act",
            type="transfer_resource",
            parameters={
                "source_resource": "warehouse_a",
                "target_resource": "warehouse_b",
                "quantity": 25.0,
            },
        ),
    }

    engine = DecisionEngine()
    rep1 = engine.decide(
        world=world,
        uncertainty_space=space,
        candidates=candidates,
        objective_metric="unserved_demand",
        satisficing_threshold=10.0,
        n_samples=15,
        seed=12345,
    )

    rep2 = engine.decide(
        world=world,
        uncertainty_space=space,
        candidates=candidates,
        objective_metric="unserved_demand",
        satisficing_threshold=10.0,
        n_samples=15,
        seed=12345,
    )

    assert isinstance(rep1, DecisionReport)
    assert rep1.fingerprint == rep2.fingerprint
    assert len(rep1.fingerprint) == 64
    assert rep1.robustness.fingerprint == rep2.robustness.fingerprint


def test_anti_overclaiming_and_no_forbidden_phrases() -> None:
    """EWM must never claim 'guarantees optimal' and must refuse single ranking when metrics disagree."""
    spec = WorldSpec.from_yaml(_get_warehouse_spec_path())
    world = WorldFactory().create_world(spec)

    space = ParameterSpace(
        name="ContractUncertainty",
        parameters=(
            ParameterDef(
                name="warehouse_a_initial",
                type="continuous",
                bounds=(50.0, 150.0),
                default=100.0,
                target="resource",
                target_id="warehouse_a",
            ),
            ParameterDef(
                name="warehouse_b_initial",
                type="continuous",
                bounds=(10.0, 80.0),
                default=20.0,
                target="resource",
                target_id="warehouse_b",
            ),
        ),
    )

    candidates = {
        "status_quo": None,
        "policy_transfer": Action(
            id="transfer_act",
            type="transfer_resource",
            parameters={
                "source_resource": "warehouse_a",
                "target_resource": "warehouse_b",
                "quantity": 25.0,
            },
        ),
    }

    engine = DecisionEngine()
    rep = engine.decide(
        world=world,
        uncertainty_space=space,
        candidates=candidates,
        objective_metric="unserved_demand",
        satisficing_threshold=0.0,
        n_samples=15,
        seed=42,
    )

    summary = rep.summary_table()
    markdown = rep.to_markdown()

    # The forbidden phrase 'guarantees optimal' must never appear anywhere
    assert "guarantees optimal" not in summary.lower()
    assert "guarantees optimal" not in markdown.lower()

    # If metrics disagree, dominant_action must be None
    if rep.robustness.refusal_to_rank:
        assert rep.robustness.dominant_action is None
        assert "refusal to rank" in summary.lower()
        assert "refusal to rank" in markdown.lower()


def test_cli_ewm_decide_execution_and_file_export(tmp_path: Path) -> None:
    """CLI subcommand `ewm decide` must execute cleanly and write valid JSON and Markdown reports."""
    spec_path = _get_warehouse_spec_path()
    out_json = tmp_path / "decision_report.json"
    out_md = tmp_path / "decision_report.md"

    exit_code = main(
        [
            "decide",
            str(spec_path),
            "--samples",
            "10",
            "--seed",
            "77",
            "--threshold",
            "5.0",
            "--efe",
            "--out",
            str(out_json),
            "--markdown-out",
            str(out_md),
        ]
    )

    assert exit_code == 0
    assert out_json.exists()
    assert out_md.exists()

    # Verify JSON content
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert "robustness" in data
    assert "vulnerabilities" in data
    assert "efe_scores" in data
    assert "fingerprint" in data

    # Verify Markdown content
    md_content = out_md.read_text(encoding="utf-8")
    assert "# Decision Analysis for" in md_content
    assert "Robustness Comparison" in md_content
