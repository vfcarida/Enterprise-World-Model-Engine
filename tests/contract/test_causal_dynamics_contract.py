"""Contract tests for Causal, Dynamics, and Invariant induction schemas and CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ewm_engine.cli.main import main
from ewm_engine.core.spec import WorldSpec
from ewm_engine.induction.causal import induce_causal
from ewm_engine.induction.report import InductionReport


@pytest.mark.contract
def test_induced_causal_world_spec_contract_compliance() -> None:
    """Verify that WorldSpec.from_causal produces schema-compliant WorldSpec instances."""
    data = [
        {"x": 1.0, "y": 2.0, "z": 3.0},
        {"x": 2.0, "y": 3.5, "z": 5.0},
        {"x": 3.0, "y": 5.0, "z": 7.0},
        {"x": 4.0, "y": 6.5, "z": 9.0},
    ]

    spec = WorldSpec.from_causal(data, name="ContractCausalWorld", seed=42)
    assert isinstance(spec, WorldSpec)

    dumped = spec.model_dump(mode="json")
    reloaded = WorldSpec.model_validate(dumped)
    assert len(reloaded.entities) == len(spec.entities)
    assert len(reloaded.relationships) == len(spec.relationships)


@pytest.mark.contract
def test_induced_dynamics_and_invariants_contract_compliance() -> None:
    """Verify that WorldSpec.from_dynamics and from_invariants validate against Pydantic schema."""
    trajectories = [
        {"stock_a": 100.0, "stock_b": 200.0},
        {"stock_a": 90.0, "stock_b": 210.0},
        {"stock_a": 110.0, "stock_b": 190.0},
        {"stock_a": 95.0, "stock_b": 205.0},
    ]

    spec_dyn = WorldSpec.from_dynamics(trajectories, name="ContractDynWorld", seed=42)
    assert isinstance(spec_dyn, WorldSpec)
    reloaded_dyn = WorldSpec.model_validate(spec_dyn.model_dump(mode="json"))
    assert len(reloaded_dyn.resources) == len(spec_dyn.resources)

    spec_inv = WorldSpec.from_invariants(trajectories, name="ContractInvWorld", seed=42)
    assert isinstance(spec_inv, WorldSpec)
    reloaded_inv = WorldSpec.model_validate(spec_inv.model_dump(mode="json"))
    assert len(reloaded_inv.constraints) == len(spec_inv.constraints)


@pytest.mark.contract
def test_causal_induction_report_contract_and_roundtrip() -> None:
    """Verify InductionReport serializes to valid JSON, computes fingerprint, and round-trips."""
    data = [
        {"a": 10.0, "b": 20.0},
        {"a": 15.0, "b": 25.0},
        {"a": 20.0, "b": 30.0},
    ]
    _, report = induce_causal(data, seed=123)

    assert isinstance(report, InductionReport)
    assert report.sha256_fingerprint != ""
    assert report.causal_equivalence_class is not None

    rep_json = report.to_json()
    rep_data = json.loads(rep_json)
    reloaded_report = InductionReport.model_validate(rep_data)

    assert reloaded_report.sha256_fingerprint == report.sha256_fingerprint
    assert reloaded_report.causal_equivalence_class is not None
    assert reloaded_report.causal_equivalence_class.asserted_edges_count == (
        report.causal_equivalence_class.asserted_edges_count
    )


@pytest.mark.contract
def test_cli_induce_causal_subcommand(tmp_path: Path) -> None:
    """Verify `ewm induce causal` runs via CLI and writes valid YAML and JSON report."""
    csv_file = tmp_path / "obs.csv"
    csv_file.write_text(
        "x,y,z\n1.0,2.0,3.0\n2.0,4.0,5.0\n3.0,6.0,7.0\n4.0,8.0,9.0\n", encoding="utf-8"
    )
    out_yaml = tmp_path / "causal_world.yaml"
    out_report = tmp_path / "causal_report.json"

    exit_code = main(
        [
            "induce",
            "causal",
            "--data",
            str(csv_file),
            "--out",
            str(out_yaml),
            "--report",
            str(out_report),
        ]
    )

    assert exit_code == 0
    assert out_yaml.exists()
    assert out_report.exists()
    rep = json.loads(out_report.read_text(encoding="utf-8"))
    assert "causal_equivalence_class" in rep


@pytest.mark.contract
def test_cli_induce_dynamics_and_invariants_subcommands(tmp_path: Path) -> None:
    """Verify `ewm induce dynamics` and `ewm induce invariants` run via CLI."""
    csv_file = tmp_path / "traj.csv"
    csv_file.write_text("stock_a,stock_b\n100.0,200.0\n90.0,210.0\n110.0,190.0\n", encoding="utf-8")

    out_dyn_yaml = tmp_path / "dyn_world.yaml"
    out_dyn_rep = tmp_path / "dyn_report.json"
    exit_dyn = main(
        [
            "induce",
            "dynamics",
            "--trajectories",
            str(csv_file),
            "--out",
            str(out_dyn_yaml),
            "--report",
            str(out_dyn_rep),
        ]
    )
    assert exit_dyn == 0
    assert out_dyn_yaml.exists()
    assert out_dyn_rep.exists()

    out_inv_yaml = tmp_path / "inv_world.yaml"
    out_inv_rep = tmp_path / "inv_report.json"
    exit_inv = main(
        [
            "induce",
            "invariants",
            "--trajectories",
            str(csv_file),
            "--out",
            str(out_inv_yaml),
            "--report",
            str(out_inv_rep),
        ]
    )
    assert exit_inv == 0
    assert out_inv_yaml.exists()
    assert out_inv_rep.exists()
