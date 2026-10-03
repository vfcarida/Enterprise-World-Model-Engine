"""Unit tests for the EWM Engine CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ewm_engine import __version__
from ewm_engine.cli.main import main


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Test `ewm --version`."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert f"ewm-engine {__version__}" in captured.out


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Test `ewm --help`."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "Enterprise World Model Engine" in captured.out


def test_cli_example_minimal(capsys: pytest.CaptureFixture[str]) -> None:
    """Test `ewm example minimal`."""
    exit_code = main(["example", "minimal"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Scenario Comparison" in captured.out


def test_cli_validate_valid_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test `ewm validate` with a valid YAML specification."""
    spec_content = """
world:
  name: "CLIWarehouse"
entities:
  - id: "wh_1"
    type: "warehouse"
resources:
  - id: "inv_1"
    entity_id: "wh_1"
    current: 50.0
    min_value: 0.0
    max_value: 100.0
constraints:
  - type: "capacity"
    parameters:
      resource_id: "inv_1"
"""
    spec_file = tmp_path / "valid_spec.yaml"
    spec_file.write_text(spec_content, encoding="utf-8")

    exit_code = main(["validate", str(spec_file)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[VALID] World specification is valid" in captured.out
    assert "CLIWarehouse" in captured.out
    assert "Entities:      1" in captured.out


def test_cli_validate_invalid_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test `ewm validate` with an invalid specification (unknown constraint)."""
    invalid_content = """
world:
  name: "BrokenWorld"
constraints:
  - type: "unknown_forbidden_constraint"
"""
    spec_file = tmp_path / "broken_spec.yaml"
    spec_file.write_text(invalid_content, encoding="utf-8")

    exit_code = main(["validate", str(spec_file)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[INVALID] Specification error" in captured.err


def test_cli_validate_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    """Test `ewm validate` with non-existent file."""
    exit_code = main(["validate", "non_existent_file.yaml"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Specification file not found" in captured.err


def test_cli_run_spec(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Test `ewm run` simulation directly from declarative specification."""
    spec_content = """
world:
  name: "CLIRunWorld"
entities:
  - id: "node_a"
    type: "depot"
resources:
  - id: "res_a"
    entity_id: "node_a"
    current: 100.0
"""
    spec_file = tmp_path / "run_spec.yaml"
    spec_file.write_text(spec_content, encoding="utf-8")
    out_file = tmp_path / "out_results.json"
    html_file = tmp_path / "out_trace.html"

    exit_code = main(
        [
            "run",
            str(spec_file),
            "--horizon",
            "3",
            "--samples",
            "2",
            "--seed",
            "99",
            "--out",
            str(out_file),
            "--html-trace",
            str(html_file),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Simulation Run Complete" in captured.out
    assert "Trajectories Total:     2" in captured.out
    assert out_file.exists()
    assert html_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "trajectories" in data
    assert len(data["trajectories"]) == 2

    html_text = html_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_text
    assert "CLIRunWorld Trace" in html_text


def test_cli_schema_list(capsys: pytest.CaptureFixture[str]) -> None:
    """Test `ewm schema list`."""
    exit_code = main(["schema", "list"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Available JSON Schemas" in captured.out
    assert "world-state" in captured.out


def test_cli_schema_show(capsys: pytest.CaptureFixture[str]) -> None:
    """Test `ewm schema show world-state`."""
    exit_code = main(["schema", "show", "world-state"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"title": "WorldState"' in captured.out
