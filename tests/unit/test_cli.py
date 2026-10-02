"""Unit tests for the EWM Engine CLI."""

from __future__ import annotations

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
