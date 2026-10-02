"""Contract test (AC-017): py.typed typing marker file exists and is configured for distribution."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "ewm_engine"


@pytest.mark.contract
def test_py_typed_marker_exists_in_source() -> None:
    """AC-017: Assert that py.typed exists in the source package root."""
    py_typed = SRC_ROOT / "py.typed"
    assert py_typed.exists(), f"PEP 561 marker missing: {py_typed}"
    assert py_typed.is_file(), f"{py_typed} must be a file"


@pytest.mark.contract
def test_py_typed_included_in_setuptools_configuration() -> None:
    """AC-017: Assert that pyproject.toml explicitly packages py.typed."""
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "py.typed" in pyproject_text, (
        "pyproject.toml must configure package-data to include 'py.typed' for PEP 561 compliance."
    )


@pytest.mark.contract
def test_wheel_contains_py_typed(tmp_path: Path) -> None:
    """AC-017: Build wheel in isolated temp directory and verify py.typed is present inside archive."""
    import subprocess

    # Build wheel into tmp_path using uv build
    cmd = [
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        str(tmp_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Wheel build failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"Expected 1 wheel, found: {wheels}"

    wheel_path = wheels[0]
    with zipfile.ZipFile(wheel_path) as z:
        names = z.namelist()
        py_typed_entry = "ewm_engine/py.typed"
        assert py_typed_entry in names, (
            f"Built wheel '{wheel_path.name}' does not contain '{py_typed_entry}'! Archive contents: {names}"
        )


@pytest.mark.contract
def test_wheel_install_and_quickstart_smoke(tmp_path: Path) -> None:
    """AC-016 / AC-021: Verify built wheel installs in a clean environment and executes the Quickstart."""
    import subprocess
    import sys

    dist_dir = tmp_path / "dist"
    venv_dir = tmp_path / "venv"

    # 1. Build wheel
    build_cmd = ["uv", "build", "--wheel", "--out-dir", str(dist_dir)]
    res = subprocess.run(build_cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert res.returncode == 0, f"Build failed:\n{res.stdout}\n{res.stderr}"

    wheel_files = list(dist_dir.glob("*.whl"))
    assert len(wheel_files) == 1
    wheel_path = wheel_files[0]

    # 2. Create isolated virtual environment
    venv_cmd = ["uv", "venv", str(venv_dir)]
    res_venv = subprocess.run(venv_cmd, capture_output=True, text=True)
    assert res_venv.returncode == 0, f"venv creation failed:\n{res_venv.stdout}\n{res_venv.stderr}"

    # Determine Python executable inside virtual environment
    if sys.platform == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"
    assert venv_python.exists(), f"Virtualenv Python not found: {venv_python}"

    # 3. Install wheel into clean virtual environment (without any dev dependencies)
    install_cmd = ["uv", "pip", "install", "--python", str(venv_python), str(wheel_path)]
    res_install = subprocess.run(install_cmd, capture_output=True, text=True)
    assert res_install.returncode == 0, (
        f"Wheel install failed:\n{res_install.stdout}\n{res_install.stderr}"
    )

    # 4. Execute quickstart smoke test in clean environment
    smoke_script = (
        "import ewm_engine\n"
        "from ewm_engine import World, WorldState, Resource, Scenario, SimulationEngine\n"
        "world = World(state=WorldState(resources={'stock': Resource(id='stock', current=10.0, max_value=100.0)}))\n"
        "scenario = Scenario(horizon=2, samples=1, seed=42)\n"
        "result = SimulationEngine().run(world, scenario)\n"
        "assert len(result.trajectories) == 1\n"
        "assert result.run_metrics.completed_rollout_count == 1\n"
        "print('SMOKE_TEST_OK')\n"
    )

    res_smoke = subprocess.run(
        [str(venv_python), "-c", smoke_script],
        capture_output=True,
        text=True,
    )
    assert res_smoke.returncode == 0, (
        f"Quickstart smoke execution failed:\n{res_smoke.stdout}\n{res_smoke.stderr}"
    )
    assert "SMOKE_TEST_OK" in res_smoke.stdout
