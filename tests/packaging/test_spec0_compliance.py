"""SPEC 0 compliance test enforcing modern Python and NumPy version floors.

SPEC 0 (Scientific Python Ecosystem Coordination) guidelines:
- Python versions: drop support for versions released >36 months ago.
- NumPy versions: drop support for versions released >24 months ago.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


@pytest.mark.packaging
def test_spec0_requires_python_floor() -> None:
    """Ensure requires-python conforms to SPEC 0 3-year support window (>=3.11)."""
    pyproject_path = Path("pyproject.toml")
    assert pyproject_path.exists(), "pyproject.toml not found"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    requires_python = project.get("requires-python", "")

    assert requires_python, "requires-python must be explicitly declared in pyproject.toml"
    assert ">=3.11" in requires_python, (
        f"SPEC 0 violation: requires-python is '{requires_python}', but SPEC 0 mandates dropping "
        f"Python <=3.10 (minimum supported Python must be >=3.11)."
    )


@pytest.mark.packaging
def test_spec0_numpy_dependency_floor() -> None:
    """Ensure NumPy dependency floor conforms to SPEC 0 2-year window (>=1.26.0)."""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    dependencies = data.get("project", {}).get("dependencies", [])
    numpy_dep = next((dep for dep in dependencies if dep.startswith("numpy")), None)

    assert numpy_dep is not None, "numpy must be declared in project dependencies"
    assert ">=1.26.0" in numpy_dep or ">=2." in numpy_dep, (
        f"SPEC 0 violation: numpy dependency is '{numpy_dep}', but SPEC 0 mandates minimum NumPy >=1.26.0."
    )


@pytest.mark.packaging
def test_all_core_and_optional_dependencies_have_explicit_lower_bounds() -> None:
    """Ensure every core and optional dependency declares an explicit '>=' lower bound."""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    project = data.get("project", {})
    core_deps = project.get("dependencies", [])
    for dep in core_deps:
        assert ">=" in dep, f"Core dependency '{dep}' is missing an explicit '>=' lower bound."

    optional_deps = project.get("optional-dependencies", {})
    assert optional_deps, "optional-dependencies must not be empty"
    for extra_name, deps in optional_deps.items():
        for dep in deps:
            assert ">=" in dep, (
                f"Optional dependency '{dep}' in extra '[{extra_name}]' is missing an explicit '>=' lower bound."
            )


@pytest.mark.packaging
def test_all_extra_semantics_and_documented_exclusions() -> None:
    """Verify [all] extra includes CPU extras and intentionally excludes cuda/erroranalysis."""
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    optional_deps = data.get("project", {}).get("optional-dependencies", {})
    all_deps = optional_deps.get("all", [])
    assert all_deps, "[all] extra must be defined in pyproject.toml"

    all_joined = " ".join(all_deps)
    # Must exclude hardware-specific cupy (cuda) and Python-<3.12-restricted erroranalysis
    assert "cupy" not in all_joined, (
        "[all] extra must exclude hardware-specific 'cupy' (cuda extra)"
    )
    assert "erroranalysis" not in all_joined, (
        "[all] extra must exclude 'erroranalysis' due to Python <3.12 and scipy conflict constraints"
    )

    # Verify uv conflict table documents erroranalysis vs all
    conflicts = data.get("tool", {}).get("uv", {}).get("conflicts", [])
    assert any(
        {item.get("extra") for item in group} == {"erroranalysis", "all"} for group in conflicts
    ), "pyproject.toml [tool.uv].conflicts must declare conflict between 'erroranalysis' and 'all'"


@pytest.mark.packaging
def test_release_truth_and_mutation_baseline_gates() -> None:
    """Verify release truth script and selective mutation baseline verification succeed."""
    import subprocess
    import sys

    truth_proc = subprocess.run(
        [sys.executable, "scripts/check_release_truth.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert truth_proc.returncode == 0, (
        f"check_release_truth.py failed:\nSTDOUT:\n{truth_proc.stdout}\nSTDERR:\n{truth_proc.stderr}"
    )

    mut_proc = subprocess.run(
        [sys.executable, "scripts/run_mutation_gate.py", "--verify-baseline"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert mut_proc.returncode == 0, (
        f"run_mutation_gate.py --verify-baseline failed:\nSTDOUT:\n{mut_proc.stdout}\nSTDERR:\n{mut_proc.stderr}"
    )
