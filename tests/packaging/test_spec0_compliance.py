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
