"""Contract tests asserting zero forced external dependencies (torch, solvers, graphs) in core."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "ewm_engine"

OPTIONAL_HEAVY_DEPS = ("torch", "torchvision", "z3", "networkx", "scipy")


@pytest.mark.contract
def test_import_ewm_engine_does_not_pull_heavy_dependencies() -> None:
    """AC-019: Assert that importing ewm_engine does not import torch, z3, or networkx into sys.modules."""
    import subprocess
    import sys

    code = (
        "import sys, ewm_engine; "
        f"deps = {OPTIONAL_HEAVY_DEPS!r}; "
        "imported = [d for d in deps if d in sys.modules]; "
        "assert not imported, f'Importing ewm_engine pulled heavy optional dependencies {imported}'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.contract
def test_no_top_level_optional_imports_in_core_modules() -> None:
    """AC-019: Statically assert that no core source module contains top-level imports of optional dependencies."""
    core_packages = [
        "core",
        "constraints",
        "dynamics",
        "provenance",
        "simulation",
        "evaluation",
        "serialization",
        "durability",
        "cards",
        "verification",
        "experimentation",
        "cosim",
        "multiagent",
        "reporting",
        "service",
        "trackers",
    ]

    for pkg in core_packages:
        pkg_dir = SRC_ROOT / pkg
        for py_file in pkg_dir.rglob("*.py"):
            with open(py_file, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))

            for node in tree.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for heavy in OPTIONAL_HEAVY_DEPS:
                            if alias.name == heavy or alias.name.startswith(f"{heavy}."):
                                pytest.fail(
                                    f"Forbidden top-level import '{alias.name}' in core module {py_file}:{node.lineno}"
                                )
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    for heavy in OPTIONAL_HEAVY_DEPS:
                        if mod == heavy or mod.startswith(f"{heavy}."):
                            pytest.fail(
                                f"Forbidden top-level from-import '{mod}' in core module {py_file}:{node.lineno}"
                            )


@pytest.mark.contract
def test_world_spec_and_factory_require_no_heavy_dependencies() -> None:
    """AC-019: Assert that parsing WorldSpec and instantiating WorldFactory requires no ML/solver/LLM."""
    import subprocess
    import sys

    code = (
        "import sys; "
        "from ewm_engine.core.spec import WorldFactory; "
        'spec_yaml = \'\'\'\\nworld:\\n  name: "DepCheckWorld"\\nentities:\\n  - id: "e1"\\n    type: "node"\\nresources:\\n  - id: "r1"\\n    current: 10.0\\nconstraints:\\n  - type: "capacity"\\n    parameters:\\n      resource_id: "r1"\\ndynamics:\\n  type: "transfer"\\n  parameters:\\n    action_type: "transfer_resource"\\n\'\'\'; '
        "factory = WorldFactory(); "
        "world = factory.create_from_yaml(spec_yaml); "
        "assert world is not None; "
        "assert world.initial_state.get_resource('r1').current == 10.0; "
        f"deps = {OPTIONAL_HEAVY_DEPS!r}; "
        "imported = [d for d in deps if d in sys.modules]; "
        "assert not imported, f'WorldFactory pulled heavy optional dependencies {imported}'"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
