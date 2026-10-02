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
    import sys

    # Importing root package
    import ewm_engine  # noqa: F401

    for heavy_dep in OPTIONAL_HEAVY_DEPS:
        assert heavy_dep not in sys.modules, (
            f"Importing ewm_engine pulled heavy optional dependency '{heavy_dep}' into sys.modules!"
        )


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
    import sys

    from ewm_engine.core.spec import WorldFactory

    spec_yaml = """
    world:
      name: "DepCheckWorld"
    entities:
      - id: "e1"
        type: "node"
    resources:
      - id: "r1"
        current: 10.0
    constraints:
      - type: "capacity"
        parameters:
          resource_id: "r1"
    dynamics:
      type: "transfer"
      parameters:
        action_type: "transfer_resource"
    """
    factory = WorldFactory()
    world = factory.create_from_yaml(spec_yaml)
    assert world is not None
    assert world.initial_state.get_resource("r1").current == 10.0

    for heavy_dep in OPTIONAL_HEAVY_DEPS:
        assert heavy_dep not in sys.modules
