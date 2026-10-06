"""Architecture boundary tests verifying modular layering and dependency rules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_ROOT = REPO_ROOT / "src" / "ewm_engine"


def _extract_imports(
    file_path: Path, include_function_level: bool = False
) -> list[tuple[int, str]]:
    """Parse a Python source file and extract imported module names."""
    with open(file_path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    imports: list[tuple[int, str]] = []

    def _is_type_checking(node: ast.AST) -> bool:
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                return True
            if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                return True
        return False

    nodes_to_inspect = []
    if include_function_level:
        for node in ast.walk(tree):
            nodes_to_inspect.append(node)
    else:
        for node in tree.body:
            if _is_type_checking(node):
                continue
            nodes_to_inspect.append(node)

    for node in nodes_to_inspect:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.lineno, node.module))

    return imports


@pytest.mark.architecture
def test_core_does_not_import_higher_layers() -> None:
    """core MUST NOT import from simulation, evaluation, integrations, or adapters at top-level."""
    core_dir = SRC_ROOT / "core"
    forbidden = ("simulation", "evaluation", "integrations", "adapters")

    for py_file in core_dir.rglob("*.py"):
        imports = _extract_imports(py_file, include_function_level=False)
        for line_no, mod_name in imports:
            for f in forbidden:
                target = f"ewm_engine.{f}"
                if mod_name == target or mod_name.startswith(f"{target}."):
                    pytest.fail(
                        f"Architecture boundary violation in {py_file}:{line_no} - "
                        f"'core' must not import from '{f}' (found import '{mod_name}')"
                    )


@pytest.mark.architecture
def test_mid_layers_do_not_import_simulation() -> None:
    """dynamics, constraints, and provenance MUST NOT import from simulation."""
    mid_dirs = [SRC_ROOT / "dynamics", SRC_ROOT / "constraints", SRC_ROOT / "provenance"]
    forbidden = "ewm_engine.simulation"

    for layer_dir in mid_dirs:
        for py_file in layer_dir.rglob("*.py"):
            imports = _extract_imports(py_file, include_function_level=True)
            for line_no, mod_name in imports:
                if mod_name == forbidden or mod_name.startswith(f"{forbidden}."):
                    pytest.fail(
                        f"Architecture boundary violation in {py_file}:{line_no} - "
                        f"'{layer_dir.name}' must not import from 'simulation' (found '{mod_name}')"
                    )


@pytest.mark.architecture
def test_constraints_do_not_import_dynamics() -> None:
    """constraints MUST NOT import from dynamics."""
    constraints_dir = SRC_ROOT / "constraints"
    forbidden = "ewm_engine.dynamics"

    for py_file in constraints_dir.rglob("*.py"):
        imports = _extract_imports(py_file, include_function_level=True)
        for line_no, mod_name in imports:
            if mod_name == forbidden or mod_name.startswith(f"{forbidden}."):
                pytest.fail(
                    f"Architecture boundary violation in {py_file}:{line_no} - "
                    f"'constraints' must not import from 'dynamics' (found '{mod_name}')"
                )


@pytest.mark.architecture
def test_core_and_simulation_do_not_import_integrations() -> None:
    """integrations and adapters MUST NOT be imported by any core or simulation module."""
    dirs_to_check = [SRC_ROOT / "core", SRC_ROOT / "simulation"]
    forbidden_prefixes = ("ewm_engine.integrations", "ewm_engine.adapters")

    for layer_dir in dirs_to_check:
        for py_file in layer_dir.rglob("*.py"):
            imports = _extract_imports(py_file, include_function_level=True)
            for line_no, mod_name in imports:
                for prefix in forbidden_prefixes:
                    if mod_name == prefix or mod_name.startswith(f"{prefix}."):
                        pytest.fail(
                            f"Architecture boundary violation in {py_file}:{line_no} - "
                            f"'{layer_dir.name}' must not import '{prefix}'"
                        )


@pytest.mark.architecture
def test_spec_parser_does_not_perform_arbitrary_dynamic_imports() -> None:
    """core.spec MUST NOT perform dynamic import of arbitrary modules via importlib or __import__."""
    spec_file = SRC_ROOT / "core" / "spec.py"
    if not spec_file.exists():
        return

    with open(spec_file, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(spec_file))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in ("eval", "exec", "__import__"):
                pytest.fail(
                    f"Security/architecture violation in {spec_file}:{node.lineno} - "
                    f"Use of '{func.id}' is strictly forbidden in specification parser."
                )
            elif isinstance(func, ast.Attribute) and func.attr == "import_module":
                pytest.fail(
                    f"Security/architecture violation in {spec_file}:{node.lineno} - "
                    "Dynamic 'importlib.import_module' is forbidden in specification parser."
                )


@pytest.mark.architecture
def test_core_and_simulation_do_not_import_opentelemetry() -> None:
    """Core, simulation, constraints, and dynamics MUST NOT import opentelemetry."""
    checked_dirs = [
        SRC_ROOT / "core",
        SRC_ROOT / "simulation",
        SRC_ROOT / "constraints",
        SRC_ROOT / "dynamics",
        SRC_ROOT / "hooks",
        SRC_ROOT / "provenance",
    ]
    for d in checked_dirs:
        for py_file in d.rglob("*.py"):
            imports = _extract_imports(py_file, include_function_level=True)
            for line_no, mod_name in imports:
                if mod_name == "opentelemetry" or mod_name.startswith("opentelemetry."):
                    pytest.fail(
                        f"Architecture boundary violation in {py_file}:{line_no} - "
                        f"'{d.name}' must not depend on OpenTelemetry (found import '{mod_name}')"
                    )


@pytest.mark.architecture
def test_otel_adapter_does_not_import_opentelemetry_sdk() -> None:
    """integrations.otel MUST NOT import opentelemetry.sdk in library code (API-only contract)."""
    otel_file = SRC_ROOT / "integrations" / "otel.py"
    if not otel_file.exists():
        pytest.fail(f"Expected adapter file {otel_file} does not exist.")

    imports = _extract_imports(otel_file, include_function_level=True)
    for line_no, mod_name in imports:
        if mod_name == "opentelemetry.sdk" or mod_name.startswith("opentelemetry.sdk."):
            pytest.fail(
                f"OpenTelemetry API-only constraint violated in {otel_file}:{line_no} - "
                f"Library code must depend only on 'opentelemetry-api', never SDK (found '{mod_name}')"
            )


@pytest.mark.architecture
def test_core_and_simulation_do_not_import_solvers_or_planners() -> None:
    """Core, simulation, constraints, dynamics, provenance, and evaluation MUST NOT import z3, ortools, or scipy."""
    checked_dirs = [
        SRC_ROOT / "core",
        SRC_ROOT / "simulation",
        SRC_ROOT / "constraints",
        SRC_ROOT / "dynamics",
        SRC_ROOT / "hooks",
        SRC_ROOT / "provenance",
        SRC_ROOT / "evaluation",
    ]
    forbidden_prefixes = ("z3", "ortools", "scipy")
    for d in checked_dirs:
        for py_file in d.rglob("*.py"):
            imports = _extract_imports(py_file, include_function_level=True)
            for line_no, mod_name in imports:
                for prefix in forbidden_prefixes:
                    if mod_name == prefix or mod_name.startswith(f"{prefix}."):
                        pytest.fail(
                            f"Architecture boundary violation in {py_file}:{line_no} - "
                            f"'{d.name}' must not depend on '{prefix}' (found import '{mod_name}')"
                        )


@pytest.mark.architecture
def test_no_torch_outside_experimental_ml() -> None:
    """Core, simulation, constraints, dynamics, provenance, evaluation, and integrations MUST NOT import torch."""
    checked_dirs = [
        SRC_ROOT / "core",
        SRC_ROOT / "simulation",
        SRC_ROOT / "constraints",
        SRC_ROOT / "dynamics",
        SRC_ROOT / "hooks",
        SRC_ROOT / "provenance",
        SRC_ROOT / "evaluation",
        SRC_ROOT / "integrations",
    ]
    for d in checked_dirs:
        for py_file in d.rglob("*.py"):
            imports = _extract_imports(py_file, include_function_level=True)
            for line_no, mod_name in imports:
                if mod_name == "torch" or mod_name.startswith("torch."):
                    pytest.fail(
                        f"Architecture boundary violation in {py_file}:{line_no} - "
                        f"'{d.name}' must not depend on PyTorch (found import '{mod_name}')"
                    )

    # dynamics_eval.py must also be Torch-free
    eval_file = SRC_ROOT / "experimental" / "dynamics_eval.py"
    if eval_file.exists():
        eval_imports = _extract_imports(eval_file, include_function_level=True)
        for line_no, mod_name in eval_imports:
            if mod_name == "torch" or mod_name.startswith("torch."):
                pytest.fail(
                    f"Architecture boundary violation in {eval_file}:{line_no} - "
                    f"'dynamics_eval.py' must be Torch-free (found import '{mod_name}')"
                )

    # dynamics_torch.py must not import Torch at top-level
    torch_file = SRC_ROOT / "experimental" / "dynamics_torch.py"
    if torch_file.exists():
        top_imports = _extract_imports(torch_file, include_function_level=False)
        for line_no, mod_name in top_imports:
            if mod_name == "torch" or mod_name.startswith("torch."):
                pytest.fail(
                    f"Architecture boundary violation in {torch_file}:{line_no} - "
                    f"'dynamics_torch.py' must lazily load PyTorch, never at top-level (found '{mod_name}')"
                )


@pytest.mark.architecture
def test_top_level_import_does_not_load_torch() -> None:
    """Importing ewm_engine or ewm_engine.experimental must NOT load torch into sys.modules."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-c",
        "import sys; import ewm_engine; import ewm_engine.experimental; assert 'torch' not in sys.modules, 'torch leaked into sys.modules'",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Importing ewm_engine or experimental leaked torch: {proc.stderr}"
