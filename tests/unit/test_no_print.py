"""Verify that no module in src/ewm_engine uses the forbidden print() statement."""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "ewm_engine"


def test_no_print_in_library_code() -> None:
    """ast-scan src/ewm_engine/** for print( -> none."""
    violations: list[str] = []

    for py_file in SRC_ROOT.rglob("*.py"):
        with open(py_file, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == "print":
                    rel_path = py_file.relative_to(SRC_ROOT.parent.parent)
                    violations.append(f"{rel_path}:{node.lineno}")

    assert not violations, (
        f"Found forbidden print() calls in library code ({len(violations)} found):\n"
        + "\n".join(violations)
    )
