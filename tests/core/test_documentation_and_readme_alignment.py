"""Verification test suite for EWM-007: Documentation & README Alignment.

Verifies:
1. Original README backup at `docs/legacy/README_legacy.md`.
2. Required `README.md` structure (Problem Statement -> Verified Capabilities -> Installation
   -> Tested Quickstart -> Domain Examples -> Limitations & Mathematical Boundaries of Causal Models)
   and removal of unverified citations.
3. 100% clean execution of all Python code blocks in `README.md`.
4. 100% clean execution of all code cells in `docs/tutorials/1_quickstart.ipynb`.
5. Explicit Supported vs. Experimental delineation across `docs/models/*.md` and `docs/datasets/*.md`.
6. Zero broken local file or anchor links in `README.md`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README_PATH = REPO_ROOT / "README.md"
LEGACY_README_PATH = REPO_ROOT / "docs" / "legacy" / "README_legacy.md"
NOTEBOOK_PATH = REPO_ROOT / "docs" / "tutorials" / "1_quickstart.ipynb"


def test_legacy_readme_backup_exists() -> None:
    """Original README.md must be backed up at docs/legacy/README_legacy.md."""
    assert LEGACY_README_PATH.exists(), "Missing docs/legacy/README_legacy.md backup"
    legacy_text = LEGACY_README_PATH.read_text(encoding="utf-8")
    assert "## 1. What EWM Engine Is" in legacy_text
    assert len(legacy_text) > 5000


def test_readme_structure_and_factual_hygiene() -> None:
    """README.md must follow the EWM-007 structure, delineate Supported vs Experimental, and document causal limits."""
    readme_text = README_PATH.read_text(encoding="utf-8")

    required_sections = [
        "## 1. Problem Statement",
        "## 2. Verified Capabilities (Supported vs. Experimental)",
        "## 3. Installation",
        "## 4. Tested Quickstart",
        "## 5. Domain Examples (Finance & SWE)",
        "## 6. Limitations & Mathematical Boundaries of Causal Models",
        "## 7. Reproducibility & Documentation Map",
        "## 8. Security, Governance & Citation",
    ]
    positions = []
    for header in required_sections:
        assert header in readme_text, f"Missing required section header in README.md: {header}"
        positions.append(readme_text.index(header))
    assert positions == sorted(
        positions
    ), "README.md sections are not in the required chronological order"

    # Unverified 2026 citations must be removed from README.md
    assert "Business World Model" not in readme_text
    assert "World of Workflows" not in readme_text

    # Mathematical boundaries of causal models must be explicitly documented
    for causal_term in (
        "Unobserved Confounding",
        "non-identifiable",
        "Causal Markov Condition",
        "Faithfulness",
        "Markov Equivalence Class",
        "exchangeability",
        "EXTERNALLY_REPRODUCED = 0",
    ):
        assert (
            causal_term in readme_text
        ), f"Missing causal limitation term in README.md: {causal_term}"


def test_all_readme_python_blocks_execute_cleanly() -> None:
    """Every ```python block in README.md must execute without error."""
    readme_text = README_PATH.read_text(encoding="utf-8")
    pattern = re.compile(r"```python\r?\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(readme_text)
    assert len(blocks) >= 4, f"Expected at least 4 Python blocks in README.md, found {len(blocks)}"

    for idx, code in enumerate(blocks, start=1):
        ns: dict[str, object] = {"__name__": f"__readme_block_{idx}__"}
        exec(compile(code, f"<README.md block #{idx}>", "exec"), ns)


def test_quickstart_notebook_executes_end_to_end() -> None:
    """docs/tutorials/1_quickstart.ipynb must be valid nbformat v4 JSON and all code cells must execute."""
    assert NOTEBOOK_PATH.exists(), "Missing docs/tutorials/1_quickstart.ipynb"
    nb_data = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nb_data.get("nbformat") == 4
    cells = nb_data.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(code_cells) >= 4, "Expected at least 4 code cells in 1_quickstart.ipynb"

    shared_ns: dict[str, object] = {"__name__": "__quickstart_notebook__"}
    for idx, cell in enumerate(code_cells, start=1):
        source_lines = cell.get("source", [])
        code = "".join(source_lines) if isinstance(source_lines, list) else str(source_lines)
        exec(compile(code, f"<1_quickstart.ipynb cell #{idx}>", "exec"), shared_ns)


def test_model_and_dataset_cards_clarify_experimental_scope() -> None:
    """docs/models/*.md and docs/datasets/*.md must carry explicit Experimental maturity banners."""
    model_files = [
        REPO_ROOT / "docs" / "models" / "financial-jepa-encoder.md",
        REPO_ROOT / "docs" / "models" / "ct-tgnn-evolvegcn.md",
        REPO_ROOT / "docs" / "models" / "aiops-stgcn-and-causal-rca.md",
    ]
    for mf in model_files:
        text = mf.read_text(encoding="utf-8")
        assert "Experimental" in text, f"Missing Experimental tier notice in {mf}"
        assert "Scope & Maturity Notice" in text, f"Missing Scope & Maturity Notice in {mf}"

    dataset_files = [
        REPO_ROOT / "docs" / "datasets" / "market-1t.md",
        REPO_ROOT / "docs" / "datasets" / "sec-edgar-fundamentals.md",
        REPO_ROOT / "docs" / "datasets" / "aiops-telemetry.md",
    ]
    for df in dataset_files:
        text = df.read_text(encoding="utf-8")
        assert "Experimental" in text, f"Missing Experimental tier notice in {df}"
        assert (
            "Scope & Data Availability Notice" in text
        ), f"Missing Scope & Data Availability Notice in {df}"

    refs_text = (REPO_ROOT / "docs" / "REFERENCES.md").read_text(encoding="utf-8")
    assert "REF-42" in refs_text
    assert "arXiv:2108.13264" in refs_text


def test_readme_links_resolve_cleanly() -> None:
    """All local file and anchor links in README.md must resolve with zero errors."""
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_readme_links.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Broken links in README.md:\n{res.stdout}\n{res.stderr}"
