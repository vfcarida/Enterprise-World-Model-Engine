"""Verification suite for Prompt E10: Documentation Architecture and API Documentation.

Verifies:
1. Historical documentation archive in `/docs/legacy/`.
2. API reference documentation for `StateEncoder`, `RetrieveAndReasonEngine`, and
   `EngineCausalInterface` (`docs/reference/api/ocel_cda_causal.md` & `docs/reference/api/core.md`).
3. Step-by-step tutorials for the E7 SE microservice incident cascade (`docs/tutorials/swe.md`)
   and E6 Finance double-entry ledger (`docs/tutorials/finance.md`).
4. Explicit documentation of the mathematical boundaries of causal identification
   (unmeasured confounders violating the modularity assumption) in `docs/concepts/causality.md`.
5. 100% clean execution of all ```python ... ``` code blocks across the updated documentation.
6. Semantic artifact-hygiene check confirming zero leaked API keys, chat transcript artifacts,
   or assistant narration blocks.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"


def test_e10_legacy_documentation_archive_exists() -> None:
    """Verify previous documentation files are archived in /docs/legacy/."""
    legacy_dir = DOCS_DIR / "legacy"
    expected_legacy_files = (
        "README_legacy.md",
        "index_legacy.md",
        "getting-started_legacy.md",
        "architecture_legacy.md",
        "causality_legacy.md",
        "api_core_legacy.md",
        "finance_tutorial_legacy.md",
        "swe_tutorial_legacy.md",
    )
    for fname in expected_legacy_files:
        fpath = legacy_dir / fname
        assert fpath.exists(), f"Missing archived legacy documentation file: {fpath}"
        assert fpath.stat().st_size > 100


def test_e10_api_reference_and_causal_identification_boundaries() -> None:
    """Verify API reference for StateEncoder, RetrieveAndReasonEngine, EngineCausalInterface and causal limits."""
    api_ref_path = DOCS_DIR / "reference" / "api" / "ocel_cda_causal.md"
    core_api_path = DOCS_DIR / "reference" / "api" / "core.md"
    causality_path = DOCS_DIR / "concepts" / "causality.md"
    arch_path = DOCS_DIR / "architecture.md"

    assert api_ref_path.exists()
    api_text = api_ref_path.read_text(encoding="utf-8")
    for symbol in ("StateEncoder", "RetrieveAndReasonEngine", "EngineCausalInterface"):
        assert symbol in api_text
        assert symbol in core_api_path.read_text(encoding="utf-8")

    causality_text = causality_path.read_text(encoding="utf-8")
    for required_concept in (
        "Unmeasured Confounders",
        "Modularity Assumption",
        "non-identifiable",
        "Markov Equivalence Class",
        "Positivity",
    ):
        assert required_concept in causality_text, (
            f"Missing causal boundary concept '{required_concept}' in docs/concepts/causality.md"
        )

    arch_text = arch_path.read_text(encoding="utf-8")
    assert "Core Engine" in arch_text
    assert "Finance Domain Pack" in arch_text
    assert "Software Engineering Pack" in arch_text


def test_e10_all_tutorial_and_api_python_code_blocks_execute_cleanly() -> None:
    """Execute every ```python code block in the E10 documentation files to confirm zero errors."""
    target_docs = (
        DOCS_DIR / "reference" / "api" / "ocel_cda_causal.md",
        DOCS_DIR / "tutorials" / "swe.md",
        DOCS_DIR / "tutorials" / "finance.md",
        DOCS_DIR / "getting-started.md",
        DOCS_DIR / "concepts" / "causality.md",
    )
    code_block_re = re.compile(r"```python\r?\n(.*?)```", re.DOTALL)

    total_blocks_executed = 0
    for doc_path in target_docs:
        assert doc_path.exists(), f"Missing documentation file: {doc_path}"
        text = doc_path.read_text(encoding="utf-8")
        blocks = code_block_re.findall(text)
        assert len(blocks) >= 1, f"Expected at least 1 Python block in {doc_path}"
        for idx, code in enumerate(blocks, start=1):
            ns: dict[str, object] = {"__name__": f"__{doc_path.stem}_block_{idx}__"}
            compiled = compile(code, f"<{doc_path.name} block #{idx}>", "exec")
            exec(compiled, ns)
            total_blocks_executed += 1

    assert total_blocks_executed >= 8


def test_e10_semantic_artifact_hygiene_no_leaked_keys_or_chat_narration() -> None:
    """Confirm zero leaked API keys, chat transcript artifacts, or assistant narration in docs."""
    target_docs = (
        DOCS_DIR / "index.md",
        DOCS_DIR / "getting-started.md",
        DOCS_DIR / "architecture.md",
        DOCS_DIR / "concepts" / "causality.md",
        DOCS_DIR / "reference" / "api" / "index.md",
        DOCS_DIR / "reference" / "api" / "core.md",
        DOCS_DIR / "reference" / "api" / "ocel_cda_causal.md",
        DOCS_DIR / "tutorials" / "swe.md",
        DOCS_DIR / "tutorials" / "finance.md",
    )
    forbidden_patterns = (
        r"sk-[a-zA-Z0-9]{20,}",
        r"ghp_[a-zA-Z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"AIza[0-9A-Za-z\-_]{35}",
        r"<USER_REQUEST>",
        r"<SYSTEM_MESSAGE>",
        r"As an AI language model",
        r"Here is the updated file",
    )
    combined_re = re.compile("|".join(forbidden_patterns))

    for doc_path in target_docs:
        text = doc_path.read_text(encoding="utf-8")
        match = combined_re.search(text)
        assert match is None, (
            f"Artifact hygiene violation in {doc_path}: found forbidden pattern {match.group(0)!r}"
        )
