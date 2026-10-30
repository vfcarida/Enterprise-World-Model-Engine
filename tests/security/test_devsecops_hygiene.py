"""DevSecOps hygiene, authorship, documentation card, and supply-chain verification tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.security
def test_devsecops_sanitization_and_audit_scripts_exist() -> None:
    """Verify sanitize_git_history.sh, audit_secrets.sh, and enforce_authorship.sh are present and valid."""
    sanitize_script = REPO_ROOT / "scripts" / "sanitize_git_history.sh"
    audit_script = REPO_ROOT / "scripts" / "audit_secrets.sh"
    author_script = REPO_ROOT / "scripts" / "enforce_authorship.sh"

    assert sanitize_script.exists()
    assert audit_script.exists()
    assert author_script.exists()

    sanitize_text = sanitize_script.read_text(encoding="utf-8")
    assert "git-filter-repo" in sanitize_text
    assert "git filter-branch" not in sanitize_text

    audit_text = audit_script.read_text(encoding="utf-8")
    assert "gitleaks" in audit_text

    author_text = author_script.read_text(encoding="utf-8")
    assert "Vinicius Caridá" in author_text


@pytest.mark.security
def test_architecture_model_cards_and_dataset_cards_present() -> None:
    """Verify ARCHITECTURE.md, model cards, and dataset cards exist and credit Vinicius Caridá."""
    required_docs = [
        REPO_ROOT / "ARCHITECTURE.md",
        REPO_ROOT / "docs" / "templates" / "README_TEMPLATE.md",
        REPO_ROOT / "docs" / "models" / "financial-jepa-encoder.md",
        REPO_ROOT / "docs" / "models" / "ct-tgnn-evolvegcn.md",
        REPO_ROOT / "docs" / "models" / "aiops-stgcn-and-causal-rca.md",
        REPO_ROOT / "docs" / "datasets" / "market-1t.md",
        REPO_ROOT / "docs" / "datasets" / "sec-edgar-fundamentals.md",
        REPO_ROOT / "docs" / "datasets" / "aiops-telemetry.md",
    ]

    for doc_path in required_docs:
        assert doc_path.exists(), f"Missing required documentation file: {doc_path}"
        text = doc_path.read_text(encoding="utf-8")
        assert "Caridá" in text or "Carida" in text
        assert "Generated with Claude" not in text
        assert "Co-Authored-By:" not in text


@pytest.mark.security
def test_no_prompt_scaffolding_or_ai_coauthor_trailers_in_git() -> None:
    """Verify zero tracked PROMPT-* files and zero Co-Authored-By AI trailers in git history."""
    ls_proc = subprocess.run(
        ["git", "ls-files", "*PROMPT*"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ls_proc.stdout.strip() == "", f"Tracked PROMPT files found: {ls_proc.stdout}"

    log_proc = subprocess.run(
        ["git", "log", "--all", "--grep=Co-Authored-By", "--oneline"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert log_proc.stdout.strip() == "", f"Co-Authored-By trailers found: {log_proc.stdout}"


@pytest.mark.security
def test_dockerfile_pinned_digest_and_opa_policy_present() -> None:
    """Verify Dockerfile uses a pinned @sha256 digest and OPA Gatekeeper Rego policy exists."""
    dockerfile = REPO_ROOT / "Dockerfile"
    assert dockerfile.exists()
    dockerfile_text = dockerfile.read_text(encoding="utf-8")
    assert "@sha256:" in dockerfile_text

    opa_policy = REPO_ROOT / "opa_policies" / "remediation_policy.rego"
    assert opa_policy.exists()
    assert "package kubernetes.admission" in opa_policy.read_text(encoding="utf-8")
