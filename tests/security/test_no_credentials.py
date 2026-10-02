"""Security test (AC-023): Verify no hardcoded secrets, API tokens, or private keys exist in the repository."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

SECRET_PATTERNS = [
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub Personal Access Token", re.compile(r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}\b")),
    ("PyPI API Token", re.compile(r"\bpypi-AgEIcHlwaS5vcmc[A-Za-z0-9_-]{50,}\b")),
    ("Private Key Header", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    (
        "Generic Password/Secret Assignment",
        re.compile(
            r"""(?i)(?:api_key|secret_key|private_key|auth_token)\s*=\s*['"][a-zA-Z0-9_\-]{20,}['"]"""
        ),
    ),
]

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "site",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

SCAN_EXTENSIONS = {
    ".py",
    ".yaml",
    ".yml",
    ".json",
    ".md",
    ".toml",
    ".txt",
    ".sh",
}


@pytest.mark.security
def test_no_hardcoded_secrets_or_credentials() -> None:
    """AC-023: Scan repository files for accidental secrets, credentials, or private keys."""
    violations: list[str] = []

    for root, dirs, files in Path(REPO_ROOT).walk():
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            file_path = root / file
            if file_path.suffix not in SCAN_EXTENSIONS:
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            for line_no, line in enumerate(content.splitlines(), start=1):
                # Ignore references in this test file itself or docs mentioning regexes
                if file_path.name == "test_no_credentials.py":
                    continue

                for pattern_name, regex in SECRET_PATTERNS:
                    if regex.search(line):
                        rel_path = file_path.relative_to(REPO_ROOT)
                        violations.append(
                            f"{rel_path}:{line_no} matches {pattern_name}: {line.strip()[:60]}..."
                        )

    assert not violations, (
        f"Security violation: found potential hardcoded secrets ({len(violations)} found):\n"
        + "\n".join(violations)
    )
