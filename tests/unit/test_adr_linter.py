"""Unit tests for the ADR Link and Status Integrity Linter (scripts/check_adr_links.py)."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path so we can import the script
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_adr_links import (  # noqa: E402
    check_adr_references,
    check_reciprocal_status,
    find_existing_adrs,
    main,
)


def test_adr_linter_passes_on_repository() -> None:
    """Ensure the ADR linter exits with code 0 on the actual repository."""
    assert main() == 0


def test_adr_linter_detects_nonexistent_adr(tmp_path: Path) -> None:
    """Verify that referencing an undefined ADR number is caught as an error."""
    # Create fake ADR dir with ADR-001
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)
    adr_1 = adr_dir / "ADR-001-initial.md"
    adr_1.write_text("# ADR-001: Initial\n\nReferences ADR-999.\n", encoding="utf-8")

    adrs = find_existing_adrs(adr_dir)
    assert 1 in adrs

    errors = check_adr_references(tmp_path, adrs)
    assert any("Referenced nonexistent ADR-999" in err for err in errors)


def test_adr_linter_detects_missing_reciprocal_supersedes(tmp_path: Path) -> None:
    """Verify that if ADR-001 is superseded by ADR-002, ADR-002 must declare superseding ADR-001."""
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)

    adr_1 = adr_dir / "ADR-001-old.md"
    adr_1.write_text(
        "# ADR-001: Old\n\n* Status: superseded by [ADR-002](ADR-002-new.md)\n",
        encoding="utf-8",
    )

    adr_2 = adr_dir / "ADR-002-new.md"
    # Notice: adr_2 forgets to mention superseding ADR-001
    adr_2.write_text(
        "# ADR-002: New\n\n* Status: accepted\n* Supersedes: None\n",
        encoding="utf-8",
    )

    adrs = find_existing_adrs(adr_dir)
    errors = check_reciprocal_status(adrs)
    assert any("does not list 'Supersedes: ADR-001'" in err for err in errors)


def test_adr_linter_accepts_valid_reciprocal_supersedes(tmp_path: Path) -> None:
    """Verify that valid reciprocal supersedes backlinks pass without error."""
    adr_dir = tmp_path / "docs" / "adr"
    adr_dir.mkdir(parents=True)

    adr_1 = adr_dir / "ADR-001-old.md"
    adr_1.write_text(
        "# ADR-001: Old\n\n* Status: superseded by [ADR-002](ADR-002-new.md)\n",
        encoding="utf-8",
    )

    adr_2 = adr_dir / "ADR-002-new.md"
    adr_2.write_text(
        "# ADR-002: New\n\n* Status: accepted\n* Supersedes: [ADR-001](ADR-001-old.md)\n",
        encoding="utf-8",
    )

    adrs = find_existing_adrs(adr_dir)
    errors = check_reciprocal_status(adrs)
    assert not errors
