"""ADR Link and Status Integrity Linter.

Verifies that:
1. All ADR files in docs/adr/ are uniquely numbered and conform to naming conventions.
2. All references to ADR-XXX across docs, proposals, RFCs, and root governance files resolve to existing ADRs.
3. Reciprocal backlinks for superseded ADRs are strictly maintained (ADR-A superseded by ADR-B <==> ADR-B supersedes ADR-A).
4. All API Change Proposals (.github/proposals/) and RFCs (rfcs/) reference valid ADRs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ADR_FILENAME_REGEX = re.compile(r"^ADR-(\d{3,4})-(.+)\.md$")
ADR_REF_REGEX = re.compile(r"\bADR-(\d{3,4})\b")
SUPERSEDED_BY_REGEX = re.compile(
    r"(?:superseded\s+by\s+(?:\[?ADR-)?(\d{3,4})|Status:\s*superseded\s+by\s+\[?ADR-(\d{3,4}))",
    re.IGNORECASE,
)
SUPERSEDES_REGEX = re.compile(
    r"(?:Supersedes:\s*(?:\[?ADR-)?(\d{3,4})|supersedes\s+\[?ADR-(\d{3,4}))",
    re.IGNORECASE,
)


def find_existing_adrs(adr_dir: Path) -> dict[int, Path]:
    """Index all ADR files by their integer number."""
    existing_adrs: dict[int, Path] = {}
    for path in sorted(adr_dir.glob("ADR-*.md")):
        match = ADR_FILENAME_REGEX.match(path.name)
        if match:
            num = int(match.group(1))
            if num in existing_adrs:
                print(
                    f"[ERROR] Duplicate ADR number {num}: {path.name} and {existing_adrs[num].name}"
                )
                sys.exit(1)
            existing_adrs[num] = path
    return existing_adrs


def check_reciprocal_status(existing_adrs: dict[int, Path]) -> list[str]:
    """Check reciprocal superseded / supersedes backlinks between ADRs."""
    errors: list[str] = []
    superseded_by_map: dict[int, set[int]] = {}
    supersedes_map: dict[int, set[int]] = {}

    for num, path in existing_adrs.items():
        content = path.read_text(encoding="utf-8")

        # Find any 'superseded by' references
        for m in SUPERSEDED_BY_REGEX.finditer(content):
            target = int(m.group(1) or m.group(2))
            superseded_by_map.setdefault(num, set()).add(target)

        # Find any 'supersedes' references
        for m in SUPERSEDES_REGEX.finditer(content):
            target = int(m.group(1) or m.group(2))
            supersedes_map.setdefault(num, set()).add(target)

    # Validate reciprocal relationships
    for source_num, targets in superseded_by_map.items():
        for target_num in targets:
            if target_num not in existing_adrs:
                errors.append(
                    f"ADR-{source_num:03d} ({existing_adrs[source_num].name}) claims to be superseded by nonexistent ADR-{target_num:03d}"
                )
            elif source_num not in supersedes_map.get(target_num, set()):
                errors.append(
                    f"ADR-{source_num:03d} is superseded by ADR-{target_num:03d}, but ADR-{target_num:03d} does not list 'Supersedes: ADR-{source_num:03d}'"
                )

    for source_num, targets in supersedes_map.items():
        for target_num in targets:
            if target_num not in existing_adrs:
                errors.append(
                    f"ADR-{source_num:03d} ({existing_adrs[source_num].name}) claims to supersede nonexistent ADR-{target_num:03d}"
                )
            elif source_num not in superseded_by_map.get(target_num, set()):
                errors.append(
                    f"ADR-{source_num:03d} supersedes ADR-{target_num:03d}, but ADR-{target_num:03d} is not marked 'superseded by ADR-{source_num:03d}'"
                )

    return errors


def check_adr_references(repo_root: Path, existing_adrs: dict[int, Path]) -> list[str]:
    """Check that all ADR references in the codebase point to valid ADR numbers."""
    errors: list[str] = []
    files_to_scan: list[Path] = []

    # Directories to scan
    scan_dirs = [
        repo_root / "docs",
        repo_root / ".github" / "proposals",
        repo_root / "rfcs",
    ]
    for d in scan_dirs:
        if d.exists():
            files_to_scan.extend(d.rglob("*.md"))

    # Root markdown files
    for root_file in repo_root.glob("*.md"):
        files_to_scan.append(root_file)

    for path in sorted(files_to_scan):
        # Ignore templates
        if "TEMPLATE" in path.name.upper() or "0000-" in path.name:
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            errors.append(f"Could not read {path}: {exc}")
            continue

        lines = content.splitlines()
        for idx, line in enumerate(lines, start=1):
            for match in ADR_REF_REGEX.finditer(line):
                adr_num = int(match.group(1))
                if adr_num not in existing_adrs:
                    rel_path = path.relative_to(repo_root)
                    errors.append(f"{rel_path}:{idx}: Referenced nonexistent ADR-{adr_num:03d}")

    return errors


def check_proposals_and_rfcs(repo_root: Path, existing_adrs: dict[int, Path]) -> list[str]:
    """Ensure all API Change Proposals and RFCs cross-link to valid ADRs."""
    errors: list[str] = []

    proposals_dir = repo_root / ".github" / "proposals"
    if proposals_dir.exists():
        for prop_path in sorted(proposals_dir.glob("ACP-*.md")):
            content = prop_path.read_text(encoding="utf-8")
            refs = [int(m.group(1)) for m in ADR_REF_REGEX.finditer(content)]
            if not refs:
                rel = prop_path.relative_to(repo_root)
                errors.append(f"{rel}: Proposal does not cross-link to any ADR")

    rfcs_dir = repo_root / "rfcs"
    if rfcs_dir.exists():
        for rfc_path in sorted(rfcs_dir.glob("RFC-*.md")):
            if "0000" in rfc_path.name:
                continue
            content = rfc_path.read_text(encoding="utf-8")
            refs = [int(m.group(1)) for m in ADR_REF_REGEX.finditer(content)]
            if not refs:
                rel = rfc_path.relative_to(repo_root)
                errors.append(f"{rel}: RFC does not cross-link to any ADR")

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    adr_dir = repo_root / "docs" / "adr"

    if not adr_dir.exists():
        print(f"[ERROR] ADR directory not found at {adr_dir}")
        return 1

    existing_adrs = find_existing_adrs(adr_dir)
    print(f"ADR Linter: Discovered {len(existing_adrs)} ADRs in {adr_dir.relative_to(repo_root)}")

    all_errors: list[str] = []

    # 1. Reciprocal status check
    reciprocal_errors = check_reciprocal_status(existing_adrs)
    all_errors.extend(reciprocal_errors)

    # 2. Cross-reference existence check
    reference_errors = check_adr_references(repo_root, existing_adrs)
    all_errors.extend(reference_errors)

    # 3. ACP and RFC cross-linking check
    proposal_errors = check_proposals_and_rfcs(repo_root, existing_adrs)
    all_errors.extend(proposal_errors)

    if all_errors:
        print("\n[ERROR] ADR Integrity Violations Found:")
        for err in all_errors:
            print(f"  - {err}")
        print(f"\nTotal violations: {len(all_errors)}")
        return 1

    print("[SUCCESS] All ADR references, reciprocal backlinks, and proposal links are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
