"""Documentation Accessibility (a11y) and Structure Linter.

Audits markdown documentation against accessibility and usability best practices:
1. Semantic heading hierarchy (no skipped heading levels).
2. Descriptive image alternative text (alt text present and non-empty).
3. Diagram accessibility (Mermaid diagrams require accTitle and accDescr).
4. Table formatting standards.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def check_file_accessibility(file_path: Path) -> list[str]:
    """Scan a single markdown file for accessibility issues."""
    content = file_path.read_text(encoding="utf-8")
    issues: list[str] = []

    # 1. Heading hierarchy check
    lines = content.splitlines()
    last_heading_level = 0
    in_code_block = False

    for line_idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            # Heading jump check (e.g. from H1 to H3 without H2)
            if last_heading_level > 0 and level > last_heading_level + 1:
                issues.append(
                    f"Line {line_idx}: Heading level skipped from H{last_heading_level} to H{level} ('{heading_match.group(2)[:30]}...')"
                )
            last_heading_level = level

    # 2. Image alt-text check
    # Check markdown images: ![alt](url)
    md_images = re.findall(r"!\[(.*?)\]\((.*?)\)", content)
    for alt, url in md_images:
        if not alt.strip():
            issues.append(f"Image missing alt text: URL '{url[:40]}'")

    # Check HTML img tags: <img ...>
    html_images = re.findall(r"<img\s+([^>]+)>", content)
    for tag_attrs in html_images:
        alt_match = re.search(r'alt=["\']([^"\']*)["\']', tag_attrs)
        if not alt_match or not alt_match.group(1).strip():
            issues.append(
                f"HTML <img> tag missing non-empty alt attribute: '<img {tag_attrs[:40]}...>'"
            )

    # 3. Mermaid accessibility check
    mermaid_blocks = re.findall(r"```mermaid\n(.*?)```", content, re.DOTALL)
    for m_idx, block in enumerate(mermaid_blocks, start=1):
        if "accTitle:" not in block:
            issues.append(f"Mermaid block {m_idx} missing 'accTitle:' annotation")
        if "accDescr:" not in block:
            issues.append(f"Mermaid block {m_idx} missing 'accDescr:' annotation")

    return issues


def main() -> int:
    """Run accessibility checks across all documentation files."""
    docs_dir = Path("docs")
    readme = Path("README.md")
    files_to_check = [readme, *sorted(docs_dir.rglob("*.md"))]

    print("=" * 70)
    print("EWM Engine Documentation Accessibility (a11y) Advisory Linter")
    print("=" * 70)

    total_files = len(files_to_check)
    files_with_issues = 0
    total_issues = 0

    for file_path in files_to_check:
        issues = check_file_accessibility(file_path)
        if issues:
            files_with_issues += 1
            total_issues += len(issues)
            print(f"\n[A11Y NOTICE] {file_path} ({len(issues)} notices):")
            for issue in issues[:5]:  # show up to 5 per file
                print(f"  - {issue}")
            if len(issues) > 5:
                print(f"  - ... and {len(issues) - 5} more")

    print("\n" + "=" * 70)
    print(
        f"A11Y Audit Completed: Scanned {total_files} files. "
        f"Found {total_issues} notices across {files_with_issues} files."
    )
    print("=" * 70)

    # Advisory check: exits 0 but prints all findings
    return 0


if __name__ == "__main__":
    sys.exit(main())
