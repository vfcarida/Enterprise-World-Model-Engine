"""Script to verify that all GitHub Action uses references are pinned to full 40-character SHAs.

Enforces supply-chain immutability and OpenSSF Scorecard compliance.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Regex matching 40-character hexadecimal SHA
SHA_REGEX = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
USES_REGEX = re.compile(r"^\s*(?:-\s+)?uses:\s*([^\s#]+)(?:\s*#\s*(.*))?$")


def check_action_pins(repo_root: Path) -> int:
    """Scan all workflows and actions for unpinned uses references."""
    yaml_files: list[Path] = []
    workflows_dir = repo_root / ".github" / "workflows"
    actions_dir = repo_root / ".github" / "actions"

    if workflows_dir.exists():
        yaml_files.extend(workflows_dir.glob("*.yml"))
        yaml_files.extend(workflows_dir.glob("*.yaml"))

    if actions_dir.exists():
        yaml_files.extend(actions_dir.rglob("*.yml"))
        yaml_files.extend(actions_dir.rglob("*.yaml"))

    errors: list[str] = []
    total_uses = 0

    for file_path in sorted(yaml_files):
        rel_path = file_path.relative_to(repo_root)
        lines = file_path.read_text(encoding="utf-8").splitlines()

        for idx, line in enumerate(lines, start=1):
            match = USES_REGEX.match(line)
            if not match:
                continue

            action_ref = match.group(1).strip()
            comment = (match.group(2) or "").strip()

            # Ignore local actions and docker images
            if action_ref.startswith("./") or action_ref.startswith("docker://"):
                continue

            total_uses += 1

            if "@" not in action_ref:
                errors.append(
                    f"{rel_path}:{idx}: Action '{action_ref}' lacks version/SHA reference."
                )
                continue

            action_name, ref = action_ref.split("@", 1)
            if not SHA_REGEX.match(ref):
                errors.append(
                    f"{rel_path}:{idx}: Action '{action_name}' is pinned to mutable ref '@{ref}'. "
                    f"Must be pinned to an immutable 40-character commit SHA."
                )
            elif not comment:
                # Warning or enforcement for missing version comment
                errors.append(
                    f"{rel_path}:{idx}: Action '{action_name}' is missing a human-readable tag comment (e.g., '# v1.2.3')."
                )

    print(
        f"Action SHA Pinning Check: Scanned {len(yaml_files)} YAML files, {total_uses} external action references."
    )

    if errors:
        print("\n[ERROR] Unpinned or non-compliant GitHub Action references found:")
        for err in errors:
            print(f"  - {err}")
        print(f"\nTotal violations: {len(errors)}")
        return 1

    print("[SUCCESS] All GitHub Action references are securely pinned to 40-character commit SHAs.")
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    sys.exit(check_action_pins(root))
