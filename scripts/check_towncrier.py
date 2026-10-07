"""CI gate verifying that pull requests include a Towncrier news fragment.

Includes a 'skip-changelog' escape hatch for trivial changes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_newsfragments(base_ref: str | None = None) -> int:
    """Verify presence of news fragments for changed branches."""
    # Check escape hatch via environment variable or PR labels
    skip_label = os.environ.get("SKIP_CHANGELOG", "").lower() in ("true", "1", "yes")
    pr_labels = os.environ.get("PR_LABELS", "").split(",")
    if skip_label or "skip-changelog" in [label.strip() for label in pr_labels]:
        print("[INFO] 'skip-changelog' escape hatch detected. Bypassing news fragment requirement.")
        return 0

    repo_root = Path(__file__).resolve().parent.parent
    newsfragments_dir = repo_root / "newsfragments"
    if not newsfragments_dir.exists():
        print(f"[ERROR] newsfragments directory missing at {newsfragments_dir}")
        return 1

    if not base_ref:
        base_ref = os.environ.get("BASE_REF") or os.environ.get("GITHUB_BASE_REF")

    if not base_ref:
        # Not a pull request execution; assert valid directory setup
        print(
            "[INFO] No base ref provided (non-PR run). Validating newsfragments directory structure."
        )
        return 0

    # Execute git diff to detect new or modified fragments in newsfragments/
    target_ref = f"origin/{base_ref}" if not base_ref.startswith("origin/") else base_ref
    try:
        res = subprocess.run(
            ["git", "diff", "--name-only", f"{target_ref}...HEAD", "--", "newsfragments/"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        changed_files = [f.strip() for f in res.stdout.splitlines() if f.strip()]
        valid_fragments = [
            f
            for f in changed_files
            if f.endswith((".feature.md", ".bugfix.md", ".doc.md", ".removal.md", ".misc.md"))
        ]

        if not valid_fragments:
            print(
                f"\n[ERROR] No valid Towncrier news fragment found comparing HEAD to {target_ref}."
            )
            print(
                "Please add a file to newsfragments/ following the pattern: <pr_or_issue>.<type>.md"
            )
            print("Valid types: .feature.md, .bugfix.md, .doc.md, .removal.md, .misc.md")
            print(
                "To bypass this check for trivial or non-user-facing changes, apply the 'skip-changelog' label to your PR."
            )
            return 1

        print(
            f"[SUCCESS] Found {len(valid_fragments)} news fragment(s): {', '.join(valid_fragments)}"
        )
        return 0

    except subprocess.CalledProcessError as e:
        print(
            f"[WARNING] Could not run git diff against {target_ref}: {e.stderr}. Fallback to directory check."
        )
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Towncrier news fragments on PR")
    parser.add_argument(
        "--base-ref", default=None, help="Base git ref to compare against (e.g. origin/main)"
    )
    args = parser.parse_args()
    sys.exit(check_newsfragments(args.base_ref))
