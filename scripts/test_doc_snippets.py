"""Automated documentation code snippet execution gate.

Extracts and executes runnable Python code blocks across tutorials,
getting-started, and reference examples to prevent code-documentation drift.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def extract_python_snippets(md_path: Path) -> list[str]:
    """Extract fenced python code blocks from a markdown document."""
    text = md_path.read_text(encoding="utf-8")
    # Matches ```python ... ```
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    runnable = []
    for b in blocks:
        code = b.strip()
        # Skip signatures, pseudo-code fragments, or ellipses
        if (
            "from ewm_engine" in code
            or "import ewm_engine" in code
            or "SimulationEngine" in code
            or "WorldState" in code
        ):
            if "..." not in code.splitlines()[-1] and not code.startswith("def "):
                runnable.append(code)
    return runnable


def run_snippet(code: str, file_context: str, idx: int) -> bool:
    """Execute a single python snippet within an isolated dictionary scope."""
    scope: dict[str, object] = {"__name__": "__main__"}
    try:
        exec(code, scope)
        return True
    except Exception as exc:
        sys.stderr.write(
            f"\n[ERROR] Snippet execution failed in {file_context} (Block {idx}):\n"
            f"{exc}\n\nSnippet Code:\n{code}\n"
        )
        return False


def main() -> int:
    """Scan and execute all candidate markdown snippets."""
    target_docs = [
        Path("README.md"),
        Path("docs/getting-started.md"),
        Path("docs/tutorials/first-world.md"),
        Path("docs/tutorials/counterfactual-interventions.md"),
    ]

    total_snippets = 0
    passed_snippets = 0
    failed = False

    print("=" * 70)
    print("EWM Engine Documentation Snippet Execution Gate")
    print("=" * 70)

    for doc in target_docs:
        if not doc.exists():
            print(f"[WARN] File not found: {doc}")
            continue

        snippets = extract_python_snippets(doc)
        if not snippets:
            continue

        print(f"\n[TESTING] {doc} ({len(snippets)} runnable snippets found)")
        # We allow cumulative execution per document so multi-step tutorials share state
        doc_scope: dict[str, object] = {"__name__": "__main__"}
        for idx, snippet in enumerate(snippets, start=1):
            total_snippets += 1
            try:
                exec(snippet, doc_scope)
                passed_snippets += 1
                print(f"  [PASS] Snippet {idx}")
            except Exception as exc:
                print(f"  [FAIL] Snippet {idx}: {exc}")
                failed = True

    print("\n" + "=" * 70)
    print(f"Results: {passed_snippets}/{total_snippets} snippets passed successfully.")
    print("=" * 70)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
