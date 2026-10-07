# Newsfragments

This directory contains individual changelog entries ("news fragments") for [Towncrier](https://towncrier.readthedocs.io/).
Rather than editing `CHANGELOG.md` directly in pull requests (which causes merge conflicts across branches), contributors submit a small Markdown file in this directory.

## How to Author a News Fragment

When creating a pull request, add a file in this directory named:

`<pr_or_issue_number>.<type>.md`

### Fragment Types
- **`.feature.md`**: New features or capabilities.
- **`.bugfix.md`**: Fixes for bugs or unexpected behavior.
- **`.doc.md`**: Significant documentation improvements or guides.
- **`.removal.md`**: Deprecations, API removals, or breaking changes (adhering to NEP-23).
- **`.misc.md`**: Internal refactoring, dependency updates, or CI improvements.

### Format & Style
- Write in standard GitHub Flavored Markdown.
- Keep the summary user-facing (1–3 sentences explaining *why* and *what* changed).
- Link to relevant symbols or ADRs if applicable.

*Example*: `newsfragments/42.feature.md`:
```markdown
Added `@deprecated` decorator and `deprecate()` runtime warning utility in `ewm_engine.core.deprecation` adhering to NEP-23 guidelines.
```

### Escape Hatch
If a pull request does not require a changelog entry (e.g., trivial typo fix or CI experiment), apply the `skip-changelog` label to the pull request to bypass the `towncrier check` CI gate.
