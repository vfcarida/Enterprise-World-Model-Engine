# Curated Good First Issues & Contributor Tasks

Welcome to the **Enterprise World Model Engine (EWM Engine)**!

We actively maintain a catalog of curated, scoped starter tasks for new contributors. Each task is mentored by an experienced maintainer, includes clear reproduction/implementation targets, and provides pointer files to help you get oriented quickly.

Before starting:
1. Review [CONTRIBUTING.md](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/CONTRIBUTING.md) for local environment setup with `uv`.
2. Review [CODE_OF_CONDUCT.md](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/CODE_OF_CONDUCT.md).
3. Comment on the corresponding issue (or open a PR referencing the task ID below) so a mentor can assign it to you.

---

## Issue Catalog

### GFI-001: Add LaTeX String Sanitization to Equation Exporters
- **Mentor:** `@vfcarida` (Vinicius Caridá)
- **Area:** `area:core`
- **Difficulty:** Introductory
- **Description:** When exporting systemic trace DAGs or constraint formulas to LaTeX, certain variable names containing underscores (e.g., `inventory_level_t`) can cause unescaped math errors.
- **Relevant Files:** `src/ewm_engine/provenance/trace.py`, `tests/unit/test_trace.py`
- **Expected Outcome:** Add a helper `sanitize_latex_identifier(name: str) -> str` that escapes raw underscores and math symbols; verify with parameterized unit tests.

---

### GFI-002: Add Explicit Validation Test for StateFingerprint with Empty Metadata
- **Mentor:** `@vfcarida`
- **Area:** `area:core`
- **Difficulty:** Introductory
- **Description:** The canonical state fingerprinting routine handles empty metadata dictionaries gracefully, but this edge case lacks an explicit regression test in `test_provenance.py`.
- **Relevant Files:** `src/ewm_engine/core/state.py`, `tests/unit/test_provenance.py`
- **Expected Outcome:** Add a test verifying that `state.fingerprint()` produces a stable, non-empty 64-character hex SHA-256 when `metadata={}`.

---

### GFI-003: Improve CLI Color Palette & Help Formatting for `ewm --help`
- **Mentor:** `@vfcarida`
- **Area:** `area:cli`
- **Difficulty:** Introductory
- **Description:** Improve the Rich CLI console output for top-level `ewm --help` and subcommands (`ewm run`, `ewm inspect`, `ewm validate`) to use the official purple/cyan accent theme matching project branding.
- **Relevant Files:** `src/ewm_engine/cli/main.py`, `tests/unit/test_cli.py`
- **Expected Outcome:** Consistent Rich styles and updated snapshot assertions in CLI tests.

---

### GFI-004: Add Markdown Table Formatting Utility for Scenario Comparisons
- **Mentor:** `@vfcarida`
- **Area:** `area:reporting`
- **Difficulty:** Introductory
- **Description:** Add a convenience function `format_comparison_table_markdown(result: ScenarioComparisonResult) -> str` to render side-by-side metric tables in GitHub Flavored Markdown.
- **Relevant Files:** `src/ewm_engine/reporting/formatters.py`, `tests/unit/test_reporting.py`
- **Expected Outcome:** Pure function formatting candidate vs. baseline deltas with bootstrap confidence intervals as a GFM table; 100% test coverage.

---

### GFI-005: Add Type Annotations & Docstrings to Benchmark Fixtures
- **Mentor:** `@vfcarida`
- **Area:** `area:benchmarks`
- **Difficulty:** Introductory
- **Description:** Several test fixtures in `benchmarks/families/` lack explicit type annotations under Mypy strict mode.
- **Relevant Files:** `benchmarks/families/multi_agent_cascade.py`, `benchmarks/protocol.py`
- **Expected Outcome:** Add full PEP 484 type hints and NumPy docstrings; ensure `mypy --strict benchmarks` succeeds.

---

### GFI-006: Add Validation Warning When Constraint Threshold Is Non-Finite
- **Mentor:** `@vfcarida`
- **Area:** `area:constraints`
- **Difficulty:** Introductory
- **Description:** If a user defines a numeric threshold constraint with `float('nan')` or `float('inf')`, the constraint evaluator should raise a descriptive `ValueError` or emit a warning during specification construction rather than during rollout step $t$.
- **Relevant Files:** `src/ewm_engine/constraints/evaluator.py`, `tests/unit/test_constraints.py`
- **Expected Outcome:** Add validation check in constraint constructor and unit test verifying clear error message.

---

### GFI-007: Add JSON Schema Version Badge Generator Script
- **Mentor:** `@vfcarida`
- **Area:** `area:tooling`
- **Difficulty:** Introductory
- **Description:** Create a lightweight developer script `scripts/generate_schema_badges.py` that reads JSON schemas from `schemas/` and generates static SVG/JSON badges representing schema versioning.
- **Relevant Files:** `scripts/generate_schema_badges.py`, `schemas/`
- **Expected Outcome:** Clean standalone script adhering to `ruff` and type annotations.

---

### GFI-008: Improve Error Diagnosability on Unknown YAML Top-Level Keys
- **Mentor:** `@vfcarida`
- **Area:** `area:serialization`
- **Difficulty:** Introductory
- **Description:** When loading a world specification from YAML via `load_world_yaml`, unrecognized top-level keys should list available valid alternatives (e.g., did you mean `entities` instead of `entity`?).
- **Relevant Files:** `src/ewm_engine/serialization/yaml_loader.py`, `tests/unit/test_serialization.py`
- **Expected Outcome:** Leverage `difflib.get_close_matches` to suggest typos in `InvalidSpecificationError`.

---

### GFI-009: Add Sensitivity Analysis Example Jupyter Notebook
- **Mentor:** `@vfcarida`
- **Area:** `area:docs`
- **Difficulty:** Intermediate
- **Description:** Add an educational walkthrough notebook in `examples/notebooks/sensitivity_analysis_walkthrough.ipynb` demonstrating Sobol global sensitivity analysis using the SALib adapter.
- **Relevant Files:** `examples/notebooks/`, `docs/tutorials/`
- **Expected Outcome:** Executable, well-documented notebook with synthetic supply chain dynamics.

---

### GFI-010: Add Pre-Commit Config for Trailing Whitespace and End-of-File Fixer
- **Mentor:** `@vfcarida`
- **Area:** `area:infra`
- **Difficulty:** Introductory
- **Description:** Provide a `.pre-commit-config.yaml` file that runs standard linters (`ruff`, `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`) locally before commit.
- **Relevant Files:** `.pre-commit-config.yaml`, `CONTRIBUTING.md`
- **Expected Outcome:** Tested pre-commit configuration documented in contributor guide.

---

### GFI-011: Add Mermaid Diagram Syntax Validator Utility
- **Mentor:** `@vfcarida`
- **Area:** `area:trace`
- **Difficulty:** Introductory
- **Description:** Add a verification utility `validate_mermaid_syntax(mermaid_str: str) -> bool` to confirm exported Mermaid graph strings contain balanced node identifiers and legal edge arrows (`-->`).
- **Relevant Files:** `src/ewm_engine/provenance/mermaid.py`, `tests/unit/test_mermaid.py`
- **Expected Outcome:** Parser validating Mermaid structure; comprehensive unit test suite.

---

### GFI-012: Add Summary Statistics Exporter for Monte Carlo Rollouts
- **Mentor:** `@vfcarida`
- **Area:** `area:simulation`
- **Difficulty:** Introductory
- **Description:** Add helper functions to compute non-parametric distributional summaries (median, IQR, 5th/95th percentiles) from `SimulationResult.trajectories` across all sample paths.
- **Relevant Files:** `src/ewm_engine/simulation/statistics.py`, `tests/unit/test_simulation_statistics.py`
- **Expected Outcome:** Deterministic math functions without external dependencies; tested against synthetic trajectories.

---

## How to Claim an Issue
To claim any of these tasks:
1. Open a comment on GitHub stating: *"I would like to work on GFI-XXX, please assign me."*
2. The designated mentor (`@vfcarida`) will assign the issue, guide your PR, and review your code.
3. Once merged, you will be recognized in the [all-contributors](https://github.com/vfcarida/Enterprise-World-Model-Engine#contributors) table and enter the [SPEC 9 Maintainer Ladder](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/GOVERNANCE.md).
