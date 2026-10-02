# ADR-014: CI/CD Quality Gates, Supply-Chain Hardening, and Reproducible Automation

## Status
Accepted

## Date
2026-10-02

## Context
As an enterprise simulation kernel, Enterprise World Model Engine (EWM Engine) must guarantee uncompromised correctness, mathematical reproducibility, backward compatibility, and security. The authoritative project specification (`docs/specs/spec-driven-development.md`) defines explicit quality gates, matrix constraints, coverage thresholds, and supply-chain requirements (Gap G16, AC-015, AC-016, AC-017, AC-022, AC-023).

Historically, continuous integration pipelines in Python repositories frequently suffer from:
1. **Implicit or Flaky Environments**: Drifting transitive dependencies caused by unpinned or floating package installations.
2. **Coarse-Grained Quality Gates**: Monolithic CI scripts where formatting, linting, type checking, and tests fail together without clear attribution.
3. **Supply-Chain Vulnerabilities**: Unpinned third-party GitHub Actions vulnerable to tag mutation, compromised upstream releases, and long-lived stored PyPI deployment tokens.
4. **Under-Tested Distribution Artifacts**: Packages published without verifying that the built wheel installs cleanly in an isolated environment and executes the Quickstart workflow without extraneous dev dependencies.

## Decision

### 1. Standardization on `uv`
All local development workflows and CI jobs standardize on `uv` (`astral-sh/setup-uv`) as the fast, deterministic virtual environment and package manager:
- `uv sync --extra dev` is used for consistent environment synchronization.
- All testing and linting commands execute via `uv run <command>`.
- The `setup-uv` cache is enabled across all workflow runners.

### 2. Twelve Distinct CI Quality Gates
The CI workflow (`.github/workflows/ci.yml`) is decomposed into twelve visible, independently verifiable quality gates:
1. `lint`: Static analysis and style verification with Ruff (`ruff check src tests examples`).
2. `format-check`: Formatting conformance verification with Ruff (`ruff format --check src tests examples`).
3. `type-check`: Strict static type checking with Mypy (`mypy src tests`).
4. `unit-tests`: Unit tests across a multi-platform matrix (`{ubuntu-latest, macos-latest, windows-latest} × {Python 3.11, 3.12}`) with coverage reporting.
5. `property-tests`: Invariant testing via Hypothesis (`pytest tests/property`).
6. `integration-tests`: End-to-end multi-step simulation tests (`pytest tests/integration`).
7. `contract-tests`: Public API, PEP 561 marker, schema snapshots, and serialization contract tests (`pytest tests/contract`).
8. `security-tests`: Safe deserialization, canonical data validation, and credential scanning (`pytest tests/security`).
9. `example-tests`: Normative acceptance fixtures including Minimal Warehouse and CivicFlow (`pytest tests/examples`).
10. `architecture-tests`: Import boundary and zero-leakage dependency assertions (`pytest tests/architecture`).
11. `package-build`: Wheel and sdist generation with `python -m build`, metadata verification with `twine check`, and clean virtualenv installation and Quickstart execution smoke test (`AC-016`, `AC-021`).
12. `docs-build`: Documentation build in strict mode (`mkdocs build --strict`).

A root aggregation gate (`ci-gates`) depends on all twelve jobs and serves as the mandatory status check for GitHub branch protection.

### 3. Coverage Thresholds and Core Area Gate
- **Global Coverage**: Overall test coverage is strictly enforced at $\ge 85.0\%$ in `pyproject.toml` (`[tool.coverage.report] fail_under = 85`).
- **Core Area Thresholds**: Core packages (`ewm_engine.core`, `ewm_engine.simulation`, `ewm_engine.constraints`, and `ewm_engine.provenance`) must maintain $\ge 90.0\%$ test coverage, verified by `scripts/check_coverage.py`.

### 4. Supply-Chain Hardening and Pinned Actions
- **Immutable Action SHAs**: Every third-party GitHub Action across all workflows (`ci.yml`, `docs.yml`, `codeql.yml`, `release.yml`) is pinned to a 40-character immutable commit SHA, annotated with human-readable version comments (e.g., `actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2`).
- **Automated CodeQL**: Weekly and pull-request CodeQL static security analysis for Python (`.github/workflows/codeql.yml`).
- **No Stored Tokens / OIDC Publishing**: Release automation (`.github/workflows/release.yml`) runs exclusively on `v*` git tags using GitHub Actions OpenID Connect (OIDC) Trusted Publishing to PyPI (`pypa/gh-action-pypi-publish`), eliminating stored credentials.
- **Pre-Merge Secret Scanning**: An automated AST/regex security test (`tests/security/test_no_credentials.py`) asserts that no AWS keys, GitHub tokens, PyPI tokens, or private keys are committed into the repository (`AC-023`).

### 5. Packaging and Core Dependency Isolation
- `py.typed` is packaged in the root of the source distribution and built wheels (`AC-017`).
- Contract tests assert that importing `ewm_engine` does not import optional heavy libraries (`torch`, `z3`, `networkx`) into `sys.modules` (`AC-019`).
- CI validates that core installs and imports in a completely clean environment without extras.

## Consequences

### Positive
- Every pull request provides immediate, clear visibility into which specific gate (lint, types, contracts, security, docs) passed or failed.
- Total reproducibility across OS platforms (Linux, macOS, Windows) and supported Python versions (3.11, 3.12).
- Zero exposure to supply-chain tag mutation or long-lived secret leaks.
- Confidence that released packages install and run without unintended dev or optional dependencies.

### Negative / Trade-offs
- Matrix and multi-job execution requires runner capacity on GitHub Actions.
- Adding new source code requires maintaining test coverage to remain $\ge 85\%$ overall and $\ge 90\%$ in core packages.
