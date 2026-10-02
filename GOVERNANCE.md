# Project Governance: Enterprise World Model Engine

## Overview
The Enterprise World Model Engine (EWM Engine) is dedicated to building an open, reproducible, and scientifically grounded reference implementation for modeling enterprise and socio-technical world dynamics.

## Governance Model

The project follows a **Consensus-Seeking Maintainer Model**:
- **Maintainers**: Core contributors responsible for reviewing pull requests, managing releases, guiding the technical roadmap, and ensuring architectural coherence.
- **Contributors**: Anyone who submits issues, proposes research directions, writes code, improves documentation, or contributes benchmarks.

## Decision-Making Principles
1. **Scientific Integrity**: Hypotheses are not presented as established facts. Causal claims require rigorous structural assumptions and identification.
2. **Modularity over Monoliths**: Components must be loosely coupled with minimal required external dependencies.
3. **Architecture Decision Records (ADRs)**: Significant architectural alterations, deprecations, or core protocol changes require an ADR submitted via pull request and reviewed by maintainers before implementation.

## Branch Protection & CI Quality Gates

All changes to the codebase must pass through pull requests and satisfy strict automated quality gates before merge into `main`. Direct pushes to `main` are strictly prohibited.

### Required CI Status Checks

Pull requests must achieve green status on all 12 spec-mandated quality gates:

| Job Name | Description | Command / Tool |
| :--- | :--- | :--- |
| `lint` | Static code analysis & style | `uv run ruff check src tests examples` |
| `format-check` | Formatting conformance | `uv run ruff format --check src tests examples` |
| `type-check` | Strict type verification | `uv run mypy src tests` |
| `unit-tests` | Matrix testing across OS and Python | `uv run pytest tests/unit` (`{ubuntu, macos, windows} × {3.11, 3.12}`) |
| `property-tests` | Hypothesis generative invariant testing | `uv run pytest tests/property` |
| `integration-tests` | Multi-step simulation integration tests | `uv run pytest tests/integration` |
| `contract-tests` | Public API & schema compatibility | `uv run pytest tests/contract` |
| `security-tests` | Safe serialization & credential scan | `uv run pytest tests/security` |
| `example-tests` | Normative acceptance fixtures | `uv run pytest tests/examples` |
| `architecture-tests`| Dependency import boundary assertions | `uv run pytest tests/architecture` |
| `package-build` | Wheel build, twine check, clean venv smoke | `uv run python -m build && uv run twine check dist/*` |
| `docs-build` | Strict MkDocs documentation build | `uv run mkdocs build --strict` |
| `ci-gates` | Aggregation status check | Validates all 12 jobs succeeded |

### Coverage Enforcements
- **Overall Coverage**: $\ge 85.0\%$ enforced by `pytest-cov` / `pyproject.toml`.
- **Core Area Thresholds**: $\ge 90.0\%$ for `ewm_engine.core`, `ewm_engine.simulation`, `ewm_engine.constraints`, and `ewm_engine.provenance` enforced via `scripts/check_coverage.py`.

### Supply-Chain Security & Releases
- **Action Pinning**: All GitHub Actions workflows pin third-party actions to immutable 40-character commit SHAs with version comments.
- **CodeQL**: Automated static analysis runs on every pull request and weekly schedule.
- **Secret Scanning**: Automated pre-merge assertions verify no credentials, access tokens, or private keys exist in the repository (AC-023).
- **Trusted Publishing (OIDC)**: PyPI releases are authenticated strictly through GitHub Actions OpenID Connect (OIDC) without long-lived API tokens or stored secrets.

