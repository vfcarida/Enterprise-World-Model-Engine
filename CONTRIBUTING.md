# Contributing to Enterprise World Model Engine (EWM Engine)

Thank you for your interest in contributing to the **Enterprise World Model Engine (EWM Engine)**!

We are building a research-grade, production-quality open-source reference implementation for modeling, simulating, and evaluating complex organizational and socio-technical systems.

---

## 1. Code of Conduct
This project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

---

## 2. Development Setup

### Prerequisites
- Python 3.11 or higher
- Git

### Initializing Environment with uv (Recommended)
We standardize developer environments and CI on [`uv`](https://docs.astral.sh/uv/):
```bash
git clone https://github.com/vfcarida/Enterprise-World-Model-Engine.git
cd Enterprise-World-Model-Engine

# Sync virtual environment and all development dependencies reproducibly
uv sync --extra dev

# Run test suite
uv run pytest -q

# Run formatters and strict type checks
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy src tests
```

### Alternative Setup with standard pip
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows
pip install -e ".[dev]"
```

### Pre-commit Hooks
Install the automated git hooks to automatically check formatting, linting, and typing before committing:
```bash
pip install pre-commit
pre-commit install
# Run manually against all files:
pre-commit run --all-files
```

---

## 3. Engineering Guidelines & Quality Gates

To ensure high scientific credibility, architectural integrity, and reproducibility:

### Type Checking & Code Style
- **Strict Typing**: All public code must have explicit type annotations. Do not use unconstrained `dict`, `list`, or `Any` where typed domain models (`WorldState`, `Entity`, `Resource`, `Action`, `Constraint`) apply.
- **Ruff**: Enforces formatting and linting.
  ```bash
  ruff check src tests examples benchmarks
  ruff format --check src tests examples benchmarks
  ```
- **Mypy**: Must pass with zero errors in strict mode:
  ```bash
  mypy src tests examples benchmarks
  ```

### Testing
- Run the full test suite with coverage:
  ```bash
  pytest --cov=ewm_engine --cov-report=term-missing
- All new features must include unit tests and, where applicable, Hypothesis property tests (validating invariants like conservation of resources, determinism, or branch independence).
- Run performance benchmark regression tests:
  ```bash
  pytest -m benchmark
  python benchmarks/run_benchmarks.py
  ```

---

## 4. Architectural Rules

1. **Epistemic Honesty**: Never conflate observational prediction ($P(Y \mid X)$) with causal interventional simulation ($P(Y \mid \text{do}(X))$). Explicitly tag all transition models and trace edges with appropriate `EvidenceLevel`.
2. **Explicit Structure vs. Probabilistic Dynamics**: Do not encode known organizational physical limits into black-box neural weights. Model physical and policy limits as explicit `Constraint` instances.
3. **Pluggable & Decoupled**: The core engine must not depend on LLMs, GPUs, cloud services, or external databases. Heavy machine learning and solver libraries remain strictly optional.
4. **Reproducibility**: Simulations must be deterministic given the same initial state, configuration, and random seed.

---

## 5. Stability Policy & API Change Proposals

EWM Engine adheres strictly to Semantic Versioning (SemVer 2.0.0) across the `1.x` release series. See [docs/stability-policy.md](docs/stability-policy.md) for full policy details.

If you propose to:
- Add a new symbol to `ewm_engine.__all__`,
- Modify public method parameters or signatures,
- Deprecate an existing capability, or
- Modify persisted model fields or serialization schemas:

You **must** submit an **API Change Proposal (ACP)** using the template at [`.github/API_CHANGE_PROPOSAL.md`](.github/API_CHANGE_PROPOSAL.md) and receive maintainer approval before implementation. Automated contract tests (`tests/contract/test_api_compatibility.py`, AC-024) enforce that unauthorized API drift fails in CI.

---

## 6. Submitting Pull Requests

1. Fork the repository and create your feature branch: `git checkout -b feature/my-feature`.
2. Ensure all 20 CI quality gates pass (`pytest`, `ruff`, `mypy --strict`, `zizmor`, `check_action_pins`).
3. Add a Towncrier news fragment in `newsfragments/<pr>.<type>.md` (see `newsfragments/README.md`), or request the `skip-changelog` label if the PR is non-user-facing.
4. Submit a Pull Request targeting the `main` branch using the provided [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).

---

## 7. Commit Conventions & Deprecations

### Conventional Commits
We recommend following the [Conventional Commits 1.0.0](https://www.conventionalcommits.org/) convention for clean, structured git logs:
- `feat: <summary>` — Adds a new user-facing feature.
- `fix: <summary>` — Fixes a bug or unintended behavior.
- `docs: <summary>` — Documentation only changes.
- `refactor: <summary>` — Code change that neither fixes a bug nor adds a feature.
- `test: <summary>` — Adding or updating test suites.
- `chore: <summary>` — Maintenance, packaging, or CI changes.

### Deprecations & Stability (NEP-23)
Any planned deprecation must follow the [Deprecation Policy](docs/deprecation-policy.md), remaining functional across at least two minor releases (`>= 2 minor` or `>= 1 year`) and emitting `warnings.warn(..., DeprecationWarning, stacklevel=2)`. Never introduce deprecations or removals in patch releases.

