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

---

## 3. Engineering Guidelines & Quality Gates

To ensure high scientific credibility, architectural integrity, and reproducibility:

### Type Checking & Code Style
- **Strict Typing**: All public code must have explicit type annotations. Do not use unconstrained `dict`, `list`, or `Any` where typed domain models (`WorldState`, `Entity`, `Resource`, `Action`, `Constraint`) apply.
- **Ruff**: Enforces formatting and linting.
  ```bash
  ruff check src tests examples
  ruff format --check src tests examples
  ```
- **Mypy**: Must pass with zero errors in strict mode:
  ```bash
  mypy src tests
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
2. Ensure all tests, lint checks, and type checks pass.
3. Submit a Pull Request targeting the `main` branch using the provided [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).

