# SPEC 0 Support Policy

- **Status**: Active
- **Standard**: Scientific Python Ecosystem Coordination (SPEC 0)
- **Authority**: Enterprise World Model Engine Maintainers

---

## 1. Overview

The Enterprise World Model Engine (`ewm-engine`) participates in the scientific Python ecosystem and adheres to **[SPEC 0 — Minimum Supported Versions](https://scientific-python.org/specs/spec-0000/)**.

SPEC 0 defines a predictable, time-based cadence for dropping support for aging Python versions and upstream scientific dependencies, allowing the engine to leverage modern language features, security improvements, and performance optimizations without premature breaking changes.

---

## 2. Policy Windows

1. **Python Language Support (36-Month Window)**:
   - EWM Engine drops support for a Python minor version **36 months (3 years)** after its initial stable release.
   - *Example*: Python 3.10 was released in October 2021 and was retired from the support matrix in late 2024. Python 3.11 (released October 2022) is supported through late 2025/2026.
2. **Core Dependencies Support (24-Month Window)**:
   - EWM Engine drops support for core dependency releases (such as `numpy`) **24 months (2 years)** after their initial release date.
   - *Example*: NumPy 1.25 was retired; NumPy `>=1.26.0` is the active floor.

---

## 3. Current Version Support Matrix

| Technology | Minimum Floor | Supported Range | Status |
| :--- | :---: | :---: | :--- |
| **Python** | `>=3.11` | 3.11, 3.12, 3.13 | 3.10 and earlier dropped per SPEC 0 |
| **NumPy** | `>=1.26.0` | 1.26.x, 2.x | 1.25 and earlier dropped per SPEC 0 |
| **Pydantic**| `>=2.6.0` | 2.6+ | Pydantic v1 dropped |
| **PyYAML** | `>=6.0.1` | 6.0+ | Modern safe loader |

---

## 4. Automated CI Enforcement & Lower-Bound Resolution

Compliance with SPEC 0 is enforced on every commit, pull request, and scheduled resolution run via the automated packaging test suite and dependency resolution jobs:

- [`tests/packaging/test_spec0_compliance.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/tests/packaging/test_spec0_compliance.py): Inspects `pyproject.toml` to assert that `requires-python`, `numpy`, and all declared core and optional dependencies specify explicit `>=` lower bounds without regressing below mandated SPEC 0 floors.
- **Periodic Lower-Bound Validation (`--resolution=lowest-direct`)**: The [`.github/workflows/unlocked-resolution.yml`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/.github/workflows/unlocked-resolution.yml) workflow tests both the latest unpinned upstream packages (`--upgrade`) and minimum declared direct lower bounds (`uv pip install --resolution=lowest-direct`) across supported Python versions to verify that `pyproject.toml` floors remain truthful and functional.
- **Multi-OS & Interpreter Matrix**: CI tests against active supported Python versions (`3.11`, `3.12`, `3.13`) across Ubuntu, macOS, and Windows, with advisory canary builds for free-threaded Python (`3.13t`) and prerelease interpreters (`3.14-dev`).

---

## 5. Optional-Dependencies Matrix Strategy & `[all]` Semantics

To prevent undeclared import leaks while avoiding combinatorial explosion across 30+ optional extras, CI enforces a risk-based optional-dependencies matrix (`optional-extras-matrix` in `.github/workflows/ci.yml`):

1. **Minimal Core (`minimal-core`)**: Installs `ewm-engine` with zero optional extras (`pydantic`, `numpy`, `pyyaml` only) and verifies that core simulation, branching, constraints, causal evaluation, and provenance hashing execute without `ImportError`.
2. **Individual & Representative Extra Bundles**: Tests representative extra profiles across Python 3.11 and 3.12:
   - `graphs` (`networkx`)
   - `dsl` (`lark`, `pygls`)
   - `otel` (`opentelemetry-api`)
   - `cli` (`rich`)
   - `calibrate,uq,causal` (`scipy`, `scikit-learn`, `networkx`)
   - `interop,config` (`simpy`, `nashpy`, `rich`, `plotly`, `fastapi`, `httpx`, `omegaconf`)
3. **`[all]` Extra Semantics**:
   - `ewm-engine[all]` bundles all general-purpose CPU-compatible scientific, solver, ML, RL, visualization, and interop extras.
   - **Intentional Exclusions from `[all]`**:
     - `cuda` (`cupy>=13.0.0`): Requires host-specific NVIDIA CUDA toolkit/driver libraries and is non-certified for deterministic equivalence (`NON_CERTIFIED_BACKEND`).
     - `erroranalysis` (`erroranalysis>=0.5.0; python_version < '3.12'`): Restricted to Python `<3.12` and pins legacy `scipy`/`scikit-learn` constraints that conflict with modern solver and calibration stacks (explicitly declared under `[tool.uv].conflicts` in `pyproject.toml`).
     - `docs` and `dev`: Build and developer tooling extras are kept separate from runtime installation targets.

---

## 6. Flakiness & Failure Triage Policy

Deterministic scientific software requires strict separation between infrastructure noise and algorithmic non-determinism. **Global test retries (`--reruns`) are strictly prohibited in standard CI gates** so that deterministic bugs or unseeded RNG leaks are never masked.

All test failures are triaged under a three-tier classification:

1. **Tier 1 — Transient Infrastructure Failure**:
   - *Scope*: Runner network timeout during package download, GitHub Actions artifact service HTTP 5xx, or runner OOM/preemption.
   - *Action*: Re-run the failed workflow job manually or via GitHub Actions job retry; never add `@pytest.mark.flaky` to source tests for runner issues.
2. **Tier 2 — Known Timing / Resource Contention Flake (Quarantine)**:
   - *Scope*: OS scheduler jitter in wall-clock performance assertions or external subprocess startup latency on shared CI runners.
   - *Action*: Prefer converting wall-clock assertions into deterministic step/allocation counters (`memray` or instruction counts via CodSpeed). If a wall-clock check must remain, isolate it in `tests/benchmark/` with an explicit issue link and owner annotation.
3. **Tier 3 — Deterministic Regression or Seed Leak**:
   - *Scope*: Any failure in `tests/unit`, `tests/property`, `tests/numerical`, `tests/contract`, `tests/integration`, or `tests/fuzz`.
   - *Action*: Blocks merge unconditionally. For Hypothesis or fuzzing failures (`scripts/run_fuzz_smoke.py`), persist the failing seed/crash payload via `--replay-crash` or `@reproduce_failure`, add a permanent regression test, and fix the underlying violation.
