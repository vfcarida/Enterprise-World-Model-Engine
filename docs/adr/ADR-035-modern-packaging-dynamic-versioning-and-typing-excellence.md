# ADR-035: Modern Packaging, Dynamic VCS Versioning with Hatchling, and Static/Runtime Typing Excellence

## Status
Accepted

## Context
Prior to Round 8 (R08), the Enterprise World Model Engine (`ewm-engine`) relied on a legacy `setuptools` build backend with a hard-coded static version string (`version = "1.0.0"`) in `pyproject.toml`. This static configuration introduced severe architectural risks:
1. **Version Drift Hazard**: Releasing new versions or maintenance patches risked version divergence between git release tags, `pyproject.toml`, documentation, and runtime `__version__`.
2. **Legacy Packaging Standards**: License declaration predated **PEP 639** (SPDX license expression specification), relying on ambiguous metadata formats.
3. **Lockfile Misinterpretation**: While `uv.lock` provided lightning-fast reproducible development environments, treating a lockfile as authoritative for downstream consumers risks hiding subtle incompatibilities across declared dependency ranges (`>=1.26`, `>=2.6`).
4. **Extras Inconsistency**: The project defined 35+ granular feature extras (`[solvers]`, `[ml]`, `[causal]`, `[interop]`), but the umbrella `all` extra was incomplete and lacked automated verification that core runs cleanly without any optional dependencies.
5. **Static Typing & Runtime Discipline**: While strict Mypy was enforced, `strict_equality` was disabled (leaving boolean operations like `ndarray == None` uncaught), and no packaging test guaranteed that the PEP 561 `py.typed` marker was present inside compiled wheel distributions. Furthermore, numerical simulation hot paths required strict protection against per-element runtime validation overhead.

## Decision

### 1. Build Backend & Dynamic Versioning (Hatchling + Hatch-VCS)
- We migrate the build backend from `setuptools` to **`hatchling`** with **`hatch-vcs`** (`hatchling.build`).
- **Dynamic Versioning from Git Tags**:
  - The authoritative version is computed dynamically from git tags (`vX.Y.Z`).
  - We configure `local_scheme = "no-local-version"` to omit local git commit hash suffixes (e.g. `+g1a2b3c`), as public PyPI indexes strictly reject local version segments.
  - `fallback_version = "1.0.0"` is declared to ensure unversioned source checkouts (e.g., zip downloads without git history) build deterministically.
  - `hatch-vcs` generates an untracked `src/ewm_engine/_version.py` during builds, which is bundled inside release wheels and queried by `ewm_engine.__version__`.
- Hardcoded version strings are completely eliminated from `pyproject.toml` and source code.

### 2. PEP 639 License Standard & Metadata Completeness
- Package licensing is migrated to the modern **PEP 639** standard:
  ```toml
  license = "Apache-2.0"
  license-files = ["LICENSE"]
  ```
- Deprecated `License :: OSI Approved :: Apache Software License` Trove classifiers are removed in favor of the unambiguous SPDX expression.

### 3. uv Resolution Architecture: Dev Lockfile vs. Downstream Range Freedom
- **`uv.lock` as Dev-Only Reproducibility Spine**: The lockfile is committed to ensure deterministic CI runs, benchmarks, and developer onboarding. However, we explicitly establish that downstreams resolve against declared range boundaries in `pyproject.toml` (`pydantic>=2.6.0`, `numpy>=1.26.0`, `pyyaml>=6.0`).
- **Scheduled Unlocked Range Resolution**: A scheduled CI workflow (`.github/workflows/unlocked-resolution.yml`) runs weekly without `uv.lock`, executing `uv pip install --system --upgrade ".[all,dev]"` to test the test suite against the latest published upstream versions, catching breaking range incompatibilities proactively.
- **PEP 751 Tracking**: We track the emerging **PEP 751** (`pylock.toml`) standard as a future vendor-neutral portable lockfile format.

### 4. Extras Hygiene & Core-Only Import Isolation
- **Complete `all` Extra**: The `all` extra is audited and synchronized to represent the true complete superset of all optional feature dependencies across graphics, solvers, machine learning, experimentation, and interoperability.
- **Lazy Import Isolation**: Optional libraries (`torch`, `z3`, `ortools`, `ray`, `gymnasium`, `fastapi`, `plotly`, `mlflow`) must **never** be imported at package root or in core simulation modules.
- **Packaging Test Gate**: Automated tests in `tests/packaging/test_core_only_import.py` verify that:
  1. `import ewm_engine` loads zero optional dependencies into `sys.modules`.
  2. The Five-Minute Quickstart workflow (state, constraints, dynamics, simulation, branching, scenario comparison) executes successfully using only core dependencies.

### 5. Static Typing Hardening & Speed-Canary (Mypy Strict + Pyrefly)
- **Mypy Strict Blocking Gate**:
  - Pinned in dev environment.
  - Enabled flags: `strict = true`, `warn_unused_ignores = true`, `strict_equality = true`, `warn_return_any = true`, `disallow_untyped_defs = true`.
  - 100% pass across all 273 source files.
- **`Any`-Expression Leakage Budget**:
  - Evaluated via `mypy src --any-exprs-report`.
  - Current measured type coverage is **95.12%** (only 4.88% Any expressions across 39,413 AST expressions).
  - We establish a permanent leakage budget: precision coverage must remain $\ge 92\%$ ($\le 8\%$ Any expressions).
- **Pyrefly Non-Blocking Speed-Canary**:
  - We incorporate Meta's `pyrefly` type checker in CI as an advisory speed-canary (`continue-on-error: true`).
- **PEP 561 Wheel Verification**:
  - Automated test `tests/packaging/test_wheel_contents.py` unpacks built wheels and asserts that `ewm_engine/py.typed` is physically present.

### 6. Runtime Validation Discipline in Numeric Hot Paths
- **Zero Per-Element Validation in Hot Paths**:
  - Pydantic models and procedural type inspections are strictly restricted to initialization and configuration phases.
  - Per-element validation inside multi-step Monte Carlo rollout loops is strictly prohibited.
- **Vectorized Boundary Validation**:
  - Public numeric boundaries utilize `validate_numeric_array_boundary` (`src/ewm_engine/core/boundary.py`) to validate `ndarray` type, expected dimensionality, finiteness, and dtype in a single vectorized NumPy pass.
- **Runtime Type Checking in Tests**:
  - We integrate `beartype` on public boundary methods and enable `typeguard` during test execution to catch typing regressions without runtime overhead in production.

## Consequences

### Positive
- **Zero Version Drift**: Creating a git release tag (`git tag v1.1.0`) automatically updates the wheel version, PyPI metadata, and `ewm_engine.__version__`.
- **Packaging Integrity**: Every built wheel is guaranteed to include `py.typed` and dynamic version information, verified by CI.
- **Downstream Protection**: Unlocked range resolution tests guarantee that dependency constraints reflect real compatibility.
- **Extreme Type Safety**: 100% strict Mypy pass with `strict_equality` prevents subtle comparison bugs while `beartype`/boundary validation protects entrypoints.

### Negative / Trade-offs
- Developers building from source without git history must specify `SETUPTOOLS_SCM_PRETEND_VERSION` or rely on `fallback_version = "1.0.0"`.
