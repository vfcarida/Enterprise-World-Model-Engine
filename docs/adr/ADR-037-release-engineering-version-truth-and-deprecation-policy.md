# ADR-037: Release Engineering, Version Truth & Deprecation Policy

- **Status**: Accepted
- **Date**: 2026-10-07
- **Authors**: Vinicius Caridá <vfcarida@gmail.com>
- **Deciders**: Enterprise World Model Engine Maintainers
- **Consulted**: Community Contributors, Scientific Python Working Group, Release Engineers
- **Informed**: All Downstream Users and Ecosystem Integrators

---

## 1. Context & Problem Statement

Prior to this decision, the Enterprise World Model Engine repository suffered from **version drift**:
1. **Hardcoded Stale Versioning**: `pyproject.toml` statically declared `version = "1.0.0"` and `src/ewm_engine/__init__.py` declared `__version__ = "1.0.0"`. Meanwhile, development on `main` had already shipped complete subsystems spanning Tracks T1 through T9 (v1.1 through v1.5: distributed Monte Carlo, durability/replay, trajectory verification, DoE/optimization, multi-agent coordination, and serving), alongside experimental v2.0 graph and WSL prototypes.
2. **Changelog Git Merge Contention**: Maintaining a single monolithic `CHANGELOG.md` file edited concurrently by multiple pull requests caused frequent git merge conflicts.
3. **Absence of a Formal Deprecation Policy**: While SemVer 2.0.0 was stated as policy, there was no codified deprecation lifecycle, runtime warning utility, or test enforcement, creating uncertainty for enterprise adopters regarding API longevity.
4. **Release Truth Invariants Missing**: The GitHub release workflow lacked a programmatic truth assertion verifying that built package artifacts exactly matched the triggering git release tag, allowing potential accidental publication of untagged or dirty `.dev` versions.

---

## 2. Decision Drivers

- **Truth in SemVer**: Eliminate all hardcoded version strings and synchronize metadata, documentation, and git tags with shipped engineering reality.
- **Owner-Driven Version Reconciliation**: Formally establish the true release version through maintainer consensus, balancing shipped stable capabilities with experimental namespaces.
- **Zero-Conflict Changelog Automation**: Adopt a distributed news-fragment workflow (Towncrier) with automated CI verification and label-based escape hatches.
- **NEP-23 Deprecation Discipline**: Guarantee enterprise backwards compatibility by adopting the Python scientific standard (NEP-23), requiring a minimum two-minor-release deprecation window, runtime `DeprecationWarning` emissions with `stacklevel=2`, and strict test coverage.
- **Release Verification Gate**: Ensure release pipelines abort immediately if built package versions do not match release git tags or contain development/local segments (`.dev` or `+`).

---

## 3. Decision Outcome

We decided to implement a unified release engineering, version truth, and deprecation framework:

### 3.1 Reconciled Release Truth: v1.5.0
In alignment with maintainer consultation, **`v1.5.0`** is established as the canonical release version:
- **Stable & Beta Subsystems (v1.0 – v1.5)**: Core simulation kernel, scenario evaluation, distributed Monte Carlo, OpenTelemetry observability, durability (`TraceLog`, `EventStore`, `ResultStore`), trajectory verification (Oracle DAG and STL robustness), scientific experimentation (DoE, sensitivity, optimization), platform interoperability (co-simulation, FMI, SimPy, Mesa, PySD), multi-agent coordination (Nashpy), and simulation-as-a-service (FastAPI REST).
- **Experimental Horizons (v2.0-alpha)**: Heterogeneous relational graph state models, Graph Neural Networks, and the World Specification Language (WSL) reside under `ewm_engine.experimental.*`, explicitly decoupled from `1.x` stability guarantees.

### 3.2 Dynamic VCS Version Truth Gate
- **Hatch-VCS Engine**: Version computation is strictly dynamic, derived from git tags using `local_scheme = "no-local-version"` to guarantee clean PyPI acceptance.
- **Release Truth Gate (`.github/workflows/release.yml`)**: On tag push, CI executes a mandatory verification gate:
  ```bash
  EXPECTED_VERSION="${TAG_NAME#v}"
  DETECTED_VERSION=$(uv run python -c "import ewm_engine; print(ewm_engine.__version__)")
  test "${DETECTED_VERSION}" = "${EXPECTED_VERSION}"
  ```
  Publication is blocked if versions mismatch or if `.dev` / `+local` segments are present.

### 3.3 Towncrier News-Fragment Changelog Automation
- **Newsfragments Directory (`newsfragments/`)**: Changes are submitted as isolated fragment files named `<pr>.<type>.md`, where types include `.feature`, `.bugfix`, `.doc`, `.removal`, and `.misc`.
- **Towncrier CI Gate**: Integrated as the 20th required quality gate (`towncrier-check` in `.github/workflows/ci.yml`). Pull requests are validated via `scripts/check_towncrier.py`, with an emergency escape hatch via the `skip-changelog` PR label.
- **Compilation**: At release time, maintainers run `uv run towncrier build --version X.Y.Z` to compile fragments into `CHANGELOG.md` atomically.

### 3.4 Conventional Commits & Release Please Automation
- **Conventional Commits**: Documented in `CONTRIBUTING.md` as the recommended commit standard (`feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`, `perf:`).
- **Release Please**: Configured via `.github/release-please-config.json`, `.release-please-manifest.json`, and `.github/workflows/release-please.yml` to automatically curate release pull requests and draft releases.
- **GitHub Release Notes**: Configured via `.github/release.yml` with category classifications.

### 3.5 NEP-23 Deprecation Lifecycle
Adopted and documented in `docs/deprecation-policy.md`:
1. **Window**: Minimum of **2 minor releases** or **1 full calendar year** before removal (e.g., deprecated in v1.5.0, removed no earlier than v1.7.0).
2. **Patch Invariance**: Deprecations and removals are strictly prohibited in patch releases (`1.x.Y`).
3. **Runtime Mechanics**: Implemented `ewm_engine.core.deprecation` providing:
   - `@deprecated(since="...", removed_in="...", alternative="...")`: decorator for functions and classes.
   - `deprecate(...)`: helper for conditional arguments or branches.
   - Always emits standard `DeprecationWarning` with `stacklevel=2`.
   - Appends `.. deprecated::` markers to docstrings.
4. **Test Enforcement**: Every deprecated API must be asserted via `pytest.warns(DeprecationWarning)`.

### 3.6 SPEC 0 Support Policy
- Codified in `docs/spec0-policy.md`: 36-month Python support window (Python >= 3.11) and 24-month dependency window (NumPy >= 1.26.0).
- Programmatically validated by `tests/packaging/test_spec0_compliance.py`.

---

## 4. Consequences

### Positive
- **SemVer Truth Restored**: Version numbers accurately convey ecosystem stability and subsystem maturity.
- **Elimination of Changelog Conflicts**: Towncrier news fragments allow hundreds of PRs to land concurrently without modifying `CHANGELOG.md`.
- **Enterprise Adoption Confidence**: NEP-23 deprecation guarantees protect production mission-critical deployments against unexpected breaking changes.
- **Deterministic Release Publishing**: Release pipelines guarantee that only clean, tag-identical, attested distributions reach PyPI.

### Negative / Trade-offs
- **PR Overhead**: Contributors must add a small news fragment file in `newsfragments/` for user-facing changes (mitigated by the `skip-changelog` label for chores).

---

## 5. References

- [NEP 23: Backwards Compatibility and Deprecation Policy](https://numpy.org/neps/nep-0023-backwards-compatibility.html)
- [Towncrier Documentation](https://towncrier.readthedocs.io/)
- [Release Please: Automated Releases](https://github.com/googleapis/release-please)
- [Semantic Versioning 2.0.0](https://semver.org/)
- [SPEC 0: Minimum Supported Versions](https://scientific-python.org/specs/spec-0000/)
