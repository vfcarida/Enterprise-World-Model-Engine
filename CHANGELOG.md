# Changelog

All notable changes to the Enterprise World Model Engine (EWM Engine) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-02

Establish the official v1.0.0 Stable Contract according to the authoritative project specification (`docs/specs/spec-driven-development.md`). All 24 Acceptance Criteria (AC-001 through AC-024) and the full Definition of Done are satisfied.

### Added
- **Public API Stability & Contract Gate (AC-001, AC-024)**: Locked the 22 canonical Stable symbols in `ewm_engine.__all__` and added `tests/contract/test_api_compatibility.py` to prevent unintentional drift.
- **Phase-Aware Constraint Semantics (AC-006, AC-007, AC-008)**:
  - Added `ConstraintPhase` enum (`PRE_ACTION`, `POST_TRANSITION`) and `TrajectoryStatus` (`COMPLETED`, `INVALID`, `FAILED`).
  - Added normative semantics: `HARD + PRE_ACTION` rejects invalid actions pre-transition; `HARD + POST_TRANSITION` invalidates the rollout (`TrajectoryStatus.INVALID`) without state fabrication.
- **Deep State Immutability & Canonical Fingerprinting (AC-005, AC-010)**:
  - Deep defensive copying for `WorldState` collections and dictionaries.
  - RFC 8785 canonical JSON serializer with deterministic sorting, float normalization, and rejection of NaN/Infinity values.
  - Cryptographic SHA-256 fingerprinting on `WorldState`, `Scenario`, and `Provenance`.
- **Systemic Trace & Epistemic Honesty (AC-010, AC-011)**:
  - Explicit `EvidenceLevel` classification (`STRUCTURAL`, `INTERVENTIONAL`, `QUASI_CAUSAL`, `PREDICTIVE`, `ASSUMED`).
  - Directed dependency trace exporting to Mermaid and NetworkX without unverified `causes` labels.
- **Safe Serialization & Trusted Component Registry (AC-002, AC-003, AC-018)**:
  - Draft 2020-12 versioned JSON Schemas committed under `schemas/`.
  - Safe YAML parser rejecting `!!python/object` and unsafe tags.
  - `WorldSpec`, `ComponentSpec`, and `WorldFactory` with trusted closed registry failing closed against unregistered types.
- **Deterministic Monte Carlo & Reproducibility (AC-004, AC-009)**:
  - `numpy.random.SeedSequence(seed).spawn(samples)` deriving isolated pseudo-random streams per rollout.
  - Bitwise reproducibility invariant across runs and operating systems.
- **Normative Acceptance Scenarios (AC-012, AC-013, AC-014)**:
  - `examples/minimal_warehouse/`: Minimal normative inventory transfer fixture.
  - `examples/civicflow/`: Flagship disaster relief logistics simulation with flood hydrology and causeway closures.
- **Observability Layer & Lifecycle Hooks (M8)**:
  - `Hook` protocol and `HookRegistry` emitting typed events (`SimulationStarted`, `ConstraintEvaluated`, `StepCompleted`, `RolloutCompleted`, `SimulationFinished`).
  - Programmatic `RunMetrics` attached to `SimulationResult` and `Provenance`.
- **CI/CD Quality Gates & Supply-Chain Hardening (AC-015, AC-016, AC-017, AC-022, AC-023)**:
  - 12 visible CI gates in GitHub Actions standardized on `uv`.
  - Matrix testing on `{Ubuntu, macOS, Windows} × {Python 3.11, 3.12}`.
  - Test coverage quality gates: $\ge 85\%$ overall and $\ge 90\%$ in core packages via `scripts/check_coverage.py`.
  - 100% of third-party actions pinned to immutable commit SHAs with version comments.
  - PyPI publishing via OIDC Trusted Publishing with zero stored secrets.
  - Automated pre-merge secret scanning test (`test_no_credentials.py`).
  - Packaging contract verifying `py.typed` and clean virtualenv wheel installation smoke test.
- **Ecosystem & Machine Learning Adapters (v1.0.0+)**:
  - Added `EnterpriseGymEnv` in `ewm_engine.integrations.gym`: standard Gymnasium reinforcement learning environment wrapper with phase-aware constraint penalties.
  - Added `ORToolsAllocationAdapter` in `ewm_engine.integrations.ortools`: Google OR-Tools optimization adapter for linear programming and network flow allocation.
  - Added `SystemicTrace.to_html()` and `render_trace_html` in `ewm_engine.provenance.html_visualizer`: zero-dependency, self-contained interactive HTML/SVG graph visualizer with dark mode, node inspection, and step timeline filtering.
  - Added `WorldSimulationStateMachine` in `tests/property/test_stateful_simulation.py`: Hypothesis model-based state machine property testing verifying mass conservation and branch isolation across arbitrary interleavings.
  - Added `--html-trace` option to `ewm run` CLI command to export interactive HTML trace visualizers.
- **Documentation & Scientific Framing (AC-014, AC-021, AC-022)**:
  - Complete rewrite of `README.md` with an executable 5-minute Quickstart.
  - Conceptual docs on causality ($P(Y \mid X) \neq P(Y \mid \text{do}(X))$), uncertainty distributions, world model academic lineage, and systemic traces.
  - SemVer stability policy in `docs/stability-policy.md`.
  - v1 convergence review in `docs/specs/v1-convergence.md`.

### Changed (Pre-1.0 Breaking Refactorings)
- Reconciled `ewm_engine.__all__` to the 22 canonical Stable symbols (ADR-007). Concrete models remain importable from subpackages.
- Relocated experimental research models (`Intervention`, `LearnedDynamics`, `LinearResidualDynamics`, `RecedingHorizonSimulator`) to `ewm_engine.experimental.*`.
- Updated `Constraint.evaluate` method signature to accept `actions: Sequence[Action] = ()` and keyword-only `phase: ConstraintPhase`.
- Renamed internal `SimulationMetadata` to `Provenance`.

### Security
- Added automated adversarial deserialization test suite verifying rejection of arbitrary code execution payloads (AC-018).
- Added pre-merge credential scan asserting zero committed AWS keys, GitHub tokens, PyPI tokens, or private keys (AC-023).

---

## [0.1.0] - 2026-10-02

### Added
- Initial public alpha release of Enterprise World Model Engine.
- Basic simulation loop, prototype dynamics, constraint validation, and CLI commands.
