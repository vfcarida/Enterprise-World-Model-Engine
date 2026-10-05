# Changelog

All notable changes to the Enterprise World Model Engine (EWM Engine) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - Unreleased

### Added
- **Graduate Adapters (Alpha → Beta) + OR Planners + Agent Evaluation (ADR-018, P05)**:
  - Formalized public `ConstraintSolver` (`check(*, state, actions, time_limit_seconds) -> SolverResult`) and `ActionPlanner` (`propose(*, state, objective, time_limit_seconds) -> Sequence[Action]`) protocols with structured `SolverResult` and `SolverStatus` in `ewm_engine.integrations.protocols`.
  - Added Architectural Decision Record `ADR-018: Solver and Planner Protocols & Resource Limits`.
  - Enforced mandatory finite time limits across all mathematical solvers and planners (default 5.0s / 5000ms), ensuring no unbounded solver execution path.
  - Implemented Google OR-Tools CP-SAT discrete allocation planner (`CPSATAllocationPlanner`) as an interventional `Actor` and `ActionPlanner` with unsat core / infeasibility certificate extraction.
  - Implemented continuous linear programming allocation planner (`SciPyAllocationPlanner`) powered by the SciPy HiGHS solver backend.
  - Hardened `Z3ConstraintAdapter` to implement `ConstraintSolver` and `Constraint`, enforcing `timeout_ms` and extracting unsatisfiable cores.
  - Hardened `ORToolsAllocationAdapter` to enforce solve time limits and implement `ConstraintSolver.check`.
  - Made `CallableActorAdapter` conform to `ActionPlanner` protocol as well as `Actor`.
  - Added agent-evaluation example in `examples/agent_evaluation/` inspired by Gaia2 / ARE (arXiv:2509.17158), proving the "engine != agent" thesis with verifiable state-based scoring and OR unsat core auditability.
  - Added optional dependency group `or = ["ortools>=9.8.0", "scipy>=1.11.0"]` and updated `solvers` extra in `pyproject.toml`.
  - Graduated all adapters to Beta maturity with explicit banners in `docs/integrations/` and automated tests in CI.
- **OpenTelemetry Observability Adapter (P04)**:
  - Added API-only OpenTelemetry adapter (`OpenTelemetryHook`) in `ewm_engine.integrations.otel`, mapping lifecycle events (`SimulationStarted`, `StepCompleted`, `ConstraintEvaluated`, `RolloutCompleted`, `SimulationFinished`) to hierarchical spans (`simulation` -> `rollout` -> `step`).
  - Implemented OpenTelemetry metric instruments mapping `RunMetrics` counters and histograms (`ewm.simulation.duration`, `ewm.rollouts.total`, `ewm.steps.total`, `ewm.constraints.evaluations`, `ewm.dynamics.transitions`, etc.).
  - Added safe default attributes preventing leakage of sensitive enterprise state, memory payloads, or PII (opt-in via `include_state_attributes=True`).
  - Added optional dependency group `otel = ["opentelemetry-api>=1.25.0"]` in `pyproject.toml`, preserving zero core coupling (library never imports SDK).
  - Added unit test suite in `tests/unit/test_otel_adapter.py` asserting span tree shapes and metric emissions via in-memory exporters.
  - Added architecture import-boundary tests guaranteeing core and simulation never import OpenTelemetry and the adapter never imports SDK.
  - Added comprehensive user guide in `docs/guides/observability-otel.md` and updated `docs/architecture/observability.md`.
- **Distributed Monte Carlo with Determinism Preserved (FEAT-002, ADR-017, P03)**:
  - Added `RolloutExecutor` protocol in `ewm_engine.simulation.executors` with three backends: `SerialExecutor` (default reference), `MultiprocessingExecutor` (multi-core process pool via stdlib `concurrent.futures`), and `RayExecutor` (cluster scale-out via Ray).
  - Guaranteed strict bitwise determinism invariant: `distributed_result.logical == serial_result.logical` across trajectory statuses, seeds, state hashes, metrics, and systemic traces.
  - Implemented deterministic per-rollout RNG spawning via `np.random.SeedSequence(scenario.seed).spawn(scenario.samples)` and deterministic result reassembly in strict ascending `sample_id` order.
  - Added optional dependency group `distributed = ["ray>=2.9.0"]` in `pyproject.toml` and included it in `all`.
  - Added comprehensive property and equivalence test suite in `tests/property/test_distributed_determinism.py`.
  - Added scale-out user guide in `docs/guides/scale-out.md`.
  - Added scale-out throughput speedup benchmarks in `benchmarks/run_benchmarks.py`.
- **Evaluation Upgrade & Bootstrap Uncertainty (FEAT-001, ADR-016, P02)**:
  - Added non-parametric percentile bootstrap confidence intervals (`compute_bootstrap_ci`, `compute_bootstrap_delta`, `BootstrapDelta`, `BootstrapConfidenceInterval`) in `ewm_engine.evaluation.uncertainty`.
  - Added empirical hypothesis testing and `is_significant` flags on counterfactual deltas with epistemic guardrails distinguishing simulation variance from real-world causality.
  - Added multi-objective Pareto analysis (`ObjectiveSpec`, `ObjectiveDirection`, `ParetoFrontier`, `compute_pareto_frontier`) in `ewm_engine.evaluation.pareto`.
  - Added epistemic anti-winner guardrail refusing automated winner declaration without user-specified preference weights.
  - Enhanced `compare_scenarios` and `ScenarioComparison` with bootstrap delta intervals, significance markers in `summary_table()`, and multi-objective Pareto frontier sections while maintaining 100% backwards compatibility (AC-024).

### Fixed
- Fixed machine-specific absolute `file:///c:/Users/...` link in `docs/stability-policy.md` to use relative repository path.

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
