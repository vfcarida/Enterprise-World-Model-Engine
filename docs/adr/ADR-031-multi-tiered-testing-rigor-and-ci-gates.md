# ADR-031: Multi-Tiered Testing Rigor and CI Gate Separation Policy

## Status
Accepted

## Context
As the Enterprise World Model Engine reached functional completeness across Rounds 1 and 2 (`P01`–`P16`), the test suite grew to over 360 test functions across 9 categories. However, testing stochastic, numerical, and scientific world models introduces unique failure modes that traditional unit testing fails to address:
1. **Coverage vs. Fault Detection**: High line coverage does not guarantee fault detection. Without mutation testing, tests may exercise statements without actually asserting that semantic mutants are killed.
2. **Untrusted Serialization Surfaces**: YAML, JSON, and declarative WSL (World Specification Language) parsers handle external specifications and represent attack surfaces susceptible to malformed syntax, entity recursion, or unexpected type coercions.
3. **Absence of Test Oracles for Complex Simulations**: Complex multi-step world trajectories lack closed-form analytical oracles, making point assertions brittle and vulnerable to hidden systemic drift.
4. **Flakiness and Order-Dependence**: Uncontrolled random seeds, leaked network sockets, mutable system clocks, and test execution order dependencies can introduce flaky tests that erode developer trust in CI.
5. **Slow CI Pipelines**: Monolithic single-tier test suites running expensive exploratory tests block pull request feedback loops.

## Decision
We establish a **multi-tiered testing architecture** that strictly separates **deterministic, blocking CI gates** from **expensive, stochastic nightly campaigns**:

### Tier 1: Blocking CI Gates (High-Speed, Deterministic, Zero-Flake)
Run on every Pull Request and merge to `main`. Must pass 100%:
1. **Branch & Patch Coverage (`diff-cover`)**:
   - Branch coverage is activated globally (`branch = true`).
   - We replace arbitrary global percentage targets with a mandatory gate of **`diff-cover` patch coverage $\ge 90\%$ on changed lines**.
   - Global coverage ($\ge 85\%$ overall, $\ge 90\%$ on core) serves as an invariant floor.
   - Grounded in Martin Fowler and Google Testing engineering guidance: *Coverage is a useful tool for discovering untested code, not a proxy for test suite quality or fault detection.*
2. **Property & Metamorphic Testing (Hypothesis `ci` profile)**:
   - Configured with `max_examples=100`, strict deadline, and automatic example database replay (`.hypothesis/examples`).
   - Includes stateful invariant rollouts (`RuleBasedStateMachine`) asserting conservation laws, branch isolation, and fingerprint stability.
   - Metamorphic relations (MR-1 through MR-5) exploit the engine's explicit rule layer as a built-in oracle (permutation invariance, linear resource scaling, seed determinism, and slack soft-constraint invariance) with explicit `atol` thresholds.
3. **Hermetic & Deterministic Environment**:
   - Executed under `-W error` (warnings-as-errors) with an explicit, reviewed deprecation allowlist in `pyproject.toml`.
   - External network calls blocked via `pytest-socket` (`--disable-socket --allow-hosts=127.0.0.1`).
   - Frozen system clocks tested via `time-machine` and temporary filesystem isolation via `tmp_path`.
   - Order-dependence hunting via `pytest-randomly` with pinned reproducible seeds in CI.
4. **Structured Non-Numeric Snapshots (`syrupy`)**:
   - Snapshot testing non-numeric structural outputs (systemic trace Mermaid diagrams, ScenarioCard YAML/Markdown).
   - Numerical floats are strictly prohibited from raw string snapshots; canonical rounding and `assert_allclose` are required.
5. **PR Mutation Gate (Changed Functions Only)**:
   - Evaluates mutations only on modified functions/files in `git diff` via `mutmut` (target $\ge 70\%$). Full-suite mutation is never run as a PR gate.
6. **Bounded Fuzz Smoke Gate**:
   - Executes fast, bounded fuzzing (Atheris / structured mutation) on YAML, JSON, and WSL seed corpora to block newly attributable crashes.
7. **Fast Parallel Execution**:
   - Accelerated via `pytest-xdist` (`-n auto --dist worksteal`) and sharded across CI runners using `pytest-split` (`.test_durations`).

### Tier 2: Periodic & Nightly Campaigns (Exploratory, Deep, Advisory)
Run asynchronously on schedule (e.g., nightly at 03:00–04:00 UTC) or manually via `workflow_dispatch`:
1. **Deep Hypothesis Fuzzing (`nightly` profile & HypoFuzz)**:
   - Configured with `max_examples >= 1000`, `deadline=None`, and `Phase.target` guided search.
   - Discovered regression seeds are committed to `.hypothesis/examples` for deterministic replay in Tier 1 CI.
2. **Full Core Mutation Campaign (`mutmut`)**:
   - Runs full mutation sweeps across the six core subsystems (`core`, `simulation`, `constraints`, `provenance`, `durability`, `verification`).
   - Tracks mutation score trends over time (target range: 70–85%) and publishes structured `mutation_summary.json` artifacts.
3. **Continuous Fuzzing & OSS-Fuzz**:
   - Deep multi-minute fuzz campaigns across untrusted parsers using Atheris with structure-aware mutations.
4. **Order-Dependence Sweeps**:
   - Runs test suite with randomized seeds to proactively detect hidden state leakage between test modules.

## Consequences

### Positive
- **Fault Detection Over Form**: Mutation testing and metamorphic relations guarantee tests actually verify behavioral semantics rather than just executing lines.
- **Fast Developer Loops**: PRs receive rapid deterministic feedback (<2 minutes under `pytest-xdist`) without waiting for heavy mutation or fuzz sweeps.
- **Reproducible Regressions**: Failures uncovered during nightly Hypothesis or fuzzing runs automatically shrink to minimal reproducers and replay deterministically in CI via example databases.
- **Zero Flakiness Culture**: Tests cannot silently leak state, access the network, or rely on undefined execution orders.
- **No Core Dependency Overhead**: All testing tools (`mutmut`, `syrupy`, `pytest-socket`, `time-machine`, `atheris`, `diff-cover`) are strictly `dev`-only and never affect runtime users.

### Negative
- Requires maintaining seed corpora and `.test_durations` cache in version control.
- Mutation testing on Windows requires WSL or Linux CI execution due to mutmut 3.x OS limitations.
