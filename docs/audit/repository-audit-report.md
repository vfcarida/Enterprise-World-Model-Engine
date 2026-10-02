# Enterprise World Model Engine — Comprehensive Repository Audit Report

**Date:** 2026-10-02  
**Target Release Series:** v1.0.x / v1.1.0  
**Audit Scope:** Core architecture, developer ergonomics, CLI, solver integrations, benchmark automation, static typing, and roadmap alignment.  
**Auditor:** Principal Software Architect & Senior Maintainer  

---

## Executive Summary

The Enterprise World Model Engine (EWM Engine) has successfully achieved its v1.0.0 milestone, establishing domain-independent simulation kernel invariants, deep state immutability, first-class phase-aware constraints, reproducible Monte Carlo rollouts, safe versioned schemas, complete provenance tracking, and a 12-gate CI/CD pipeline.

This audit evaluates the codebase following the v1.0.0 release to identify operational technical debt, ergonomics gaps, developer workflow frictions, and near-term capabilities roadmapped for the v1.x series. Six actionable findings (`AUDIT-001` through `AUDIT-006`) have been identified and prioritized.

---

## Audit Backlog & Execution Queue

| Finding ID | Title | Priority | Area | Status |
| :--- | :--- | :---: | :--- | :---: |
| **AUDIT-001** | Expand CLI Capabilities for Declarative Spec Validation & Execution | **P1** | CLI / Ergonomics | READY |
| **AUDIT-002** | Add Missing `solvers` Extra Group in `pyproject.toml` for Z3 SMT Adapter | **P2** | Packaging / Dependencies | READY |
| **AUDIT-003** | Automated Performance Benchmark Regression Guard via Pytest Marker | **P2** | Testing / Benchmarks | READY |
| **AUDIT-004** | Enforce Strict Static Analysis (`mypy`, `ruff`) on `benchmarks/` in CI | **P2** | Quality / CI | READY |
| **AUDIT-005** | Align `ROADMAP.md` and Architecture Navigation to v1.0.0 Stable Contract | **P3** | Documentation / Roadmap | READY |
| **AUDIT-006** | Pre-Commit Hook Configuration for Developer Onboarding | **P3** | Developer Experience | READY |

---

## Detailed Audit Findings

### AUDIT-001: Expand CLI Capabilities for Declarative Spec Validation & Execution
- **Finding ID:** `AUDIT-001`
- **Title:** Expand CLI Capabilities for Declarative Spec Validation & Execution
- **Priority:** P1
- **Problem:** The `ewm` CLI entrypoint currently only supports running hardcoded demonstration scripts (`ewm example [minimal|civicflow]`). It lacks subcommands for users to validate their own declarative `WorldSpec` YAML/JSON files, run simulations from specification files, or inspect committed schemas.
- **Evidence:** `src/ewm_engine/cli/main.py` only implements an `example` subcommand. `WorldFactory.create_from_yaml` and `WorldSpec.from_yaml` exist in core, but are inaccessible via the command line.
- **Impact:** High user friction. Decision engineers and modelers working in YAML must write custom Python runner scripts even for basic schema checks or single scenario rollouts.
- **Recommended Solution:**
  1. Add `ewm validate <spec_file>`: Safely loads YAML/JSON `WorldSpec` and reports validity without executing code.
  2. Add `ewm run <spec_file> [--horizon H] [--samples N] [--seed S] [--out results.json]`: Compiles `WorldSpec` into a `World`, executes simulation, and outputs summary metrics / writes JSON results.
  3. Add `ewm schema [list|show <name>]`: Displays or exports committed JSON schemas.
- **Acceptance Criteria:**
  - `ewm validate <path>` outputs success summary for valid specs and exits with code 1 and error details for invalid specs.
  - `ewm run <path>` executes simulation and prints outcome metrics.
  - Unit tests in `tests/unit/test_cli.py` cover all new subcommands.
- **Testing Requirements:** Add unit tests for `validate`, `run`, and `schema` subcommands with valid, invalid, and missing file inputs.
- **Documentation Requirements:** Update `docs/getting-started.md` and CLI documentation with usage examples.
- **Dependencies:** None.
- **Risks:** CLI must preserve zero-arbitrary-code-execution security guarantees by relying strictly on `WorldFactory` and `WorldSpec`.

---

### AUDIT-002: Add Missing `solvers` Extra Group in `pyproject.toml` for Z3 SMT Adapter
- **Finding ID:** `AUDIT-002`
- **Title:** Add Missing `solvers` Extra Group in `pyproject.toml` for Z3 SMT Adapter
- **Priority:** P2
- **Problem:** `src/ewm_engine/integrations/solvers.py` informs users: `"z3-solver is not installed. Install via pip install z3-solver or pip install ewm-engine[solvers]"`. However, `[project.optional-dependencies]` in `pyproject.toml` does not define a `solvers` extra group.
- **Evidence:** `src/ewm_engine/integrations/solvers.py:73-75` raises `SimulationConfigurationError` referencing `ewm-engine[solvers]`. In `pyproject.toml`, optional dependencies only list `graphs`, `ml`, `docs`, `dev`, and `all`.
- **Impact:** User confusion when attempting to follow official error instructions to install SMT solver support.
- **Recommended Solution:**
  1. Add `solvers = ["z3-solver>=4.12.0"]` to `[project.optional-dependencies]` in `pyproject.toml`.
  2. Update documentation in `src/ewm_engine/integrations/README.md` and `docs/api/reference.md`.
  3. Add unit test asserting `SimulationConfigurationError` message accuracy when z3 is uninstalled.
- **Acceptance Criteria:**
  - `pyproject.toml` includes `solvers` extra.
  - `uv sync --extra solvers` / `pip install -e .[solvers]` installs cleanly.
- **Testing Requirements:** Test in `tests/unit/test_integrations.py` validating adapter behavior with and without z3.
- **Documentation Requirements:** Document `ewm-engine[solvers]` installation in `README.md` and `integrations/README.md`.
- **Dependencies:** None.
- **Risks:** Low; strictly optional extra.

---

### AUDIT-003: Automated Performance Benchmark Regression Guard via Pytest Marker
- **Finding ID:** `AUDIT-003`
- **Title:** Automated Performance Benchmark Regression Guard via Pytest Marker
- **Priority:** P2
- **Problem:** `benchmarks/run_benchmarks.py` exists as a standalone script, but there is no automated regression test tagged with the `@pytest.mark.benchmark` marker defined in `pyproject.toml`. Performance regressions in simulation rollout throughput are not automatically detected in local or CI test runs.
- **Evidence:** `pyproject.toml` declares `benchmark: performance benchmarks` under `markers`, but `tests/` contains zero tests utilizing this marker.
- **Impact:** Simulation throughput (steps/sec) could regress unnoticed during refactoring or model updates.
- **Recommended Solution:**
  1. Add `tests/benchmark/test_throughput_benchmark.py` tagged with `@pytest.mark.benchmark`.
  2. Benchmark a standard 30-node topology for 20 rollouts, asserting throughput exceeds a conservative regression threshold ($\ge 100$ steps/sec) and completes without error.
  3. Verify via `uv run pytest -m benchmark`.
- **Acceptance Criteria:**
  - `pytest -m benchmark` discovers and runs the throughput test.
  - Test asserts deterministic rollout count and throughput threshold.
- **Testing Requirements:** `tests/benchmark/test_throughput_benchmark.py` passing cleanly.
- **Documentation Requirements:** Document running benchmarks in `CONTRIBUTING.md`.
- **Dependencies:** None.
- **Risks:** Performance thresholds must be set conservatively to avoid flakiness across varied developer machines and CI virtual machines.

---

### AUDIT-004: Enforce Strict Static Analysis (`mypy`, `ruff`) on `benchmarks/` in CI
- **Finding ID:** `AUDIT-004`
- **Title:** Enforce Strict Static Analysis (`mypy`, `ruff`) on `benchmarks/` in CI
- **Priority:** P2
- **Problem:** CI workflows run `ruff check src tests examples`, `ruff format --check src tests examples`, and `mypy src tests`, omitting the `benchmarks/` directory from automated quality gates.
- **Evidence:** `.github/workflows/ci.yml` lines 22, 61, and 84 exclude `benchmarks/`.
- **Impact:** Untyped code or lint violations can accumulate silently in `benchmarks/`.
- **Recommended Solution:**
  1. Add full type annotations to `benchmarks/run_benchmarks.py`.
  2. Update `.github/workflows/ci.yml` and local scripts to check `src tests examples benchmarks`.
  3. Verify `uv run mypy src tests examples benchmarks` and `uv run ruff check src tests examples benchmarks`.
- **Acceptance Criteria:**
  - `mypy src tests examples benchmarks` passes with 0 errors in strict mode.
  - `ruff check src tests examples benchmarks` and `ruff format --check` pass with 0 errors.
- **Testing Requirements:** CI job execution passes with all four directories included.
- **Documentation Requirements:** Update `CONTRIBUTING.md` commands to include `benchmarks`.
- **Dependencies:** None.
- **Risks:** None.

---

### AUDIT-005: Align `ROADMAP.md` and Architecture Navigation to v1.0.0 Stable Contract
- **Finding ID:** `AUDIT-005`
- **Title:** Align `ROADMAP.md` and Architecture Navigation to v1.0.0 Stable Contract
- **Priority:** P3
- **Problem:** `ROADMAP.md` still describes the project status as `"v0.1.0 - Foundation (Alpha)"`, listing items like Declarative World Specification, SMT Solvers, and MPC as "Near-Term Roadmap (v0.2.0 - v0.4.0)", even though they have been implemented for v1.0.0. Furthermore, `docs/architecture/observability.md` and `docs/examples/minimal-warehouse.md` should be fully cross-linked.
- **Evidence:** `ROADMAP.md` line 19: `## Current Status (v0.1.0 - Foundation)`.
- **Impact:** Misleading project maturity cues for external evaluators and open-source contributors.
- **Recommended Solution:**
  1. Update `ROADMAP.md` to reflect `v1.0.0 - Stable Contract` established.
  2. Classify core capabilities as **Stable** and experimental models as **Experimental**.
  3. Archive completed near-term items as implemented in v1.0.0 and focus the roadmap on v1.1.0+ (distributed rollouts, differentiable constraints, causal discovery).
- **Acceptance Criteria:**
  - `ROADMAP.md` accurately reflects `1.0.0`.
  - `mkdocs build --strict` completes with 0 warnings.
- **Testing Requirements:** Strict docs build passes.
- **Documentation Requirements:** `ROADMAP.md` and `docs/index.md` updated.
- **Dependencies:** None.
- **Risks:** None.

---

### AUDIT-006: Pre-Commit Hook Configuration for Developer Onboarding
- **Finding ID:** `AUDIT-006`
- **Title:** Pre-Commit Hook Configuration for Developer Onboarding
- **Priority:** P3
- **Problem:** External contributors lack a standardized pre-commit configuration (`.pre-commit-config.yaml`) to automatically format, lint, and validate code before committing, leading to preventable CI test round-trips.
- **Evidence:** No `.pre-commit-config.yaml` exists in the repository root.
- **Impact:** Slower PR velocity and friction for new open-source contributors.
- **Recommended Solution:**
  1. Create `.pre-commit-config.yaml` specifying `ruff`, `ruff-format`, `check-yaml`, `check-json`, and `mypy` hooks pinned to stable versions.
  2. Document `pre-commit install` workflow in `CONTRIBUTING.md`.
- **Acceptance Criteria:**
  - `.pre-commit-config.yaml` exists, is syntactically valid, and passes `pre-commit run --all-files` (if installed) or manual check.
  - `CONTRIBUTING.md` explains setup.
- **Testing Requirements:** Validate YAML schema and ensure clean hook execution.
- **Documentation Requirements:** Document in `CONTRIBUTING.md`.
- **Dependencies:** None.
- **Risks:** None.
