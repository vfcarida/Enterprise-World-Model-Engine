# Enterprise World Model Engine — Spec-Driven Development Specification

> **Canonical project name:** Enterprise World Model Engine  
> **Short name:** EWM Engine  
> **Target file:** `docs/specs/spec-driven-development.md`  
> **Specification status:** Authoritative  
> **Target contract:** v1.x — v2.0-alpha  
> **Primary implementation language:** Python  
> **Repository language:** English  
> **This specification language:** English  
> **Last research review:** 2026-10-21  
> **Owner:** Vinicius Caridá (`vfcarida`)

---

## Executive Summary and Specification Authority

This document defines the authoritative development contract for the **Enterprise World Model Engine (EWM Engine)**. Its purpose is to formalize the conceptual thesis and architectural invariants into a precise engineering specification so that contributors and researchers can evolve the repository without reinterpreting its domain boundaries, epistemic guarantees, or core dependencies.

The adoption of **Spec-Driven Development** (`Specify → Plan → Tasks → Implement → Converge`) ensures that architectural intent and empirical evidence persist as verifiable artifacts across every release. In an enterprise world-modeling engine, the primary engineering risk is not coding velocity, but rather allowing successive extensions to introduce incompatible abstractions, bloat the zero-GPU core with heavy external dependencies, or conflate scientific hypotheses with verified causal or structural capabilities.

This specification establishes as a permanent architectural thesis that an **Enterprise World Model** represents an evolving socio-technical environment over which candidate interventions and policies can be simulated and stress-tested *before* they are committed in the real world. This formulation aligns with foundational world-models literature: Ha and Schmidhuber (2018) learn a compressed spatiotemporal representation of an environment within which policies can be evaluated, and DreamerV3 (Hafner et al., 2023) masters diverse domains by imagining future trajectories through a learned dynamics model. EWM Engine transposes this abstraction to organizational, operational, and socio-technical systems by combining immutable state, explicit operational constraints, controllable actions, exogenous shocks, pluggable dynamics, and calibrated uncertainty.

Crucially, EWM Engine does **not** assume that any single neural architecture (such as JEPA) or external solver (such as SMT) is mandatory for the core engine. Joint-Embedding Predictive Architectures—demonstrated in I-JEPA (Assran et al., 2023) and action-conditioned V-JEPA 2 (Assran et al., 2025)—represent a legitimate frontier for learned latent dynamics and are supported via the `DynamicsModel` protocol and experimental modules, rather than being hardcoded into the core runtime.

The foundational proposition proved by EWM Engine is:
> **An engineer or scientist can represent an operational world, declare its explicit hard and soft constraints, plug in a deterministic, stochastic, learned, or grey-box dynamics model, branch from an identical state snapshot across competing interventions, execute reproducible Monte Carlo rollouts, detect invariant violations, and trace how outcomes emerged—comparing policy alternatives with certified uncertainty without mislabeling observational simulation as an unqualified causal claim.**

The engine is strictly **industry-neutral**. Public examples and core abstractions never embed proprietary banking schemas or institution-specific financial rules; public artifacts serve logistics, supply chains, healthcare operations, energy grids, manufacturing, public-sector disaster response, and general resource allocation without domain leakage into the core kernel.

The keywords **MUST**, **MUST NOT**, **SHOULD**, and **MAY** in this document are normative (RFC 2119). Distinguishing the public stable API from experimental modules is governed by Semantic Versioning (SemVer 2.0.0).

### Order of Authority Within the Repository

| Priority | Artifact | Authority Scope |
| :--- | :--- | :--- |
| **Highest** | `docs/specs/spec-driven-development.md` | Authoritative product, architecture, and engineering contract |
| **High** | Accepted ADRs in `docs/adr/` | Architectural decisions that refine—but never contradict—this specification |
| **High** | Feature specifications in `docs/specs/features/` | Detailed normative contract for specific subsystem capabilities |
| **Medium** | Contract, property, and acceptance tests in `tests/` | Executable verification of specification compliance |
| **Medium** | Source code in `src/ewm_engine/` | Reference implementation of the contract |
| **Informative** | `README.md`, guides, handbook, and examples | User onboarding, pedagogy, and developer experience |
| **Exploratory** | `ROADMAP.md` (`Planned` items) and research notes | Future hypotheses not yet promoted to the stable contract |

Any change that alters a normative guarantee in this document requires an explicit update to this specification accompanied by an Architecture Decision Record (ADR), an API Change Proposal (ACP) when public signatures are affected, and passing automated acceptance tests.

---

## Product Vision, Scope, and Normative Contract

**Purpose.** EWM Engine is an open-source framework for representing, inducing, calibrating, and executing models of complex socio-technical environments, enabling decision-makers to ask *"what happens if we intervene with policy $A$ versus policy $B$?"* rather than merely *"what is the next point forecast?"*.

The framework enforces **Honest-by-Construction Epistemics**: an observational or conditional simulation is never automatically promoted to an interventional or counterfactual proof when unmeasured confounding, directional ambiguity, or positivity violations exist (Pearl, 2009; Schölkopf et al., 2021; Richens & Everitt, 2024). Scenario branching is the foundational primitive, while structural counterfactuals (`run_counterfactual` via Abduction–Action–Prediction) and causal induction (`ewm induce causal`) explicitly tag every edge and claim with its calibrated `EvidenceLevel` and fail closed when identifiability gates are not met.

**Target Audience.** EWM Engine is built for:
- **ML & AI Research Scientists** experimenting with mechanistic, grey-box, neural, or relational dynamics models;
- **Principal & Staff Software Engineers** building decision-support, digital twin, and policy-evaluation systems;
- **Operations Research & Decision Scientists** combining robust optimization, scenario discovery (PRIM/CART), and stochastic simulation;
- **AI Agent Developers** requiring a deterministic, constraint-enforcing environment substrate (via Python or MCP) independent of LLM weights.

**Out of Scope for the Core Kernel.** The core package (`ewm-engine` with zero optional extras) is not a conversational chatbot, RAG pipeline, LLM agent orchestrator, visual 3D digital twin renderer, or domain-specific financial application. External solvers, neural frameworks, and agent protocols integrate strictly through clean hexagonal adapters and optional dependency groups.

### Preservation of the Core Architectural Thesis

| Architectural Thesis Element | Status | Normative Engineering Decision |
| :--- | :--- | :--- |
| **EWM as the "physics engine" of the organization** | Preserved | Formalized as `WorldState` + `DynamicsModel` + `Action` + `ExogenousEvent` + `Constraint` |
| **Specialist models and agents orbit the EWM** | Preserved | `Actor`, planners, and MCP tools interact around the kernel; engine $\neq$ agent |
| **Explicit rules + learned dynamics** | Preserved | `Constraint` and `DynamicsModel` are decoupled first-class protocols, unified in Grey-Box dynamics |
| **Simulate alternatives prior to real-world actuation** | Preserved | Zero-leakage `Scenario.branch` and coupled Twin-Rollout SCM execution are first-class |
| **First-class uncertainty quantification** | Preserved | Seeded Monte Carlo rollouts, bootstrap CIs, CRPS, and TARP coverage calibration |
| **Systemic traceability** | Preserved | Cryptographic `Provenance`, `SystemicTrace` DAG, and `EvidenceLevel` ladder |
| **Unqualified causal claims from raw correlation** | Rejected | Forbidden; `InductionValidationGate` and `ResponsibleAIGate` fail closed on unverified claims |
| **JEPA as a mandatory core dependency** | Scoped to Adapter | Supported via `LearnedDynamics` / experimental latent adapters; core remains zero-PyTorch |
| **SMT/Z3 as a mandatory core dependency** | Scoped to Adapter | Core constraints run in pure Python/NumPy; `Z3SMTConstraint` is an optional `[solvers]` adapter |
| **Differentiable conservation projection** | Implemented (`Stable`) | Exact KKT orthogonal projection onto linear conservation laws (`ConservationProjectionLayer`) |
| **LLM required in the core loop** | Rejected | Core executes 100% deterministically on CPU without any LLM or network calls |

---

### Table A — Capability Maturity Matrix

| Capability | Initial Tier | Current Maturity | Governance / ADR | Normative Criterion |
| :--- | :--- | :--- | :--- | :--- |
| `World` + `WorldState` | **MUST** | **Stable** | ADR-001, ADR-009 | Deeply immutable, versioned, fingerprinted, and branchable state |
| `Entity`, `Relationship`, `Resource` | **MUST** | **Stable** | ADR-001, ADR-007 | Strongly typed, bounds-checked, and safely serializable |
| `Action` & `ExogenousEvent` | **MUST** | **Stable** | ADR-001, ADR-007 | Strict separation between controllable interventions and exogenous shocks |
| `DynamicsModel` protocol | **MUST** | **Stable** | ADR-002, ADR-007 | Pluggable transition dynamics without modifying `SimulationEngine` |
| Deterministic, Stochastic & Composite dynamics | **MUST** | **Stable** | ADR-002, ADR-005, ADR-012 | Explicit `np.random.Generator` and `SeedSequence` isolation |
| Hard & Soft `Constraint` engine | **MUST** | **Stable** | ADR-003, ADR-008 | Explicit `PRE_ACTION` rejection and `POST_TRANSITION` rollout invalidation |
| Scenario branching & Monte Carlo rollouts | **MUST** | **Stable** | ADR-005, ADR-012 | Zero shared mutable state; deterministic rollout ordering |
| `Trajectory` & `SimulationResult` | **MUST** | **Stable** | ADR-001, ADR-007 | Complete machine-readable step history and terminal status |
| `Provenance` + SHA-256 fingerprint + `EvidenceLevel` | **MUST** | **Stable** | ADR-009, ADR-010 | Canonical SHA-256 hashes and epistemic standing on all trace edges |
| `SystemicTrace` causal/dependency DAG | **MUST** | **Stable** | ADR-010 | Inspectable directed acyclic graph; forbids unqualified `"causes"` labels |
| Scenario comparison, Bootstrap CIs & Pareto frontier | **MUST** | **Stable** | ADR-016, ACP-001 | Non-parametric percentile bootstrap CIs, CRPS, CVaR, and Pareto dominance |
| Canonical JSON, Safe YAML & Draft 2020-12 Schemas | **MUST** | **Stable** | ADR-011, ADR-018 | `StrictSafeLoader` rejects custom/pickle tags; versioned schemas in `schemas/` |
| Minimal Warehouse & CivicFlow flagship fixtures | **MUST** | **Stable** | AC-012, AC-013 | Canonical normative verification fixtures tested in CI |
| Lifecycle hooks & OpenTelemetry adapter | **SHOULD** | **Stable** | ADR-013, ADR-024 | Extensible `SimulationHook` and API-only OpenTelemetry span/metric exporter |
| Unified CLI (`ewm`) | **SHOULD** | **Stable** | ADR-007, ADR-046 | `run`, `compare`, `verify`, `doe`, `calibrate`, `optimize`, `decide`, `induce`, `mcp`, `serve` |
| Distributed & Batched Monte Carlo (`executors`, `backends`) | **FUTURE** | **Stable** | ADR-017, ADR-050 | Identical fingerprints across serial, multiprocessing, Ray, Dask, and vectorized NumPy/JAX |
| OR-Tools, SciPy HiGHS, Z3 SMT & Gymnasium adapters | **FUTURE** | **Stable** | ADR-018 | Enforced wall-clock timeouts (`time_limit_seconds` / `timeout_ms`) with clean fallback |
| Learned-Dynamics Harness & Torch Neural Residual | **FUTURE** | **Stable** | ADR-019 | Multi-step divergence, symlog scaling, CRPS, and invariant violation gates |
| Scientific Benchmark Suite (5 shift families + classic twins) | **FUTURE** | **Stable** | ADR-025, ADR-052 | Reproducible benchmark protocol (`InterventionShift`, `RuleShift`, SEIR, Beer Game, WORLD3) |
| Planning & Receding-Horizon MPC Controller | **FUTURE** | **Stable** | ADR-020 | Deterministic lookahead rollouts, CVaR scorers, and audited `PlanningDecision` records |
| OOD Detection & Honest Causal Diagnostics | **FUTURE** | **Stable** | ADR-021, ADR-033 | Mahalanobis/support OOD detection, Backdoor criterion, and E-values |
| Heterogeneous Graph State, WSL v2 & WMIF v1.0 | **FUTURE** | **Stable** | ADR-022, ADR-023, ADR-051 | Lossless v1 $\leftrightarrow$ v2 migration, safe `.wsl` compiler, and WMIF v1.0 interchange |
| Twin-Rollout Counterfactual SCM Kernel | **FUTURE** | **Stable** | ADR-040, FEAT-008 | Exact Pearl Abduction–Action–Prediction with paired exogenous noise ($U$) and PoC/PN/PS |
| Decision-Focused Scenario Discovery (`ewm decide`) | **FUTURE** | **Stable** | ADR-041, FEAT-009 | PRIM & CART vulnerability box induction, 4-metric robustness (Starr, Hurwicz, Savage, Laplace), and EFE |
| Grey-Box Conservation Layer & Symbolic Flywheel | **FUTURE** | **Stable** | ADR-042, FEAT-010 | Closed-form KKT conservation projection (0.0% violation) and SINDy symbolic regression promotion |
| Amortized Calibration with Certified Coverage (`ewm calibrate`) | **FUTURE** | **Stable** | ADR-043, FEAT-011 | Neural/Mixture Density posterior estimation, TARP ECP coverage audit, and conformal recalibration |
| Metamorphic Reliability & Competence Abstention | **FUTURE** | **Stable** | ADR-044, FEAT-012 | 5 metamorphic tests, interventional competence map, and automatic abstention (`ABSTAIN_OOD_OR_INCOMPETENT`) |
| MCP Server & Agentic Governance (`ewm mcp`) | **FUTURE** | **Stable** | ADR-045 | 8 deterministic tools, 4 resources, and Concordia-style `EWMWorldArbiter` game-master bridge |
| World-Model Induction & Honesty Gate (`ewm induce`) | **FUTURE** | **Stable** | ADR-047, ADR-048, ADR-049 | Schema, OCEL 2.0, PC/FCI/DAGMA/LiNGAM causal, SINDy, AI-Poincaré invariants, and fail-closed `InductionValidationGate` |

---

### Normative Objectives

Every release of EWM Engine MUST:
1. Provide a domain-independent simulation and counterfactual reasoning kernel;
2. Maintain a documented, semantically stable public API (`ewm_engine.__all__`);
3. Enable zero-leakage scenario branching without shared mutable state;
4. Preserve bit-exact determinism under identical configuration, dependency versions, and random seed;
5. Treat explicit operational constraints (`PRE_ACTION` and `POST_TRANSITION`) as first-class citizens;
6. Record calibrated uncertainty, cryptographic provenance, and inspectable systemic traces;
7. Never emit unqualified causal claims from observational associations;
8. Execute its core engine without a GPU, LLM, cloud API, external database, or commercial solver;
9. Install cleanly as a standards-compliant Python package (`PEP 561` typed);
10. Verify all documented examples and benchmark claims through automated tests.

---

### Table B — Acceptance Criteria Mapped to Artifacts and Automated Tests

| ID | Testable Criterion | Primary Artifact | Mandatory Automated Test |
| :--- | :--- | :--- | :--- |
| **AC-001** | All Stable symbols are importable from the root public package | `src/ewm_engine/__init__.py` | `tests/contract/test_public_api.py` |
| **AC-002** | Public serializable models round-trip through JSON without semantic loss | `src/ewm_engine/serialization/` | `tests/contract/test_serialization_roundtrip.py` |
| **AC-003** | Versioned JSON Schemas match current Pydantic models | `schemas/*.schema.json` | `tests/contract/test_schema_snapshots.py` |
| **AC-004** | Identical input + versions + seed produces bit-identical trajectories | `src/ewm_engine/simulation/engine.py` | `tests/property/test_reproducibility.py` |
| **AC-005** | Branches derived from the same state snapshot never contaminate one another | `src/ewm_engine/simulation/branching.py` | `tests/property/test_branch_isolation.py` |
| **AC-006** | Hard `PRE_ACTION` constraint rejects invalid actions prior to transition | `src/ewm_engine/constraints/` | `tests/integration/test_hard_constraints.py` |
| **AC-007** | Soft constraint logs violation and permits rollout continuation | `src/ewm_engine/constraints/` | `tests/integration/test_soft_constraints.py` |
| **AC-008** | Hard `POST_TRANSITION` violation invalidates rollout without silent state projection | `src/ewm_engine/simulation/engine.py` | `tests/integration/test_post_transition_violation.py` |
| **AC-009** | Monte Carlo execution returns `samples` rollouts with isolated `SeedSequence` streams | `src/ewm_engine/simulation/engine.py` | `tests/property/test_monte_carlo.py` |
| **AC-010** | Provenance records required configuration metadata and stable SHA-256 fingerprint | `src/ewm_engine/provenance/metadata.py` | `tests/contract/test_provenance.py` |
| **AC-011** | Systemic trace records node/edge IDs, steps, edge types, and `EvidenceLevel` | `src/ewm_engine/provenance/trace.py` | `tests/contract/test_trace_contract.py` |
| **AC-012** | Minimal Warehouse fixture reproduces normative analytical outcomes | `examples/minimal_warehouse/` | `tests/examples/test_minimal_warehouse.py` |
| **AC-013** | CivicFlow flagship compares policies and enforces hard constraints | `examples/civicflow/` | `tests/examples/test_civicflow.py` |
| **AC-014** | All documented code examples execute cleanly | `README.md`, `docs/` | `tests/examples/test_documented_examples.py` |
| **AC-015** | Ruff lint, Ruff format, strict Mypy type-checking, and Pytest suite pass | `pyproject.toml`, `.github/workflows/` | CI quality gates |
| **AC-016** | Wheel and sdist build cleanly and install in an isolated environment | `pyproject.toml` | `tests/packaging/test_core_only_install.py` |
| **AC-017** | Package distributes PEP 561 typing marker (`py.typed`) | `src/ewm_engine/py.typed` | `tests/contract/test_packaging.py` |
| **AC-018** | Unsafe YAML tags and pickle payloads are rejected by `StrictSafeLoader` | `src/ewm_engine/serialization/yaml.py` | `tests/security/test_deserialization.py` |
| **AC-019** | Core package installs without PyTorch, LLM SDKs, solvers, or databases | `pyproject.toml` | `tests/contract/test_core_dependencies.py` |
| **AC-020** | Hexagonal import boundary rules between architectural layers are enforced | `src/ewm_engine/` | `tests/architecture/test_import_boundaries.py` |
| **AC-021** | README Quickstart executes end-to-end | `README.md` | `tests/examples/test_documented_examples.py` |
| **AC-022** | Documentation builds with `mkdocs build --strict` with zero broken links | `mkdocs.yml`, `docs/` | `scripts/check_docs_integrity.py` |
| **AC-023** | Repository contains zero secrets, credentials, or prompt-scaffolding leakage | Repository root | `tests/security/` |
| **AC-024** | Public API maintains backward compatibility across 1.x releases | `tests/contract/` | `tests/contract/test_api_compatibility.py` |
| **AC-025** | Distributed Monte Carlo execution matches serial execution bit-for-bit | `src/ewm_engine/simulation/executors.py` | `tests/property/test_distributed_determinism.py` |
| **AC-026** | OpenTelemetry adapter is API-only and degrades to no-op when unattached | `src/ewm_engine/integrations/otel.py` | `tests/unit/test_otel_adapter.py` |
| **AC-027** | Solver and planner adapters enforce resource/time limits with safe fallback | `src/ewm_engine/integrations/` | `tests/integration/test_solvers_and_planners.py` |
| **AC-028** | Learned-dynamics harness evaluates multi-step error, CRPS, and invariants | `src/ewm_engine/experimental/dynamics_eval.py` | `tests/unit/test_dynamics_eval.py` |
| **AC-029** | Scientific benchmark families are 100% reproducible from seed + manifest | `benchmarks/protocol.py` | `tests/benchmark/test_scientific_benchmarks.py` |
| **AC-030** | Receding-horizon (MPC) controller decisions are deterministic and audited | `src/ewm_engine/simulation/mpc.py` | `tests/unit/test_planning_scorers.py` |
| **AC-031** | OOD detector reports support regime; causal diagnostics never auto-promote claims | `src/ewm_engine/experimental/ood.py`, `causal.py` | `tests/unit/test_ood_detection.py`, `test_causal_diagnostics.py` |
| **AC-032** | WSL v2 rejects arbitrary code execution and compiles deterministically to `World` | `src/ewm_engine/serialization/wsl.py`, `wsl_parser.py` | `tests/unit/test_wsl.py`, `tests/unit/test_wsl_dsl.py` |
| **AC-033** | Twin-Rollout Counterfactual SCM executes Abduction–Action–Prediction with paired noise | `src/ewm_engine/simulation/counterfactual.py` | `tests/contract/test_counterfactual_contract.py` |
| **AC-034** | Decision-Focused Scenario Discovery induces PRIM/CART boxes and 4 robustness metrics | `src/ewm_engine/decision/` | `tests/contract/test_decision_contract.py` |
| **AC-035** | Grey-Box Conservation Layer enforces 0.0% conservation violation via KKT projection | `src/ewm_engine/experimental/conservation.py` | `tests/contract/test_grey_box_contract.py` |
| **AC-036** | Amortized Calibration certifies TARP coverage and applies conformal recalibration | `src/ewm_engine/calibration/` | `tests/contract/test_calibration_contract.py` |
| **AC-037** | Metamorphic Reliability Suite & Competence Gate abstain on OOD/incompetent regimes | `src/ewm_engine/evaluation/reliability.py` | `tests/contract/test_reliability_contract.py` |
| **AC-038** | MCP Server exposes deterministic world-model tools and Concordia arbiter bridge | `src/ewm_engine/mcp/server.py`, `integrations/concordia.py` | `tests/unit/test_mcp_server.py`, `tests/unit/test_concordia.py` |
| **AC-039** | World-Model Induction & Honesty Gate refuse unidentifiable edges and leaking invariants | `src/ewm_engine/induction/validation.py` | `tests/contract/test_induction_validation_contract.py` |
| **AC-040** | WMIF v1.0 interchange package and cross-backend execution preserve canonical hashes | `src/ewm_engine/serialization/wmif.py`, `simulation/backends.py` | `tests/unit/test_wmif_interchange.py`, `tests/property/test_cross_backend_determinism.py` |

---

## Public API, Core Models, and Serialization Contract

The root `ewm_engine` package exports the canonical stable surface (`ewm_engine.__all__`), including core state primitives, simulation and counterfactual engines, evaluation and reliability suites, calibration, decision discovery, induction, and provenance models.

### Core State and Dynamics Protocols

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class EvidenceLevel(StrEnum):
    COUNTERFACTUAL = "counterfactual"
    STRUCTURAL = "structural"
    INTERVENTIONAL = "interventional"
    QUASI_CAUSAL = "quasi_causal"
    CALIBRATED = "calibrated"
    PREDICTIVE = "predictive"
    HYPOTHETICAL = "hypothetical"
    ASSUMED = "assumed"
    ASSERTED = "asserted"
    PROVISIONAL = "provisional"
    DIRECTIONAL_UNCERTAIN = "directional_uncertain"
    UNIDENTIFIED = "unidentified"
    CONFOUNDED = "confounded"


class DynamicsModel(Protocol):
    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action],
        events: Sequence[ExogenousEvent],
        rng: np.random.Generator,
    ) -> TransitionResult: ...
```

### Constraint Evaluation Semantics

- `HARD + PRE_ACTION + violated` $\rightarrow$ the offending action MUST be rejected prior to state transition;
- `SOFT + PRE_ACTION + violated` $\rightarrow$ record the violation in the step trace and continue execution;
- `HARD + POST_TRANSITION + violated` $\rightarrow$ mark the rollout trajectory as `INVALID_HARD_CONSTRAINT` and halt that rollout;
- `SOFT + POST_TRANSITION + violated` $\rightarrow$ record the violation in the step trace and continue execution;
- The core simulation loop MUST NOT silently clip or project an invalid post-transition state unless an explicit architectural projection layer (`ConservationProjectionLayer`) is configured by the modeler.

---

## Normative Governance Templates

### ADR Template (`.github/ADR_TEMPLATE.md`)
```markdown
# ADR-XXX: <Decision Title>

Status: Proposed | Accepted | Superseded
Date: YYYY-MM-DD
Owners: Vinicius Caridá (vfcarida)
Related Spec: docs/specs/spec-driven-development.md

## Context
Describe the architectural problem, domain constraints, and why a decision is required.

## Decision
State the architectural decision and its mathematical or engineering invariants precisely.

## Consequences
### Positive
- ...
### Negative / Trade-offs
- ...

## Validation
List the contract, property, or integration tests that verify this decision.
```

### API Change Proposal Template (`.github/API_CHANGE_PROPOSAL.md`)
```markdown
# API Change Proposal: <Title>

Status: Proposed | Approved
Target Release: <version>
Owner: Vinicius Caridá (vfcarida)

## Motivation
Why is the public API change needed?

## Compatibility Classification
- [ ] Backward-compatible addition
- [ ] Deprecation (NEP-23 compliant)
- [ ] Breaking change (requires Major version increment)
```

---

## Canonical Acceptance Fixtures

### 1. Minimal Warehouse Acceptance Scenario (`AC-012`)
A deterministic two-warehouse inventory transfer scenario (`horizon=1`, `seed=42`):
- **Baseline (No transfer):**
  - `served_demand`: `20.0`, `unserved_demand`: `30.0`
  - `final_inventory`: `{"warehouse_a": 100.0, "warehouse_b": 0.0}`
  - `hard_constraint_violations`: `0`
- **Intervention (Transfer 30 units from A to B prior to demand realization):**
  - `served_demand`: `50.0`, `unserved_demand`: `0.0`
  - `final_inventory`: `{"warehouse_a": 70.0, "warehouse_b": 0.0}`
  - `hard_constraint_violations`: `0`

### 2. CivicFlow Flagship Acceptance Scenario (`AC-013`)
A multi-region flood response and humanitarian logistics simulation:
- Compares `nearest_shelter_first` against `capacity_aware_allocation` across stochastic flood severity rollouts.
- `capacity_aware_allocation` demonstrably reduces `unserved_demand` and shelter overflow while respecting all hard capacity constraints.
- Emits a complete `SystemicTrace` DAG with explicit `EvidenceLevel` annotations and zero unqualified causal claims.

---

## Post-v1 Normative Extensions (`AC-025` through `AC-040`)

### AC-025: Strict Determinism in Distributed & Batched Monte Carlo
- **Normative Statement:** Parallel, distributed (`MultiprocessingExecutor`, `RayExecutor`, `DaskExecutor`), and vectorized (`BatchedRolloutEngine`) Monte Carlo rollouts MUST produce trajectories and state fingerprints identical to serial execution under the same seed and world specification.
- **Invariant:** Each rollout $i \in \{0, \dots, N-1\}$ consumes the deterministic child stream spawned via `np.random.SeedSequence(scenario.seed).spawn(samples)[i]`.
- **Verification:** `tests/property/test_distributed_determinism.py`, `tests/property/test_cross_backend_determinism.py`.

### AC-026: API-Only, Decoupled OpenTelemetry Observability
- **Normative Statement:** `OpenTelemetryHook` attaches via `HookRegistry` using only `opentelemetry-api` without imposing telemetry SDKs on the core runtime.
- **Verification:** `tests/unit/test_otel_adapter.py`.

### AC-027: Enforced Resource & Time Limits in Solvers and Planners
- **Normative Statement:** All OR and formal verification adapters (`ORToolsAllocationPlanner`, `SciPyAllocationPlanner`, `Z3SMTConstraint`) MUST enforce explicit wall-clock timeouts (`time_limit_seconds` / `timeout_ms`) and return structured timeout results without hanging the simulation kernel.
- **Verification:** `tests/integration/test_solvers_and_planners.py`.

### AC-028: Learned-Dynamics Evaluation Harness & Invariant Gate
- **Normative Statement:** `evaluate_dynamics_model` measures 1-step error (MAE/RMSE), multi-step rollout divergence, stochastic calibration (CRPS), and physical/accounting bound violations.
- **Verification:** `tests/unit/test_dynamics_eval.py`, `tests/integration/test_learned_neural_dynamics.py`.

### AC-029: Reproducible Scientific Benchmark Protocol
- **Normative Statement:** All five synthetic structural-shift benchmark families and classic socio-technical twins (SEIR, Beer Game, WORLD3) execute deterministically from `scripts/reproduce.py` and emit cryptographically hashed manifests.
- **Verification:** `tests/benchmark/test_scientific_benchmarks.py`, `tests/integration/test_reproduce_pipeline.py`.

### AC-030: Auditable Receding-Horizon Control (MPC)
- **Normative Statement:** `RecedingHorizonController` re-grounds at every step and emits an immutable `PlanningDecision` containing candidate scores, lookahead seeds, and pre/post state fingerprints.
- **Verification:** `tests/unit/test_planning_scorers.py`, `tests/integration/test_planning_controller.py`.

### AC-031: Out-of-Distribution (OOD) Detection & Honest Causal Diagnostics
- **Normative Statement:** OOD detectors compute empirical `grounded_fraction` and annotate traces without silently altering simulation states; causal diagnostics verify the Backdoor Criterion, positivity overlap, and E-values (`compute_e_value`) without auto-promoting observational correlations.
- **Verification:** `tests/unit/test_ood_detection.py`, `tests/unit/test_causal_diagnostics.py`.

### AC-032: Safe Declarative World Specification Language (WSL v2)
- **Normative Statement:** `.wsl` and YAML/JSON world specifications compile into `World` instances without `eval`/`exec`, with a zero-dependency recursive-descent fallback when `lark` is absent and lossless v1 $\leftrightarrow$ v2 schema migration.
- **Verification:** `tests/unit/test_wsl.py`, `tests/unit/test_wsl_dsl.py`, `tests/unit/test_migration.py`.

### AC-033: Twin-Rollout Counterfactual SCM Kernel
- **Normative Statement:** `run_counterfactual` executes Pearl's three-step structural counterfactual algorithm (Abduction of exogenous noise $U$, Action via `do(X = x)` surgical intervention, and Prediction in the twin world sharing $U$), computing Individual Treatment Effects (ITE) and Probability of Causation (PN, PS, PNS).
- **Verification:** `tests/contract/test_counterfactual_contract.py`, `tests/property/test_counterfactual_properties.py`.

### AC-034: Decision-Focused Scenario Discovery & Multi-Metric Robustness
- **Normative Statement:** `DecisionEngine.decide` samples deep parameter uncertainty via Latin Hypercube Sampling (LHS), induces interpretable vulnerability boxes via PRIM or CART, and ranks policies across Starr's Domain Criterion, Hurwicz Criterion, Savage Minimax Regret, Laplace Insufficient Reason, and Active Inference Expected Free Energy (EFE).
- **Verification:** `tests/contract/test_decision_contract.py`, `tests/unit/test_decision_discovery.py`, `tests/unit/test_decision_robustness.py`.

### AC-035: Grey-Box Conservation Layer & Neuro-Symbolic Flywheel
- **Normative Statement:** `GreyBoxDynamics` composes a mechanistic `known_model` with a `residual_estimator` and projects states through `ConservationProjectionLayer` ($z^* = \hat{z} - A^T(AA^T)^{-1}(A\hat{z} - b)$), guaranteeing 0.0% conservation violation across arbitrary adversarial residuals, while `SINDyRegressor` distills interpretable rate laws from residuals.
- **Verification:** `tests/contract/test_grey_box_contract.py`, `tests/property/test_conservation_properties.py`.

### AC-036: Amortized Calibration with Certified Coverage (TARP)
- **Normative Statement:** `CalibrationEngine` infers posterior parameter distributions from observational trajectories, audits expected coverage probability via TARP (`compute_tarp_coverage`), and applies split-conformal post-hoc recalibration when uncalibrated posteriors exhibit overconfidence.
- **Verification:** `tests/contract/test_calibration_contract.py`, `tests/unit/test_amortized_calibration.py`, `tests/unit/test_tarp_coverage.py`.

### AC-037: Metamorphic Reliability & Competence-Aware Abstention
- **Normative Statement:** `evaluate_model_reliability` executes five metamorphic invariance/directional tests (`IdentityIntervention`, `MonotonicResourceScaling`, `SupplyShockDegradation`, `ActionOrderCommutativity`, `TimeHorizonAdditivity`), maps local competence across state-action space, and triggers explicit abstention (`ABSTAIN_OOD_OR_INCOMPETENT`) when policies depend on incompetent or OOD regions.
- **Verification:** `tests/contract/test_reliability_contract.py`, `tests/unit/test_metamorphic_reliability.py`, `tests/unit/test_evaluation_abstention.py`.

### AC-038: Model Context Protocol (MCP) Server & Concordia Arbiter Bridge
- **Normative Statement:** `EWMMCPServer` exposes 8 deterministic tools and 4 URI resources over JSON-RPC stdio, and `EWMWorldArbiter` mediates multi-agent natural-language or structured intents against hard world constraints before advancing state.
- **Verification:** `tests/unit/test_mcp_server.py`, `tests/unit/test_concordia.py`.

### AC-039: World-Model Induction & Fail-Closed Honesty Gate
- **Normative Statement:** `ewm induce` discovers candidate schemas, OCEL 2.0 process models, causal equivalence classes (CPDAG/PAG), SINDy rate laws, and AI-Poincaré conservation invariants, and `InductionValidationGate` (`validate_induction`) fails closed—refusing unoriented (`DIRECTIONAL_UNCERTAIN`), confounded (`CONFOUNDED`), or DoWhy-refuted edges and leaking invariants while surfacing structured `OpenQuestion` items for human modelers.
- **Verification:** `tests/contract/test_induction_validation_contract.py`, `tests/unit/test_induction_validation.py`, `tests/unit/test_causal_induction.py`.

### AC-040: World Model Interchange Format (WMIF v1.0) & Conformance Suite
- **Normative Statement:** `export_wmif_package` and `import_wmif_package` bundle `WorldSpec`, JSON Schemas, and canonical golden trajectory hashes into a self-contained `.wmif.json` artifact verified by the 3-level conformance suite (`check_wmif_conformance`).
- **Verification:** `tests/unit/test_wmif_interchange.py`.
