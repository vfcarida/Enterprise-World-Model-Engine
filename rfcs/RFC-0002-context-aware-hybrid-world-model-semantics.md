# RFC-0002: Context-Aware Hybrid Enterprise World-Model Semantics

- **RFC Number:** 0002
- **Status:** Implemented
- **Author(s):** Vinicius Caridá (@vfcarida)
- **Created Date:** 2026-04-12
- **Target Release:** v1.5.0
- **Steering Council Sponsor:** Vinicius Caridá (@vfcarida)
- **Related ADRs:** [ADR-053](../docs/adr/ADR-053-formal-enterprise-world-model-semantics-and-context.md), [ADR-039](../docs/adr/ADR-039-responsible-ai-trust-and-anti-overclaiming-enforcement.md), [ADR-042](../docs/adr/ADR-042-grey-box-conservation-flywheel.md)
- **Related Issues / PRs:** N/A

---

## 1. Executive Summary

This RFC specifies the formal mathematical and programmatic evolution of the Enterprise World Model Engine from a structural simulator with optional learned residuals into a **context-aware hybrid enterprise world model** governed by the 10-tuple $\mathcal{W} = \langle S, A, E, K, O, T, C, U, Q, P \rangle$. It introduces first-class, typed, versioned, provenance-aware runtime context ($K_t$ via `ContextSnapshot`), separates ontic state ($S_t$, `WorldState`) from epistemic state ($P_t$, `EpistemicState`) without breaking v1 schemas, provides optional partial-observability filtering ($O_t, B_t$ via `ObservationSpec`, `ObservationFilter`, `BeliefState`), establishes an explicit mechanism taxonomy (`MechanismKind` + `MechanismProvenance`) with fail-closed causal evidence rules, replaces silent clamping on infeasible learned residuals with explicit feasibility policies (`InfeasibleResidualPolicy`: `PROJECT`, `REJECT`, `ABSTAIN`, `INVALIDATE`), defines a 6-category drift taxonomy (`DriftKind` + `DriftSignal`), and provides a vendor-neutral runtime context discovery protocol (`ContextDiscoveryProvider`).

---

## 2. Motivation & Problem Statement

Enterprise socio-technical systems differ fundamentally from stationary physical or arcade environments studied in classical world models (Ha & Schmidhuber, 2018; Hafner et al., 2025; Assran et al., 2025):

1. **Non-Stationary Deployment & Policy Context ($K_t$):** Enterprise transition laws depend on tenant configurations, workflow definitions, automation rules, SLA contracts, feature flags, service topology, and effective dates (`S_{t+1} ~ T(S_t, A_t, E_t | K_t)`). Previously, runtime context had no typed, versioned, cryptographic representation, nor a mechanism to isolate unrelated mechanisms from unrelated context changes.
2. **Ontic vs. Epistemic Entanglement:** Embedding provenance (`SystemicTrace`) inside `WorldState` conflates what is true in the modeled world (*ontic state*) with what the engine knows, assumes, or has causal identification evidence for (*epistemic state*).
3. **Partial Observability Without POMDP Tax:** Actors frequently operate under partial visibility (redacted resources, hidden relationships, unobserved secondary rules), yet simple deterministic rollouts must remain lightweight without forcing every user into a POMDP belief state.
4. **Silent Clamping in Hybrid Dynamics:** Previously, `GreyBoxDynamics` and `LinearResidualDynamics` invoked `update_resource(..., clamp=True)` when applying learned residuals without a conservation projector. If a learned residual proposed a resource delta violating physical or capacity bounds, `clamp=True` silently repaired the violation without recording projection, rejection, abstention, or invalidation.
5. **Lack of Explicit Mechanism Taxonomy & Drift Semantics:** Transition mechanisms lacked standardized metadata declaring their mechanism kind (`KNOWN_DETERMINISTIC`, `KNOWN_STOCHASTIC`, `LEARNED`, `DISCOVERED_AT_RUNTIME`, `HYBRID`), required context keys, validity domains, and fail-closed causal evidence ceilings, as well as structured drift detection across covariate, parameter, structural, rule, topology, and objective shifts.

---

## 3. Detailed Specification & Design

### 3.1 Formal 10-Tuple

$$\mathcal{W} = \langle S, A, E, K, O, T, C, U, Q, P \rangle$$

- $S$: Ontic state space (`WorldState` containing `entities`, `resources`, `relationships`, `memory`, `active_rules`, `context`).
- $A$: Endogenous interventions (`Action`).
- $E$: Exogenous disturbances (`Event`).
- $K$: Typed, versioned, immutable deployment/runtime context (`ContextSnapshot`).
- $O$: Observation space and partial observability operator (`ObservationSpec`, `ObservationFilter`, `BeliefState`).
- $T$: Context-conditioned transition mechanism $S_{t+1} \sim T(S_t, A_t, E_t \mid \Pi_{\text{req}(T)}(K_t))$ (`DynamicsModel`, `ContextualDynamicsModel`, `GreyBoxDynamics`).
- $C$: Structural, accounting, policy, and conservation invariants (`Constraint`, `ConservationLayer`).
- $U$: Epistemic and aleatoric uncertainty semantics (`TransitionResult.uncertainty`, `UncertaintyDecomposition`).
- $Q$: Multi-objective evaluation functionals (`ObjectiveVector`, `TrajectoryScorecard`).
- $P$: Epistemic state and provenance (`EpistemicState`, `MechanismProvenance`, `EvidenceLevel`, `SystemicTrace`).

### 3.2 Context Conditioning & Mechanism-Scoped Projection

`ContextSnapshot` (`ewm_engine.core.context`) is a frozen Pydantic model with:
- `context_id: str`, `version: str`, `tenant_id: str`, `environment_regime: str`
- `effective_from: datetime | None`, `effective_until: datetime | None`
- Typed maps: `workflow_definitions: dict[str, WorkflowConfig]`, `enabled_automations: dict[str, AutomationRuleConfig]`, `policy_versions: dict[str, str]`, `feature_flags: dict[str, bool]`, `topology_config: dict[str, str]`, `external_services: dict[str, ServiceEndpointConfig]`, `active_contracts: dict[str, ContractTermConfig]`, `parameters: dict[str, float | int | str | bool]`
- `provenance_source: str`, `evidence_level: EvidenceLevel`, `parent_version: str | None`

Each mechanism declares `required_context_keys: tuple[str, ...]` in its `MechanismProvenance`. Before evaluating $T$, the engine projects $K_t$ via `K_t.project(required_context_keys)`, guaranteeing that mutations to unreferenced context fields have zero effect on the projected fingerprint or transition dynamics.

### 3.3 Constraint-Preserving Hybrid Dynamics

Given structural proposal $\hat{S}_{t+1}^{\text{struct}} = f_{\text{struct}}(S_t, A_t, E_t \mid K_t)$ and learned residual $\Delta_{\theta}(S_t, A_t, E_t \mid K_t)$, `GreyBoxDynamics` (aliased as `HybridDynamics`) checks feasibility against conservation laws, resource bounds $[0, \text{max\_capacity}]$, and explicit constraints $C$. When $\hat{S}_{t+1}^{\text{struct}} \oplus \Delta_{\theta}$ is infeasible, silent clamping is prohibited; the engine executes the configured `InfeasibleResidualPolicy`:

- `PROJECT`: Projects onto the feasible set via `ConservationLayer` and/or explicit box-constraint projection, recording `ResidualFeasibilityRecord(is_feasible=False, action_taken="projected", l2_projection_distance=..., violations=...)`.
- `REJECT`: Rejects the infeasible residual component and falls back to $\hat{S}_{t+1}^{\text{struct}}$ with `action_taken="rejected"`.
- `ABSTAIN`: Returns `S_t.clone()` with zero uncertainty reduction and `action_taken="abstained"`.
- `INVALIDATE`: Marks `action_taken="invalidated"`, triggering a hard `POST_TRANSITION` constraint violation in `SimulationEngine` (`TrajectoryStatus.INVALID`).

---

## 4. Backwards Compatibility & SemVer Impact

- **Zero Breaking Changes to v1 Stable Root Surface:** All 28 symbols in `ewm_engine.__all__` (`LOCKED_V1_STABLE_SURFACE`) remain untouched.
- **Zero Breaking Changes to Locked JSON Schemas:** `WorldState`, `Trajectory`, and `TransitionResult` retain their exact JSON schema structures (`schemas/*.schema.json`). New metadata (`mechanism_provenance`, `context_metadata`, `residual_feasibility`) is stored in `TransitionResult.diagnostics` with typed `@property` accessors, and `Trajectory.epistemic_state` is exposed as a computed `@property`.
- **Simple Deterministic Path Preserved:** `World(state, dynamics, constraints)` without `runtime_context` or `observation_spec` executes identically to prior versions.

---

## 5. Security, Invariants & Supply Chain

- Introduces **zero** external dependencies (uses only Python stdlib, `pydantic`, and `numpy`).
- Enforces deep isolation in `ObservationFilter.observe(state, actor_id)` so partially observed actors cannot traverse references back to hidden resources, entities, or relationships.
- Preserves fail-closed `EvidenceLevel` semantics: `LEARNED` and `DISCOVERED_AT_RUNTIME` mechanisms cannot claim `INTERVENTIONAL` or `COUNTERFACTUAL` evidence without an explicit `identification_certificate_id`.

---

## 6. Alternatives Considered

1. **Removing `trace` from `WorldState` immediately:** Rejected for v0.1.x/v1.x because `schemas/world-state.schema.json` and existing callers rely on `state.trace`. Providing `EpistemicState.from_world_state(...)` and `Trajectory.epistemic_state` achieves clean ontic/epistemic separation (`ontic_fingerprint` vs. `epistemic_fingerprint`) with 100% backward compatibility.
2. **Allowing untyped `dict[str, Any]` for $K_t$:** Rejected because untyped dicts lack schema validation, domain-scoped projection, and deterministic canonical hashing.

---

## 7. Implementation Plan & Test Strategy

- **Core Modules:** `ewm_engine/core/context.py`, `ewm_engine/core/observation.py`, `ewm_engine/core/drift.py`, `ewm_engine/core/epistemic.py`.
- **Dynamics & Hybrid Extensions:** `ewm_engine/dynamics/base.py`, `ewm_engine/dynamics/deterministic.py`, `ewm_engine/dynamics/stochastic.py`, `ewm_engine/dynamics/learned.py`, `ewm_engine/experimental/grey_box.py`.
- **Simulation & World Integration:** `ewm_engine/core/world.py`, `ewm_engine/simulation/rollout.py`, `ewm_engine/simulation/trajectory.py`.
- **Verification Suite:** `tests/contract/test_world_model_semantics_p02.py` verifying all 8 required formal properties.

---

## 8. Unresolved Questions

- None for v1.5.0; physical removal of `WorldState.trace` is deferred to the v2.0 schema cycle.
