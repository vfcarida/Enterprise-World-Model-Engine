# ADR-053: Formal Enterprise World-Model Semantics, First-Class Runtime Context, and Ontic/Epistemic Separation

- **Status**: Accepted
- **Date**: 2026-10-21
- **Author**: Vinicius Caridá
- **Related Specifications**: [`docs/specs/spec-driven-development.md`](../specs/spec-driven-development.md), [`rfcs/RFC-0002-context-aware-hybrid-world-model-semantics.md`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/rfcs/RFC-0002-context-aware-hybrid-world-model-semantics.md), [`ADR-002`](ADR-002-pluggable-dynamics.md), [`ADR-008`](ADR-008-constraint-phase-semantics-and-rollout-invalidation.md), [`ADR-009`](ADR-009-state-immutability-and-canonical-fingerprint.md), [`ADR-010`](ADR-010-systemic-trace-epistemic-honesty.md), [`ADR-042`](ADR-042-grey-box-conservation-flywheel.md)

---

## 1. Context and Problem Statement

Classical world-model formulations in reinforcement learning and robotics—such as Ha & Schmidhuber (*World Models*, 2018), Hafner et al. (*DreamerV3*, 2023), LeCun (*A Path Towards Autonomous Machine Intelligence*, 2022), and Assran et al. (*V-JEPA 2*, 2025)—model an environment primarily through a stationary Markovian transition kernel $S_{t+1} \sim T(S_t, A_t)$.

When world models are transposed to **enterprise socio-technical and organizational systems** (supply networks, hospital operations, service workflows, infrastructure grids), four structural mismatches emerge:

1. **Context-Conditioned Non-Stationarity ($K_t$)**: Enterprise transition laws depend heavily on deployment, tenant, and runtime configuration $K_t$ (active workflow definitions, automation rules, policy versions, feature flags, service-level contracts, network topology versions, and effective dates). Representing $K_t$ as an untyped dictionary inside `WorldState` conflates mutable ontic state with versioned deployment context and prevents modular invariance (Schölkopf et al., 2021) where unrelated context changes must not perturb unrelated mechanisms.
2. **Ontic vs. Epistemic Conflation Risk ($S_t$ vs. $P_t$)**: What is true inside the modeled enterprise world at step $t$ (**ontic state** $S_t$) is fundamentally distinct from what the modeling system knows, infers, assumes, or has evidence for (**epistemic state** $P_t$). Embedding provenance or systemic traces inside $S_t$ would break Markovian state hashing (two identical states reached via different paths would produce different state hashes) and risk leaking privileged audit metadata to partially observed actors.
3. **Partial Observability ($O_t, B_t$)**: Operational actors rarely observe the entire enterprise state $S_t$. Passing the full `WorldState` object to every `Actor.act` call creates a risk of accidental API leakage of hidden entities, resources, or rules.
4. **Silent Repair in Hybrid/Learned Dynamics**: When combining symbolic structural laws with learned residual models ($\Delta S = f_{\text{known}}(\dots) + \Delta_{\text{learned}}(\dots)$), silently clamping out-of-bounds resource predictions masks model incompetence and violates [`ADR-008`](ADR-008-constraint-phase-semantics-and-rollout-invalidation.md).

---

## 2. Decision

### 2.1 Formal 10-Tuple Definition of an Enterprise World Model

We formalize an Enterprise World Model as the 10-tuple:

$$\mathcal{W} = \langle S, A, E, K, O, T, C, U, Q, P \rangle$$

where:
- **$S$ (`WorldState`)**: Ontic enterprise world state $S_t = (G_t, R_t, M_t, \Gamma_t, C_t)$ at step $t$ and continuous timestamp $\tau_t$.
- **$A$ (`Action`, `Intervention`, `ScheduledAction`)**: Controllable endogenous policy actions and surgical $do(X=x)$ interventions.
- **$E$ (`ExogenousEvent`, `ExogenousEventSource`)**: Seeded exogenous shocks and environmental perturbations.
- **$K$ (`ContextSnapshot`)**: Typed, versioned, provenance-aware deployment/tenant/runtime context conditioning transitions:
  $$S_{t+1} \sim T(S_t, A_t, E_t \mid K_t)$$
- **$O$ (`ObservationSpec`, `ObservationFilter`, `BeliefState`)**: Observation operator $O_t = h(S_t, K_t, \xi_t)$ projecting $S_t$ into a leak-free observable state for actors, with optional belief state $B_t$.
- **$T$ (`DynamicsModel`, `MechanismProvenance`, `MechanismKind`)**: Modular transition mechanisms classified into `KNOWN_DETERMINISTIC`, `KNOWN_STOCHASTIC`, `LEARNED`, `DISCOVERED_AT_RUNTIME`, and `HYBRID`.
- **$C$ (`Constraint`, `ConstraintRegistry`, `LinearConservationLaw`, `InfeasibleResidualPolicy`)**: Pre-action (`PRE_ACTION`), post-transition (`POST_TRANSITION`), and manifold conservation invariants with explicit infeasibility handling (`PROJECT`, `REJECT`, `ABSTAIN`, `INVALIDATE`).
- **$U$ (`SeedSequence`, `ExogenousNoiseRecord`, `BootstrapConfidenceInterval`, `CoverageCertificate`)**: Aleatoric simulation stochasticity ($P_{\text{model}}$) and epistemic uncertainty quantification.
- **$Q$ (`ObjectiveDirection`, `ObjectiveVector`, `RobustnessCriterion`, `ParetoFrontier`)**: Multi-objective utility, loss, CVaR tail risk, and minimax regret functionals.
- **$P$ (`EpistemicState`, `Provenance`, `SystemicTrace`, `EvidenceLevel`)**: Epistemic state and cryptographic provenance tracking what the engine knows, assumes, or has evidence for.

### 2.2 Ontic vs. Epistemic State Separation

We formally separate **Ontic State ($S_t$)** from **Epistemic State ($P_t$)**:
- `WorldState` ($S_t$) remains strictly **ontic**: it encodes only entities, relationships, resources, operational memory lags, active rule thresholds, environmental context variables, step, and timestamp. Its canonical SHA-256 `state_hash` / `fingerprint` / `ontic_fingerprint` depends solely on ontic state, and `WorldState.to_ontic_state()` strips any `_epistemic_*` scratch annotations.
- `EpistemicState` ($P_t$, defined in `ewm_engine.core.epistemic`) is introduced as an explicit immutable model capturing `trace` (`SystemicTrace`), `effective_evidence_level`, `mechanism_provenances`, `context_versions`, `context_fingerprints`, `assumptions`, `drift_signals`, and `epistemic_fingerprint`.
- `Trajectory` exposes `.epistemic_state` as a computed property without altering the locked v1 JSON Schema of `WorldState` or `Trajectory`.

### 2.3 Typed, Versioned, Scoped Runtime Context ($K_t$)

We introduce `ContextSnapshot` in `ewm_engine.core.context`:
- **Strictly Typed**: Replaces untyped configuration dictionaries with frozen Pydantic models (`WorkflowConfig`, `AutomationRuleConfig`, `ServiceEndpointConfig`, `ContractTermConfig`, `ContextSnapshot`).
- **Immutable Evolution**: `ContextSnapshot.evolve(...)` produces a new versioned snapshot without mutating prior snapshots.
- **Mechanism-Scoped Projection (Independent Causal Mechanisms)**: Every mechanism may declare `required_context_keys: tuple[str, ...]`. Before evaluating a mechanism, `ContextSnapshot.project(required_context_keys)` extracts only the declared context slice, and `ContextSnapshot.domain_fingerprint(required_context_keys)` hashes only that domain. Consequently, mutating unrelated workflows, feature flags, or tenant settings in $K_t$ leaves unrelated mechanisms bit-for-bit invariant.

### 2.4 Partial Observability Without Breaking the Simple Path

We introduce `ObservationSpec`, `ObservationFilter`, and `BeliefState` in `ewm_engine.core.observation`:
- When `observation_spec` is configured on `World` (globally or per `actor_id`), `execute_step` projects $S_t$ through `ObservationFilter.observe(state, actor_id)` before calling `actor.act(observed_state, actor_context)`.
- The projected `WorldState` is a fresh deep-copied instance containing only permitted entities, resources, relationships, memory keys, rule keys, and context keys, with **zero back-references** to the hidden `WorldState`.
- When `observation_spec` is `None` (the default), the simple full-observability deterministic path executes unchanged with zero overhead.

### 2.5 Mechanism Taxonomy and Constraint-Preserving Hybrid Dynamics

- **Mechanism Taxonomy**: `MechanismKind` (`KNOWN_DETERMINISTIC`, `KNOWN_STOCHASTIC`, `LEARNED`, `DISCOVERED_AT_RUNTIME`, `HYBRID`) and `MechanismProvenance` codify each mechanism's identifier, version, validity domain (`ValidityDomain`), required context keys, fail-closed `EvidenceLevel`, uncertainty contract, affected invariants, and failure policy (`InfeasibleResidualPolicy`).
- **No Silent Repair in Hybrid/Learned Dynamics**: `GreyBoxDynamics` (aliased as `HybridDynamics`) and `LinearResidualDynamics` no longer silently clamp out-of-bounds resource deltas. When a learned residual proposes an infeasible transition (violating resource bounds, conservation laws, or hard constraints), it is governed by `InfeasibleResidualPolicy`:
  1. `PROJECT`: Explicitly projects the residual via `ConservationProjectionLayer` and/or box-feasible bounds projection and records a `ResidualFeasibilityRecord` with `raw_deltas`, `applied_deltas`, `l2_projection_distance`, and `violations`.
  2. `REJECT`: Rejects the infeasible residual and preserves the structural transition $f_{\text{known}}(S_t, A_t, E_t \mid K_t)$.
  3. `ABSTAIN`: Refuses state evolution for the step (returning $S_t$ unchanged) and flags `action_taken="abstained"` and `diagnostics["abstained"] = True`.
  4. `INVALIDATE`: Marks `action_taken="invalidated"` and emits a hard post-transition violation (`hybrid_residual_feasibility`) that terminates the trajectory with `TrajectoryStatus.INVALID`.

### 2.6 Mechanism/Configuration Drift and Vendor-Neutral Runtime Discovery

- **Drift Taxonomy (`ewm_engine.core.drift`)**: `DriftKind` distinguishes `COVARIATE_SHIFT`, `PARAMETER_DRIFT`, `STRUCTURAL_MECHANISM_CHANGE`, `RULE_CONSTRAINT_CHANGE`, `TOPOLOGY_CHANGE`, and `OBJECTIVE_CHANGE`, evaluated via `detect_world_drift(...)`.
- **Vendor-Neutral Discovery (`ContextDiscoveryProvider`)**: Defines a pure-Python protocol and `AuditMetadataContextDiscovery` reference implementation that parses vendor-neutral runtime/audit metadata into a typed `ContextSnapshot` without coupling the core to any proprietary ERP/CRM/ITSM vendor.

---

## 3. Backward-Compatibility Analysis

1. **Root Stable API (`ewm_engine.__all__`)**: All 28 locked v1 symbols in `ewm_engine.__all__` remain unchanged (`tests/contract/test_public_api.py` and `tests/contract/test_api_compatibility.py` pass without modification).
2. **Existing 4-Argument `DynamicsModel.transition` Signature**: Preserved. `execute_step` inspects whether a dynamics model accepts `runtime_context` and passes it only when supported.
3. **JSON Schemas (`schemas/*.schema.json`)**: All 14 locked v1 schemas remain at `schema_version = "1.0.0"` with zero breaking changes.
4. **Golden Conformance Fingerprints**: When `runtime_context=None` and `observation_spec=None`, `WorldState.fingerprint` and `Trajectory` hashes are 100% bit-identical to v1.5.0.

---

## 4. References

- Ha, D., & Schmidhuber, J. (2018). *World Models*. NeurIPS 2018 (`arXiv:1803.10122`).
- LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering Diverse Domains through World Models (DreamerV3)*. (`arXiv:2301.04104`).
- Assran, M., et al. (2025). *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*. (`arXiv:2506.09985`).
- Schölkopf, B., et al. (2021). *Towards Causal Representation Learning*. Proceedings of the IEEE, 109(5), 612–634 (`arXiv:2102.11107`).
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
- Richens, J., & Everitt, T. (2024). *Robust Agents Learn Causal World Models*. ICLR 2024 (`arXiv:2402.10877`).
- Enterprise workflow & hybrid world-model benchmarking literature (*World of Workflows*; *CascadeBench: Do Enterprise Systems Need Learned World Models?*; *Business World Model*).
