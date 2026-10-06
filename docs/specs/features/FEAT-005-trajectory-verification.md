# FEAT-005: Trajectory Verification — Oracle-Graph and Temporal Logic over Traces

- **Status:** Approved / Implemented
- **Horizon:** v1.3.0 "Trajectory Verification"
- **Authors:** Verification Engineering & Formal Methods Team
- **Governance:** `01_EXPANDED_ROADMAP.md` (Track T3), ADR-027, ACP-006
- **Dependencies:** Core engine (Zero new dependencies: stdlib, NumPy, Pydantic); Optional extra `[stl]` (`rtamt`)

---

## 1. Motivation & Context

In high-stakes enterprise and socio-technical simulation, evaluating simulated executions against single-step static invariants is insufficient:
- Enterprise decision-makers and regulators require guarantees over **entire trajectory lifecycles** and multi-step dynamics.
- Concrete real-world requirements include temporal recovery bounds (*"whenever demand spikes, SLA recovery must occur within $k$ steps"*), capacity invariants (*"shelter occupancy must never exceed nominal capacity for $> 2$ consecutive steps"*), and causal sequencing (*"evacuation alert must strictly precede dispatch, with delivery arriving within 2 to 6 hours"*).
- Traditional evaluation relies either on static per-step constraint assertions or on unstructured LLM-as-a-judge prompts. Recent empirical research from Meta ARE (arXiv:2509.17158) demonstrated that structured **oracle-graph verifiers** achieve 0.98 inter-annotator agreement versus only 0.72 for LLM-only evaluators by verifying exact structured state transitions rather than ambiguous natural language prose.

This feature introduces a comprehensive trajectory verification subsystem into `ewm_engine.verification` that operates across two complementary formalisms:
1. **Oracle-Graph Verifier (Core, stdlib/Pydantic):** A directed acyclic graph (DAG) of expected systemic events evaluated across three formal axes:
   - **Consistency:** Hard parameter predicates, categorical matching, and label criteria.
   - **Causality:** Topological parent-before-child ordering; independent branches in the DAG may interleave arbitrarily.
   - **Timing:** Tolerance delay windows ($\Delta k \in [k_{\min}, k_{\max}]$, $\Delta t \in [t_{\min}, t_{\max}]$) between linked events.
2. **Temporal-Logic Monitoring:**
   - **Core Bounded-Future STL (Core, pure-NumPy):** Zero-dependency discrete monitor over `WorldState` timeseries and `Trajectory` signals evaluating atomic predicates, Boolean operators, and bounded temporal operators ($\square_{[k_1, k_2]}$, $\lozenge_{[k_1, k_2]}$, $\mathcal{U}_{[k_1, k_2]}$), returning exact Boolean verdicts and quantitative robustness margins.
   - **Full STL/MTL via RTAMT (`[stl]` Extra):** Full Signal Temporal Logic evaluation via RTAMT, computing robustness margins and aggregating empirical `RobustnessDistribution` statistics across Monte Carlo rollouts (mean, std, min, max, p05-p95 quantiles, CVaR 5%, satisfaction probability).
3. **Auditable Property-Spec DSL & Fingerprint Provenance:** A declarative Pydantic DSL where every property has a deterministic canonical SHA-256 hash that folds into scenario and result provenance fingerprints, guaranteeing verification auditability.

---

## 2. Requirements & Acceptance Criteria

### Part A: Oracle-Graph Verifier (Track T3, Core)
- **AC-041 (Oracle-Graph Verification):** Implement `evaluate_oracle_graph(graph, trace)` evaluating an `OracleGraph` over a `SystemicTrace`.
  - Enforce Consistency (exact detail matching and node categorization).
  - Enforce Causality (directed edge $(u, v)$ requires step/time of $u$ to precede or equal $v$; independent branches may interleave without violation).
  - Enforce Timing (discrete step delay $\Delta k \in [k_{\min}, k_{\max}]$ and continuous timestamp delay $\Delta t \in [t_{\min}, t_{\max}]$).
  - Guarantee non-mutation: verification strictly reads traces and trajectories without modifying underlying states or steps.

- **AC-042 (Property-Spec DSL & Provenance Folding):** Implement `PropertySpec` with canonical SHA-256 `property_hash`.
  - Provide `fold_properties_into_fingerprint(base_fp, properties)` that is deterministic and permutation-invariant.
  - Export versioned Draft 2020-12 JSON Schema (`schemas/property-spec.schema.json`).

### Part B: Temporal-Logic Monitoring (Track T3, Core + Extra)
- **AC-043 (Core Bounded-Future Discrete STL):** Implement pure-NumPy zero-dependency discrete bounded STL monitor.
  - Support `PredicateFormula`, `NotFormula`, `AndFormula`, `OrFormula`, `ImpliesFormula`, `AlwaysFormula` ($\square_{[k_1, k_2]}$), `EventuallyFormula` ($\lozenge_{[k_1, k_2]}$), and `UntilFormula` ($\mathcal{U}_{[k_1, k_2]}$).
  - Return `STLVerdict` reporting Boolean satisfaction, quantitative robustness margin, step-by-step traces, and violation steps.
  - Implement `extract_trajectory_signals` extracting resources, state variables, and step metrics into aligned NumPy float vectors.

- **AC-044 (RTAMT Full STL Integration & Robustness Distribution):** Implement `RTAMTEvaluationBackend` isolated strictly behind the optional `[stl]` extra.
  - Cross-check core STL monitor against RTAMT on overlapping fragment ($\square$, $\lozenge$), asserting agreement on Boolean verdicts and quantitative robustness margins.
  - Implement `evaluate_monte_carlo` returning `RobustnessDistribution` with mean, std, min, max, quantiles (p05, p25, p50, p75, p95), CVaR 5%, and satisfaction probability.
  - Guarantee clean isolation: when `rtamt` is absent, core imports and tests succeed without error.

- **AC-045 (External Verification Stubs):** Provide documented, isolated adapter stubs:
  - `MoonLightSTRELAdapter` for topological spatio-temporal reach-and-escape logic (isolated from JVM).
  - `LLMSoftCheckAdapter` utilizing host-provided async callable `(prompt, **kwargs) -> str` without embedding any LLM SDK into core.

---

## 3. Epistemic Constraints & Non-Goals

1. **No Causal Overclaims:** Temporal and oracle-graph satisfaction is a formal property of the *simulated trajectory*, not an ungrounded real-world guarantee. Edges with relation `"causes"` remain strictly forbidden.
2. **Zero In-Place Repair:** Verification functions strictly read traces and return structured violation audit logs; they never mutate world states or attempt automated trajectory repair.
3. **Strict Zero-Dependency Core:** The core verifier and restricted STL monitor rely exclusively on Python standard library, NumPy, and Pydantic. Heavy formal tools (RTAMT, ANTLR, MoonLight JVM) and LLM SDKs are strictly quarantined behind extras or host-provided callbacks.

---

## 4. Verification & Testing Evidence

1. **Unit Tests (`tests/unit/`):**
   - `test_oracle_graph_verifier.py`: Consistency, causality, and timing violations, as well as simulated warehouse scenario verification.
   - `test_core_stl_monitor.py`: Atomic predicates, logical combinators, bounded temporal operators, and signal extraction from trajectories.
2. **Integration Tests (`tests/integration/`):**
   - `test_rtamt_stl_verification.py`: Cross-checks core monitor against RTAMT on overlapping formulas; evaluates Monte Carlo robustness distributions across 20+ rollouts; asserts error handling on missing signals and empty inputs.
3. **Property & Invariant Tests (`tests/property/`):**
   - `test_verification_properties.py`: Verifies property hash determinism, permutation-invariance in fingerprint folding, trace and trajectory non-mutation, and topological branch independence.
4. **Contract Tests (`tests/contract/`):**
   - `test_core_dependencies.py`: Confirms `ewm_engine.verification` imports zero optional heavy dependencies in core mode.
   - `test_api_compatibility.py`: Confirms root `ewm_engine.__all__` remains locked to the v1.0.0 stable surface.
