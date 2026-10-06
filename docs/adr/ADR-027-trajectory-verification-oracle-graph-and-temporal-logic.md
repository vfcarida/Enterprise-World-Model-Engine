# ADR-027: Trajectory Verification — Oracle-Graph Verifier and Temporal Logic over Traces

## Status
Accepted

## Date
2026-10-05

## Context
In version 1.0–1.2, EWM Engine enforced per-step constraints (`HARD` and `SOFT` evaluated at `PRE_ACTION` and `POST_TRANSITION` phases) and generated detailed `EvidenceLevel` systemic traces. However, it lacked the formal capability to express and verify **temporal and causal properties across entire multi-step trajectories**.

Users and decision evaluators could not express canonical systemic requirements such as:
- *"Whenever an exogenous demand shock occurs, service level agreements must recover within $k$ steps."*
- *"Shelter occupancy must never exceed nominal capacity for more than 2 consecutive steps."*
- *"A dispatch command must precede delivery, and delivery must arrive within an expected tolerance window."*

Ad-hoc validation approaches suffer from significant drawbacks:
1. Static assertion scripts are fragile and do not track formal violation provenance.
2. LLM-as-a-judge evaluators exhibit poor reproducibility; Meta ARE's trajectory verification research (arXiv:2509.17158) demonstrated that structured oracle-graph verification achieves 0.98 inter-annotator agreement versus only 0.72 for LLM evaluators.
3. Complex formal tools (e.g. MoonLight JVM or ANTLR-dependent full STL) impose heavy runtime burdens that violate the engine's zero-dependency core principle.

## Decision

### 1. Dedicated Verification Namespace (`ewm_engine.verification`)
All verification abstractions and implementations are organized under `ewm_engine.verification`. The root `ewm_engine.__all__` surface remains strictly locked to the v1.0.0 Stable contract (`LOCKED_V1_STABLE_SURFACE`), ensuring zero breaking changes.

### 2. Oracle-Graph Verifier (Core, stdlib/Pydantic)
We port the Meta ARE verification pattern into `ewm_engine.verification.oracle`:
- `OracleNode`: Represents an expected event with category, optional label pattern, expected parameter dictionary, and optional step range.
- `OracleEdge`: Represents a directed causal or temporal dependency with minimum/maximum discrete step delays ($\Delta k \in [k_{\min}, k_{\max}]$) and continuous time delays ($\Delta t \in [t_{\min}, t_{\max}]$).
- `OracleGraph`: Directed acyclic graph of expected nodes and edges.
- `evaluate_oracle_graph(graph, trace)`: Evaluates the oracle DAG against a `SystemicTrace` on three formal axes:
  1. **Consistency:** Injective matching of oracle nodes to trace nodes based on exact parameter keys, categories, and labels.
  2. **Causality:** Enforces parent-before-child ordering for all directed edges. Topologically independent branches may freely interleave in simulation time without generating violations.
  3. **Timing:** Enforces discrete step and continuous timestamp bounds across linked events.
- **Strict Non-Mutation:** Verification functions strictly read traces and return structured `VerificationViolation` audit records; state mutation and automated repair are strictly prohibited.

### 3. Property-Spec DSL and Provenance Fingerprint Folding
We define `PropertySpec` locked at `schema_version = "1.0.0"` in `ewm_engine.verification.spec`:
- Every property has a canonical SHA-256 `property_hash` computed over its normalized schema representation.
- `fold_properties_into_fingerprint(base_fingerprint, properties)` deterministically sorts and hashes property specifications into simulation/result fingerprints. This guarantees that "which properties were verified" is an immutable, auditable part of the simulation's provenance trail.

### 4. Bounded-Future Discrete STL Monitor (Core, Pure-NumPy)
We implement a zero-dependency discrete bounded STL/MTL monitor in `ewm_engine.verification.temporal`:
- Supports atomic predicates (`PredicateFormula`), Boolean combinators (`Not`, `And`, `Or`, `Implies`), and bounded temporal operators:
  - Bounded Always: $\square_{[k_1, k_2]} \phi$
  - Bounded Eventually: $\lozenge_{[k_1, k_2]} \phi$
  - Bounded Until: $\phi \ \mathcal{U}_{[k_1, k_2]} \ \psi$
- Returns `STLVerdict` containing Boolean satisfaction verdict, quantitative robustness margin ($\rho$), and step-by-step robustness traces.
- Convenience builders: `predicate()`, `always()`, `eventually()`, `until()`, `implies()`.

### 5. Full STL/MTL via RTAMT (`[stl]` Optional Extra)
For continuous and unconstrained temporal logic specifications, we provide `RTAMTEvaluationBackend` in `ewm_engine.verification.rtamt_adapter`:
- Isolated strictly behind the `[stl]` extra (`rtamt`). When `rtamt` is absent, the core engine and tests run cleanly without import errors.
- Evaluates full STL strings and computes quantitative robustness margins.
- Cross-validated against the core bounded monitor on the overlapping fragment ($\square$, $\lozenge$), asserting bitwise and quantitative agreement.
- Aggregates empirical `RobustnessDistribution` across Monte Carlo rollouts (mean, std, min, max, p05-p95 quantiles, CVaR 5%, satisfaction probability).

### 6. External Verification Stubs
Documented stubs are provided for external engines without introducing dependencies:
- `MoonLightSTRELAdapter`: Directs users to the JVM runtime for spatio-temporal reach-and-escape logic.
- `LLMSoftCheckAdapter`: Adheres to the core zero-dependency policy by using a host-provided async callable `(prompt, **kwargs) -> str` without importing any LLM SDK into core.

### 7. Epistemic Guardrails
- A satisfied temporal or oracle property is a formal property of the *simulated trajectory*, not an ungrounded real-world guarantee.
- Relations named `"causes"` remain strictly forbidden in trace and oracle edges.

## Consequences
- **Positive:** Trajectory-level requirements can be formally specified, monitored, and audited across simulation runs.
- **Positive:** Core verifier and bounded STL monitor have zero heavy dependencies (pure Python, NumPy, Pydantic).
- **Positive:** Quantitative robustness distributions across Monte Carlo rollouts allow measuring *how safely* a policy satisfies constraints, not just binary pass/fail.
- **Positive:** Property hashes fold directly into cryptographic provenance fingerprints.
- **Positive:** Full compatibility with Meta ARE's 3-axis verification methodology.
- **Negative:** Full continuous STL monitoring requires the optional `rtamt` dependency (capped at Python $\le 3.12$ due to ANTLR runtime constraints).
