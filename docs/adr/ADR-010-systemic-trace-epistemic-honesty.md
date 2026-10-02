# ADR-010: Systemic Trace Epistemic Honesty and Non-Causal Defaults

## Status
Accepted

## Related Spec & Issues
- Spec: `docs/specs/spec-driven-development.md` (Systemic Trace, Epistemic Honesty, and Provenance)
- Acceptance Criteria: AC-010 (provenance and component versions), AC-011 (systemic trace contract)
- Gaps Closed: G12 (Provenance model + fingerprints), G5 (trace edge naming), G18 (causal honesty in code/trace)
- Milestone: M4

## Context
Many simulation and AI agent frameworks label directed dependency links in their execution graphs as "causal" by default (e.g. `A causes B`). In socio-technical, organizational, and complex enterprise domains, such claims are scientifically and epistemically unjustified:
1. An engine simulates computational transitions according to coded policies, physical approximations, or predictive models. Calling these links "causes" conflates simulated execution with empirical causal discovery (Judea Pearl's causal hierarchy).
2. If simulation graphs claim causality without explicit identification strategies (e.g., randomized interventions, do-calculus, or natural experiments), decision-makers are misled into treating observational or heuristic associations as verified levers.
3. Trace edges previously lacked explicit temporal binding to simulation steps, and edge endpoints were not validated against registered DAG nodes.

## Decision
1. **Systemic Trace is a Dependency and Provenance Graph, Not a Causal Graph**:
   - Trace edges capture simulation propagation mechanics (what triggered what during rollout), not real-world metaphysical causation.
   - The relation `"causes"` is strictly forbidden as a `TraceEdge` relation and raises a `ValueError` on validation.

2. **Explicit Structural Relations and Default `influences`**:
   - The default relation for `TraceEdge` is `"influences"`.
   - The simulation engine uses descriptive, structural relations:
     - `conditions`: An active intervention sets or bounds actor policy choices.
     - `drives`: An accepted action induces a state transition in dynamics.
     - `perturbs`: An exogenous shock alters or impacts dynamics.
     - `rejects`: A pre-action constraint validation blocks an action.
     - `triggers_violation`: An action directly causes a post-transition invariant violation.
     - `leads_to_violation`: Dynamics state progression breaches a constraint.
     - `mitigates`: An intervention or compensating action counteracts a deficit.

3. **Step Binding and Node Referential Integrity (AC-011)**:
   - Every `TraceEdge` includes a required `step: int` representing the discrete simulation step at which the link was created.
   - `SystemicTrace.add_edge` validates that both `source` and `target` node IDs pre-exist in `trace.nodes`. Dangling edges raise `ValueError`.

4. **Declared, Never Inferred `EvidenceLevel`**:
   - Every node and edge carries an explicit `EvidenceLevel` (`STRUCTURAL`, `INTERVENTIONAL`, `QUASI_CAUSAL`, `PREDICTIVE`, `ASSUMED`).
   - Default evidence level is `ASSUMED` when unknown. Core engine transitions declare `STRUCTURAL` (for conservation/laws) or `INTERVENTIONAL` (when conditioned on an active `Intervention`).
   - `EvidenceLevel` is epistemic metadata only; core does not compute statistical causal inference.

5. **First-Class Provenance Model (AC-010)**:
   - Introduced `Provenance` recording `scenario_fingerprint`, `initial_state_fingerprint`, `seed`, `horizon`, `samples`, `components: tuple[ComponentVersion, ...]`, `constraint_versions: tuple[ComponentVersion, ...]`, and `runtime_metadata`.
   - `SimulationResult` exposes `.provenance` as the authoritative audit object.

## Consequences
- **Positive**:
  - Epistemic rigor: clear, honest demarcation between simulation mechanics and empirical causal claims.
  - Zero dangling edges in systemic traces: complete graph referential integrity.
  - Exact auditability: every rollout embeds the cryptographic fingerprint of scenario, initial state, active dynamics, event sources, and constraints.
- **Negative / Breaking Changes**:
  - Code or tests asserting `relation="causes"` or relying on dangling edge additions must be updated to use valid nodes and structural relation names.
