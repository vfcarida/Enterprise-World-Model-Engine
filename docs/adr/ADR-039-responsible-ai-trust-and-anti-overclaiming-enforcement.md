# ADR-039: Responsible AI, Trust, and Anti-Overclaiming Enforcement

<!-- This ADR is written according to the MADR 4.x standard -->

* Status: accepted
* Deciders: Vinicius Caridá (@vfcarida)
* Date: 2026-10-07
* Technical Story: Responsible AI, Trust & Anti-Overclaiming Enforcement (R12)
* Supersedes: None

## Context and Problem Statement

Enterprise world models simulate complex organizational, financial, and socio-technical dynamics to inform high-stakes executive decisions. In these operating regimes, **epistemic overclaiming presents a severe operational and ethical risk**:
1. Point estimates without uncertainty intervals create a false illusion of numerical precision.
2. Observational associations are easily mistaken for causal proofs, leading to counterfactual interventions that fail or backfire in practice.
3. Neural or statistical dynamics models silently extrapolate into ungrounded state spaces when driven outside empirical support.
4. Evolving regulatory standards—notably NIST AI RMF 1.0, ISO/IEC 42001 (AIMS), and the EU AI Act (Regulation (EU) 2024/1689 Annex IV)—require structured risk management, documentation, and technical transparency.

How should the Enterprise World Model Engine enforce scientific humility and trustworthy AI practices across its architecture and public API boundaries?

## Decision Drivers

* **Structural Humility Over Aspiration**: Epistemic guardrails must be programmatic invariants enforced at the API boundary, not optional guidelines or passive comments.
* **No Unqualified Causal Escalation**: Adhere strictly to Judea Pearl's do-calculus and Donald Rubin's Potential Outcomes: observational correlations must never be promoted to `INTERVENTIONAL` without formal mathematical identifiability and covariate positivity.
* **Mandatory Uncertainty Representation**: Public forecast results must always ship with confidence/credible intervals, dispersion, and support regime flags.
* **International Framework Alignment**: Position documentation, cards, and manifests directly against NIST AI RMF, ISO/IEC 42001, and EU AI Act Annex IV technical documentation requirements without claiming legal compliance.
* **Domain-Neutral Core Rigor**: Keep core dependencies restricted to NumPy and Pydantic; wire specialized responsible-AI libraries (Fairlearn, DiCE, MS RAI ErrorAnalysis) as optional lazy-loaded adapters.

## Considered Options

* **Option 1: Aspirational Documentation & Passive Logging** — Rely on docstrings, tutorial warnings, and standard Python log messages. High risk of callers ignoring intervals, stripping uncertainty, or presenting correlation as causal truth.
* **Option 2: Heavy RAI Monolith** — Enforce fairness and explainability by making Fairlearn, DiCE, and InterpretML mandatory core dependencies. Bloats installation size, introduces cross-platform dependency conflicts, and forces domain-specific fairness criteria onto domain-neutral simulations.
* **Option 3: API-Boundary Enforcement, Mandatory Uncertainty Coupling, and Pluggable Extras (Chosen)** — Structurally guard causal claims and forecast extraction; emit structured MANAGE warnings; provide `ResponsibleAIGate` hooks; schema-validate Mitchell model cards with auto-population; provide EU AI Act Annex IV readiness renderers; and expose RAI toolkits via optional extras (`[fairness]`, `[explain]`, `[erroranalysis]`).

## Decision Outcome

Chosen option: **Option 3: API-Boundary Enforcement, Mandatory Uncertainty Coupling, and Pluggable Extras**.

### Consequences

* **Good, because**:
  * **Causal Evidence Guard**: `validate_causal_claim()` and Pydantic validators on `InterventionalQueryResult` strictly prevent elevating associations to interventional standing without verified identifiability and common support.
  * **Mandatory Uncertainty Coupling**: `ScenarioForecastResult` couples central estimates with confidence intervals, IQR, and OOD extrapolation flags. Attempting to extract bare point estimates raises `UncertaintyRequiredError` unless explicit acknowledgment is passed.
  * **NIST AI RMF 'MANAGE' Gating**: `ResponsibleAIGate` allows automated decision pipelines to block on excessive uncertainty width, ungrounded extrapolation, or lack of causal identifiability.
  * **Model & Scenario Cards Polish**: Enforces Mitchell et al. sections (intended use, out-of-scope, factors, quantitative analyses, ethical considerations, caveats). Cards feature zero-drift `populate_from_simulation()` methods.
  * **EU AI Act Readiness**: Provides `to_annex_iv()` export methods structuring simulator artifacts for Article 11(1) and Annex IV high-risk compliance preparation ahead of the August 2026 enforcement date.
  * **Zero-Bloat Core**: Core uses only Pydantic and NumPy; RAI integrations (`FairnessAuditAdapter`, `CounterfactualExplainerAdapter`, `ErrorAnalysisAdapter`) are lazy-loaded under `[fairness]`, `[explain]`, and `[erroranalysis]` extras.
* **Bad, because**:
  * Callers expecting simple scalar floats must adapt to `ScenarioForecastResult` objects or explicitly pass `acknowledge_no_uncertainty=True`.
  * Downstream pipelines must configure `ResponsibleAIGate` thresholds appropriately for their application tolerance.

### Confirmation

* Unit test suite `tests/unit/test_responsible_ai.py` verifies:
  1. Anti-auto-promotion of causal claims raising `UnqualifiedCausalClaimError`.
  2. Point estimate extraction blocking on bare calls and succeeding with explicit acknowledgment.
  3. `ResponsibleAIGate` halting execution on wide intervals and OOD extrapolation.
  4. Model and Scenario card completeness validation and `to_annex_iv()` rendering.
  5. Lazy import and error handling of optional RAI adapters.
* Documentation in `docs/responsible-ai.md` thoroughly maps NIST AI RMF 1.0 (GOVERN, MAP, MEASURE, MANAGE), GenAI Profile confabulation mitigation, ISO/IEC 42001, and the "On overclaiming" doctrine.

## Pros and Cons of the Options

### Option 1: Aspirational Documentation & Passive Logging
* Good: Minimal friction for casual users; no breaking API changes.
* Bad: Fails to prevent epistemic overclaiming; downstream decision systems will inevitably strip uncertainty; violates responsible AI principles.

### Option 2: Heavy RAI Monolith
* Good: Everything installed by default.
* Bad: Imposes heavy PyTorch/scikit-learn/Pandas/DiCE dependencies on lightweight users; brittle dependency solver resolutions.

### Option 3: API-Boundary Enforcement & Pluggable Extras
* Good: Hard mathematical guarantees at runtime; lightweight core; seamless integration with established governance frameworks.
* Bad: Requires explicit awareness of uncertainty and evidence levels by API consumers.

## More Information

* NIST AI Risk Management Framework 1.0 (NIST AI 100-1)
* NIST AI 600-1 Generative AI Profile
* ISO/IEC 42001:2023 Artificial Intelligence Management System (AIMS)
* Mitchell et al., "Model Cards for Model Reporting", FAccT 2019
* Regulation (EU) 2024/1689 (EU Artificial Intelligence Act), Annex IV
* Related ADRs: [ADR-010](ADR-010-systemic-trace-epistemic-honesty.md), [ADR-021](ADR-021-ood-detection-and-honest-causal-diagnostics.md), [ADR-033](ADR-033-scientific-ml-provenance-uq-and-causal-gates.md), [ADR-038](ADR-038-open-source-governance-community-health-and-rfc-process.md).
