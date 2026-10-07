# Responsible AI, Trust & Anti-Overclaiming Enforcement

> **Discipline**: Trustworthy & Responsible AI Engineering  
> **Research & Governance Anchors**: NIST AI RMF 1.0, NIST AI 600-1 (GenAI Profile), ISO/IEC 42001 (AIMS), Mitchell et al. (Model Cards for Model Reporting), Regulation (EU) 2024/1689 (EU AI Act Annex IV).

---

## Executive Summary

Enterprise simulations and socio-technical world models are frequently deployed to inform high-stakes organizational, operational, and capital decisions. In these settings, **scientific humility cannot be merely aspirational—it must be structurally and architecturally enforced**.

The Enterprise World Model Engine (`ewm-engine`) translates epistemic responsibility into programmatic API guarantees:
1. **No Unqualified Causal Claims**: Observational associations cannot be promoted to interventional or causal status without passing formal mathematical identifiability and covariate overlap gates.
2. **Mandatory Uncertainty Quantification**: Downstream consumers cannot extract bare point estimates without acknowledging confidence intervals, dispersion, and regime support.
3. **Automated Extrapolation Safeguards**: When rollouts stray beyond empirical support, out-of-distribution (OOD) flags trigger structured warnings and block automated actuation.
4. **Reproducible Documentation Readiness**: Model and Scenario Cards adhere to the Mitchell et al. schema, auto-populate from simulation runs to prevent drift, and export technical documentation structures aligned with ISO/IEC 42001 and EU AI Act Annex IV.

---

## 1. NIST AI Risk Management Framework (AI RMF 1.0) Mapping

The National Institute of Standards and Technology (NIST) AI RMF provides four core functions for managing AI risks: **GOVERN**, **MAP**, **MEASURE**, and **MANAGE**. The table below details how `ewm-engine` satisfies each function at the codebase and API boundaries.

| NIST RMF Function | `ewm-engine` Enforcement Mechanism | Codebase & Architecture Anchor |
| :--- | :--- | :--- |
| **GOVERN** | • Transparent community governance ladder and voting rules.<br>• Strict RFC / API Change Proposal (ACP) process for breaking changes.<br>• RFC 9116 security policy with 48h SLA and GitHub Private Vulnerability Reporting.<br>• Automated commit signing, SHA pinning, and supply-chain provenance. | • [ADR-038](adr/ADR-038-open-source-governance-community-health-and-rfc-process.md)<br>• `GOVERNANCE.md`<br>• `SECURITY.md`<br>• `.well-known/security.txt` |
| **MAP** | • Mitchell-aligned `ModelCard` with mandatory `out_of_scope` use cases.<br>• Scenario-level `ScenarioCard` requiring explicit `assumptions` and operating boundaries.<br>• Boundary and support envelope detection preventing silent regime drift. | • `ewm_engine.cards.models.ModelCard`<br>• `ewm_engine.cards.models.ScenarioCard`<br>• [ADR-021](adr/ADR-021-ood-detection-and-honest-causal-diagnostics.md) |
| **MEASURE** | • BCa / non-parametric bootstrap confidence intervals on all aggregate metrics.<br>• Continuous Ranked Probability Score (CRPS) and coverage calibration tests.<br>• Identifiability check (Backdoor Criterion) and common support / propensity overlap.<br>• Verification Oracle evaluating signal temporal logic (STL) and graph invariants. | • `ewm_engine.evaluation.uncertainty`<br>• `ewm_engine.experimental.causal`<br>• `ewm_engine.verification.oracle`<br>• [ADR-033](adr/ADR-033-scientific-ml-provenance-uq-and-causal-gates.md) |
| **MANAGE** | • Structured warning taxonomy (`WideUncertaintyWarning`, `ExtrapolationWarning`).<br>• `ResponsibleAIGate` blocking decision pipelines on excessive uncertainty or OOD states.<br>• Mandatory uncertainty coupling via `ScenarioForecastResult`.<br>• Strict prohibition of `"causes"` edge promotion. | • `ewm_engine.core.trust.ResponsibleAIGate`<br>• `ewm_engine.core.trust.ScenarioForecastResult`<br>• [ADR-039](adr/ADR-039-responsible-ai-trust-and-anti-overclaiming-enforcement.md) |

### Generative AI Profile (NIST AI 600-1) Alignment

NIST AI 600-1 highlights **confabulation / hallucination** as a primary risk in generative architectures: models emit plausible-sounding narratives that violate physical, chemical, or financial reality.

In `ewm-engine`, learned neural dynamics (e.g., Torch MLP residual learners or Graph Neural Networks) are never permitted to operate as unconstrained autoregressive generators. Instead:
- **First-Class Invariant Filters**: Physical conservation laws (mass, energy, inventory conservation, double-entry bookkeeping) act as hard pre- and post-transition constraints.
- **Rollout Invalidation**: Any dynamic step violating a hard physical constraint immediately invalidates the rollout trajectory (`TrajectoryStatus.INVALID`), preventing confabulated states from contaminating scenario evaluations.

---

## 2. API-Boundary Epistemic Guarantees

### Causal Evidence Guard: Anti-Auto-Promotion

In observational enterprise data, correlation is ubiquitous while genuine causal identification is rare. Under SPEC 9 and [ADR-033](adr/ADR-033-scientific-ml-provenance-uq-and-causal-gates.md), `ewm-engine` forbids auto-promoting an observational association to a causal claim.

```python
from ewm_engine.core.trust import validate_causal_claim, UnqualifiedCausalClaimError
from ewm_engine.provenance.evidence import EvidenceLevel

# Attempting to emit an INTERVENTIONAL claim without identifiability or overlap fails:
try:
    level = validate_causal_claim(
        claimed_level=EvidenceLevel.INTERVENTIONAL,
        is_identifiable=False,  # Unobserved confounding or open backdoor paths
        overlap_satisfied=False, # Positivity violation
        strict=True,
    )
except UnqualifiedCausalClaimError as exc:
    print(f"Blocked overclaim: {exc}")
```

If `strict=False`, the engine downgrades the claim to `EvidenceLevel.PREDICTIVE` and emits an `InsufficientCausalEvidenceWarning`.

### Mandatory Uncertainty Forecasts

Point estimates without uncertainty intervals mislead decision-makers by implying false precision. Public simulation forecasts are wrapped in `ScenarioForecastResult`:

```python
result = engine.run(scenario)
forecast = result.get_forecast("net_inventory", confidence_level=0.90)

print(forecast)
# Output:
# Forecast[net_inventory]: point=142.500 | 90% CI=[118.200, 168.400] | IQR=24.100 | Regime=Grounded | Evidence=predictive

# Extracting bare point estimate without acknowledging uncertainty raises an error:
try:
    val = forecast.get_point_estimate()
except UncertaintyRequiredError:
    # Forces explicit consumer opt-in:
    val = forecast.get_point_estimate(acknowledge_no_uncertainty=True)
```

### Configurable MANAGE Gating (`ResponsibleAIGate`)

Downstream automated execution pipelines can install `ResponsibleAIGate` to block decisions that violate safety or epistemic criteria:

```python
from ewm_engine.core.trust import ResponsibleAIGate, DecisionBlockedError

gate = ResponsibleAIGate(
    max_relative_ci_width=1.5,       # Block if CI width > 150% of estimate
    require_causal_identifiability=True, # Flag purely associational estimates
    block_on_extrapolation=True,     # Block if rollout entered OOD space
    strict=True,
)

try:
    report = gate.validate_forecast(forecast)
except DecisionBlockedError as err:
    print(f"Execution halted by Responsible AI Gate: {err.reasons}")
```

---

## 3. Model & Scenario Cards (Mitchell et al.)

All trained dynamics models and simulation scenarios ship with machine-readable, schema-validated metadata cards based on Mitchell et al. (*Model Cards for Model Reporting*, FAccT 2019).

### Schema Completeness Guarantee

Both `ModelCard` and `ScenarioCard` enforce comprehensive sections:
1. **Model Details**: Developer, architecture, release version, runtime interactions.
2. **Intended Use & Out-of-Scope**: Explicit positive intended operational use cases and strictly forbidden out-of-scope deployments.
3. **Factors & Subgroups**: Demographic, organizational, or regional factors across which dynamics may vary.
4. **Metrics & Quantitative Analyses**: Evaluation metrics, sample sizes, confidence intervals, and distributional summaries.
5. **Training & Evaluation Data**: Provenance fingerprints, split methodologies, and environmental conditions.
6. **Ethical Considerations & Caveats**: Potential societal impacts, systemic risks, and documented limitations.

### Zero-Drift Auto-Population

Manual documentation frequently drifts from code. `ewm-engine` prevents card drift by providing automated population methods directly from execution outputs:

```python
# Auto-populate metrics and quantitative distribution summaries
card.populate_from_simulation(sim_result)
card.validate_mitchell_completeness(strict=True)
```

---

## 4. International Governance Frameworks

### ISO/IEC 42001 (AIMS) Technical Evidence

ISO/IEC 42001 specifies requirements for establishing, implementing, maintaining, and continually improving an Artificial Intelligence Management System (AIMS). The artifacts emitted by `ewm-engine` serve as direct audit evidence for AIMS controls:

- **Clause 6.1 (Actions to address risks and opportunities)**: Provided by `ScenarioCard` assumptions, failure mode catalogues, and `ResponsibleAIGate` threshold definitions.
- **Clause 8.4 (AI system impact assessment)**: Provided by `ModelCard` ethical considerations, quantitative analysis across subgroups, and sensitivity analyses.
- **Clause 9.1 (Monitoring, measurement, analysis, and evaluation)**: Provided by cryptographic provenance manifests (`Provenance`), deterministic state fingerprints, and CRPS/coverage calibration tests.

### EU AI Act Annex IV Technical Documentation Readiness

Regulation (EU) 2024/1689 (the European Union Artificial Intelligence Act) establishes comprehensive obligations for high-risk AI systems. High-risk obligations become enforceable from **August 2026**.

To support engineering readiness ahead of the 2026 enforcement timeline, `ewm-engine` provides `to_annex_iv()` reprojection methods on both `ModelCard` and `ScenarioCard`:

```python
annex_iv_payload = card.to_annex_iv()
```

This exports a structured dictionary mapping system parameters into Annex IV sections:
- **Section 1**: General description of the AI system, intended purpose, and out-of-scope scenarios.
- **Section 2**: Detailed description of system elements, theoretical assumptions, constraints, and training/validation datasets.
- **Section 3**: Monitoring, functioning, and control (extrapolation safeguards, human oversight mechanisms).
- **Section 4**: Risk management system, ethical considerations, and invariant gates.
- **Section 5**: Lifecycle changes and cryptographic provenance fingerprint.

> [!IMPORTANT]
> **Legal Disclaimer**: The `to_annex_iv()` renderer is a technical readiness tool for developers and compliance teams. It reflects simulator configuration and execution data; it does **not** constitute legal advice or a formal declaration of regulatory conformity.

---

## 5. On Overclaiming: Principles of Scientific Humility

> "The first principle is that you must not fool yourself—and you are the easiest person to fool."  
> — Richard Feynman

Deploying world models to guide enterprise policy carries inherent risks of overclaiming. We document three core distinctions every practitioner must maintain.

### 1. What a Bootstrap CI Does and Does Not Mean

A 90% bootstrap confidence interval $[L, U]$ computed over $N$ Monte Carlo rollouts represents **sampling variance given the model specifications and stochastic seeds**.

- **What it means**: If the world model equations, parameters, and random distributions were an exact representation of reality, the true scenario mean would fall within $[L, U]$ in approximately 90% of hypothetical repeated simulation experiments.
- **What it does NOT mean**: It does **not** imply that reality will conform to $[L, U]$ with 90% probability. A bootstrap CI cannot account for structural misspecification, unobserved regime shifts, or invalid foundational assumptions.

### 2. Correlation vs. Causal Diagnostics vs. Causal Proof

The engine's causal layer implements formal diagnostics based on Judea Pearl's do-calculus and Donald Rubin's Potential Outcomes framework:
- **Observational Association**: $P(Y \mid X=x)$. Measures correlation.
- **Interventional Distribution**: $P(Y \mid do(X=x))$. Measures counterfactual effect *under the assumption that the supplied DAG represents the true causal structure* and that no unobserved confounders exist.
- **Common Support / Positivity**: Verifies that both treated and control units have non-zero probability of assignment across the covariate space.

**Crucial Warning**: Passing Backdoor Criterion identifiability and propensity score overlap is a **mathematical property of the model DAG**, not empirical proof of physical causality. If the analyst omits a critical confounding variable from the DAG, the model will faithfully compute an identifiable causal estimate that is completely false in the real world.

### 3. Surface Realism $\ne$ Systemic Validity

Generative AI, Large Language Models, and deep neural networks excel at creating **surface realism**: generated text, dialogue, and synthetic time-series that "look and feel" convincing to human reviewers.

Surface realism is an epistemic hazard:
- An LLM agent may generate an eloquently reasoned quarterly supply plan that secretly violates double-entry cash balances or material conservation laws.
- A deep autoregressive network may forecast a smooth economic recovery by drifting into state spaces where physical capacity constraints are invisibly violated.

In `ewm-engine`, **systemic validity always supersedes surface realism**. Plausibility is rejected unless verified against immutable algebraic conservation laws, invariant bounds, and formal temporal logic.

---

## 6. Optional Responsible AI Tooling (Extras)

For specialized fairness audits, counterfactual explainability, and error cohort discovery, `ewm-engine` provides pluggable adapters:

| Tooling | Extra Package | Purpose in World Models |
| :--- | :--- | :--- |
| **Fairlearn** | `ewm-engine[fairness]` | Evaluates demographic parity and equalized odds in actor decision policies across sensitive cohorts. |
| **DiCE & InterpretML** | `ewm-engine[explain]` | Computes actionable counterfactual shifts: what minimal changes in world state variables prevent invariant failures? |
| **Microsoft RAI ErrorAnalysis** | `ewm-engine[erroranalysis]` | Builds decision trees over rollout traces to isolate sub-populations and feature spaces where simulation error is concentrated. |

```bash
# Install optional RAI packages
pip install ewm-engine[fairness,explain,erroranalysis]
```

All adapters are lazy-imported and domain-neutral, ensuring that the engine core remains lightweight, robust, and zero-dependency beyond NumPy and Pydantic.
