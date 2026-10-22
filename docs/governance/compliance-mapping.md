# Regulatory & Compliance Mapping: NIST AI RMF 1.0 & EU AI Act

> **Executive Scope**: Technical governance mapping for Chief Legal Officers, General Counsel, Chief Risk Officers, and Compliance Architects.  
> **Authoritative Anchor**: Translates system architecture into technical inputs for Regulation (EU) 2024/1689 (EU AI Act) and NIST AI 100-1 (AI Risk Management Framework 1.0).

---

> [!IMPORTANT]
> **Legal Disclaimer & Regulatory Scope Notice**:  
> The mappings, exports, and architectural mechanisms documented herein represent **technical design characteristics and automated documentation capabilities** of the open-source Enterprise World Model Engine (`ewm-engine`). They provide structured technical evidence to streamline organizational compliance. **They do not constitute legal advice, regulatory certification, or a formal declaration of regulatory conformity.** Legal compliance under the EU AI Act or NIST AI RMF requires an organization-wide assessment encompassing business processes, data handling, and organizational governance beyond the software engine.

---

## 1. Executive Summary

As enterprise simulations and AI-assisted decision systems transition into high-stakes operational and capital allocation workflows, regulatory frameworks demand verifiable technical governance:
* **The Compliance Challenge:** Modern AI regulations penalize opaque "black-box" predictions, unquantified model uncertainty, and unverified data provenance.
* **The EWM Advantage:** EWM was architected from inception to make compliance **structurally easier**. By enforcing hard conservation laws, deterministic state fingerprints, formal causal identifiability gates, and automated regulatory documentation exports, EWM provides the verifiable technical substrate required by internal audit and external regulators.

---

## 2. NIST AI Risk Management Framework (AI RMF 1.0) Mapping

The National Institute of Standards and Technology (NIST) AI RMF defines four core functions: **GOVERN**, **MAP**, **MEASURE**, and **MANAGE**. EWM maps directly to these functions at the API boundary:

```
  ┌────────────────────────────────────────────────────────────────────────┐
  │                        NIST AI RMF 1.0 Alignment                       │
  └────────────────────────────────────────────────────────────────────────┘
        │                       │                     │                │
        ▼                       ▼                     ▼                ▼
   1. GOVERN                 2. MAP              3. MEASURE       4. MANAGE
  • RFC / ACP Process      • Mitchell Cards     • BCa Bootstrap  • WideUncertaintyWarning
  • RFC 9116 Security      • Scope Envelopes    • CRPS Scoring   • ResponsibleAIGate
  • SLSA L2+ Provenance    • Out-of-Scope Lists • STL Oracles    • Identifiability Gates
```

### Detailed NIST AI RMF Function Matrix

| NIST Function & Category | Subcategory Intent | EWM Implementation & Technical Artifact | Shipped Codebase Anchor |
| :--- | :--- | :--- | :--- |
| **GOVERN 1.1** | Legal and regulatory requirements are understood and managed. | Structured Model and Scenario Cards record regulatory classification, intended operational environment, and compliance constraints. | [`ewm_engine.cards.models.ModelCard`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/cards/models.py) |
| **GOVERN 1.2** | Accountability and supply-chain integrity are established. | SLSA Level 2+ provenance generation, SHA-256 state hashing, and deterministic PRNG seed trees ensure complete auditability of results. | [`ewm_engine.core.seed.SeedSequenceTree`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/seed.py)<br>[ADR-034](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-034-supply-chain-security-provenance-and-vulnerability-management.md) |
| **MAP 1.1** | Context of deployment and business impacts are documented. | Explicit `intended_use` and `out_of_scope` lists embedded directly into machine-readable simulation artifacts. | [`ewm_engine.cards.models.ScenarioCard`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/cards/models.py) |
| **MAP 2.3** | Scientific and operational boundaries are identified. | Boundary envelope and support detection detect when simulation trajectories extrapolate beyond empirical data support. | [`ewm_engine.evaluation.abstention`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/evaluation/abstention.py)<br>[ADR-021](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-021-ood-detection-and-honest-causal-diagnostics.md) |
| **MEASURE 1.1** | Approaches for measuring risk, error, and bias are utilized. | Non-parametric BCa bootstrap confidence intervals, conformal prediction coverage, and Continuous Ranked Probability Score (CRPS). | [`ewm_engine.evaluation.uncertainty`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/evaluation/uncertainty.py) |
| **MEASURE 2.2** | Systemic validity and physical laws are formally verified. | First-class algebraic invariant checks and Signal Temporal Logic (STL) Oracles continuously verify physical/financial invariants. | [`ewm_engine.verification.oracle`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/verification/oracle.py)<br>[ADR-027](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-027-trajectory-verification-oracle-graph-and-temporal-logic.md) |
| **MANAGE 1.1** | Identified risks are tracked, prioritized, and mitigated. | Structured warning hierarchy (`WideUncertaintyWarning`, `ExtrapolationWarning`) flags degradation before executive action. | [`ewm_engine.core.trust.ResponsibleAIGate`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/trust.py) |
| **MANAGE 2.2** | Mechanism to block automated actuation when risks exceed tolerance. | Formal gate rejects causal claims lacking backdoor identification or covariate overlap, halting downstream automated decisions. | [`ewm_engine.core.trust.validate_causal_claim`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/trust.py) |

---

## 3. European Union Artificial Intelligence Act (Regulation 2024/1689) Mapping

The EU AI Act establishes a risk-based framework. While pure simulation kernels for internal modeling are often classified as low-to-medium risk, when deployed to guide decisions in **critical infrastructure, employment allocation, or public services (Annex III)**, high-risk obligations apply.

EWM provides direct technical building blocks to fulfill the mandatory requirements for high-risk systems under **Chapter III**:

### Article-by-Article Alignment

```
           EU AI Act High-Risk Requirements (Chapter III)
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
Article 11 & Annex IV       Article 14                  Article 15
Technical Documentation     Human Oversight             Accuracy & Robustness
(Auto-Export via Card)      (Refusal Gates & Trace)     (Physics-IQ & Invariants)
```

#### 1. Article 11 & Annex IV: Technical Documentation
* **Requirement:** Detailed technical documentation drawn up before the system is placed on the market or put into service, demonstrating compliance.
* **EWM Solution:** Both `ModelCard` and `ScenarioCard` provide a programmatic `.to_annex_iv()` method:
  ```python
  from ewm_engine.cards.models import ScenarioCard

  card = ScenarioCard.from_simulation(scenario, trajectory_result)
  annex_iv_spec = card.to_annex_iv()
  ```
  This auto-populates all five required Annex IV sections:
  1. *Section 1:* General description, intended purpose, and version history.
  2. *Section 2:* Detailed description of system components, constraints, and underlying assumptions.
  3. *Section 3:* Monitoring, functioning, and human oversight controls.
  4. *Section 4:* Risk management system and invariant failure mitigation.
  5. *Section 5:* Cryptographic provenance manifest and change log.

#### 2. Article 14: Human Oversight
* **Requirement:** High-risk AI systems must be designed to enable natural persons to oversee their operation, understand outputs, and override or halt the system.
* **EWM Solution:**
  * **No Black-Box Actuation:** EWM is an advisory simulation kernel, not an autonomous actuator.
  * **Visual Trace Explanations:** Automated Mermaid graph trace diffs show human reviewers the precise causal path of an intervention.
  * **Epistemic Refusal:** When unobserved confounders make causal identification impossible, the engine halts with an `UnqualifiedCausalClaimError` rather than emitting misleading guidance to human operators.

#### 3. Article 15: Accuracy, Robustness, and Cybersecurity
* **Requirement:** High-risk systems must achieve appropriate levels of accuracy, robustness, and resilience against errors, faults, and inconsistencies.
* **EWM Solution:**
  * **Pinned-Runtime Bitwise Reproducibility (`BITWISE_REPRODUCIBLE`):** Eliminates unseeded stochastic drift under pinned Python/NumPy CPU runtimes (`OMP_NUM_THREADS=1`), paired with explicit numerical equivalence tolerances (`NUMERICALLY_EQUIVALENT_WITH_TOLERANCE`) across heterogeneous accelerators.
  * **Physics-IQ Guardrail:** Continuously monitors invariant conservation rates, ensuring simulation trajectories cannot drift into physically impossible states.
  * **Supply-Chain Security:** Automated dependency pinning, vulnerability scanning, and SLSA Level 2+ build attestations.

---

## 4. The Anti-Hallucination Advantage in Regulated Environments

The primary regulatory hazard of deploying generative Large Language Models in corporate decision pipelines is **confabulation** (NIST AI 600-1 GenAI Profile):
* Under the EU AI Act (Article 15) and financial regulatory standards (e.g., SR 11-7 model risk management), models that invent financial assets or violate physical capacity are non-compliant.
* **EWM eliminates confabulation at the kernel level**: physical conservation laws (inventory, mass, cash flow) are hard-coded algebraic constraints. The model cannot hallucinate positive cash flow without double-entry balancing.

---

## 5. Summary Checklist for Compliance & Audit Officers

| Compliance Domain | Audit Question | EWM Technical Evidence Provided |
| :--- | :--- | :--- |
| **Model Documentation** | Can we produce an up-to-date technical dossier of model assumptions? | `card.to_annex_iv()` structured JSON/YAML export. |
| **Replay & Audit** | Can a disputed forecast from 18 months ago be reproduced exactly? | Deterministic `SeedSequence` + SHA-256 state fingerprint. |
| **Uncertainty Disclosure** | Are decision-makers warned about extrapolation risks? | BCa bootstrap confidence intervals + `WideUncertaintyWarning`. |
| **Causal Validity** | Is management confusing observational correlation with true policy effect? | Backdoor criterion identifiability check + E-values. |
| **Data Integrity** | Are physical conservation laws verified during simulation? | Automated invariant verification via `TrajectoryStatus.INVALID`. |
