# Six Decision-Grade Guarantees

> *"Enterprise executives do not buy neural weights or speculative forecasts; they buy decision-grade capability: auditability, certified uncertainty, inviolable physical reality, and regulatory defensibility."*

---

## Executive Summary

Most enterprise software projects exploring "AI world models" stall at the transition from laboratory prototype to capital allocation. The barrier is almost never raw predictive accuracy — it is the absence of **decision-grade guarantees**. 

Large language models hallucinate impossible inventories; unconstrained neural networks violate mass conservation; and black-box point forecasts hide existential tail risk from the Board of Directors.

Enterprise World Model Engine (`ewm-engine`) provides **six testable, architecturally enforced guarantees** designed to satisfy the Chief Executive, Chief Operating Officer, Chief Financial Officer, and Chief Risk Officer.

> [!NOTE]
> **Maturity & Scope Notice**: While core simulation primitives (state immutability, conservation constraints, deterministic seed trees) are hardened and gated by automated CI, advanced modules (such as neural residual dynamics, SBI calibration, and GNN propagation) are marked **Beta / Experimental** under [SPEC 0](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/spec0-policy.md). EWM does not claim uniform production-readiness across experimental subpackages.

---

## The Six Guarantees

```
                   ┌────────────────────────────────────────────────────────┐
                   │               Decision-Grade Guarantees                │
                   └────────────────────────────────────────────────────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
1. No Hallucination         2. Mandatory Uncertainty    3. Cryptographic Audit
   (Conservation Invariants)   (Conformal & E-Values)      (Zero-Drift Replay)
     │                           │                           │
     ▼                           ▼                           ▼
4. Explainable Causality    5. Regulatory Compliance    6. Decision-Backtesting
   (SCM & Mermaid Traces)      (NIST RMF & EU AI Act)      (Minimax Regret Bounds)
```

---

### Guarantee 1: Zero Hallucination via Enforced Conservation Laws

> **Executive Statement:** *Physical resources, ledger balances, and contractual constraints cannot be conjured or erased by algorithmic optimism.*

#### Why This Matters to the Enterprise
Generative AI models and deep neural networks are unconstrained token or function approximators. In supply chain planning, an LLM agent or unconstrained regressor will casually ship goods from a non-existent warehouse or generate positive cash flow without double-entry debits.

#### How EWM Enforces It
EWM separates **KNOWN explicit physical rules** from **LEARNED probabilistic dynamics**. Physical conservation laws (mass balance, capacity limits, double-entry accounting) are hard algebraic invariants evaluated before and after every transition:
* Any action or state transition that violates an invariant triggers immediate rollout rejection (`TrajectoryStatus.INVALID`).
* In hybrid systems, learned neural networks are restricted to modeling residual dynamics over a hard conservation layer ([ADR-042](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-042-grey-box-conservation-flywheel.md)).
* Invariant violations are continuously tracked and scored via the **Physics-IQ Guardrail** ([ADR-044](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-044-causal-correctness-metamorphic-evaluation.md)).

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.constraints.standard`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/constraints/standard.py) & [`ewm_engine.experimental.conservation`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/experimental/conservation.py).
* **Executable Acceptance Fixture:** [`examples/minimal_warehouse/run.py`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/examples/minimal_warehouse/run.py) (reproducible mass conservation test verifying $S + E + I + R = N$ with zero drift).
* **Automated Contract Test:** [`tests/contract/test_grey_box_contract.py`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/tests/contract/test_grey_box_contract.py).

---

### Guarantee 2: Mandatory Uncertainty Quantification (No Bare Point Estimates)

> **Executive Statement:** *The engine refuses to deliver a single speculative number without certified confidence intervals and out-of-support warnings.*

#### Why This Matters to the Enterprise
Point estimates ("revenue will grow 8.2%") hide catastrophic downside tails. A model that is 99% confident in a 1% probability disaster is a corporate liability. High-stakes capital allocation demands honest dispersion bounds.

#### How EWM Enforces It
* **Mandatory Dispersion:** The public API rejects bare point predictions via [`ScenarioForecastResult`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/core/trust.py). Every forecast outputs non-parametric BCa bootstrap confidence intervals and distribution quantiles ($p05, p50, p95$).
* **Finite-Sample Conformal Bands:** Simulation rollouts can be calibrated to certified $(1-\alpha)$ prediction coverage intervals without making restrictive Gaussian assumptions.
* **Anytime-Valid E-Values:** Hypothesis comparisons between scenarios report sequential E-values, guaranteeing false-discovery rate control even when decision-makers stop tests early based on interim data.
* **Epistemic Refusal (RWM-U):** When scenario parameters enter out-of-distribution (OOD) territory where historical support is zero, the engine issues structured `WideUncertaintyWarning` and down-weights the policy during optimization ([ADR-039](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-039-responsible-ai-trust-and-anti-overclaiming-enforcement.md)).

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.evaluation.uncertainty`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/uncertainty.py) & [`ewm_engine.evaluation.abstention`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/abstention.py).
* **Executable Proof:** See the CI-generated hero asset [`assets/hero-proof-dark.svg`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/assets/hero-proof-dark.svg) showing empirical conformal bands.
* **Automated Contract Test:** [`tests/contract/test_calibration_contract.py`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/tests/contract/test_calibration_contract.py).

---

### Guarantee 3: Cryptographic Auditability & Bit-for-Bit Reproducibility

> **Executive Statement:** *Any simulation run, scenario branch, or board-level recommendation can be replayed bit-for-bit three years later from an immutable cryptographic hash.*

#### Why This Matters to the Enterprise
When external regulators, internal audit committees, or litigators demand to know why a $50M inventory write-down was authorized, "the model weights updated in the cloud" is an unacceptable defense. Decisions must be tied to deterministic evidence.

#### How EWM Enforces It
* **Immutable State Fingerprinting:** Every [`WorldState`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/core/state.py) calculates an SHA-256 fingerprint from canonical JSON byte representations, enforcing state immutability.
* **Hierarchical PRNG Seed Trees:** Uses NumPy 64-bit `SeedSequence` trees to isolate stochastic branches. Branching a counterfactual scenario cannot leak entropy or disturb parallel Monte Carlo trajectories ([ADR-012](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-012-determinism-reproducibility-and-branch-isolation.md)).
* **Supply-Chain Provenance:** Packaged with SLSA Level 2+ provenance attestations, pinned dependencies, and hermetic build recipes ([ADR-034](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-034-supply-chain-security-provenance-and-vulnerability-management.md)).

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.core.seed`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/core/seed.py) & [`ewm_engine.provenance.trace`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/provenance/trace.py).
* **Executable Acceptance Fixture:** [`tests/examples/test_documented_examples.py`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/tests/examples/test_documented_examples.py) (`test_readme_quickstart_executes`).
* **Automated Invariant Test:** [`tests/unit/test_hermetic_and_clock.py`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/tests/unit/test_hermetic_and_clock.py).

---

### Guarantee 4: Explainable Causality & Transparent Ripple Effects

> **Executive Statement:** *Recommendations reveal the exact structural mechanism behind an outcome — who caused what, through which path, and under what assumptions.*

#### Why This Matters to the Enterprise
Correlation-based models suggest associations that collapse under policy changes (Goodhart's Law / Lucas Critique). Decision-makers need to see the causal chain of transmission before pulling organizational levers.

#### How EWM Enforces It
* **Native Structural Causal Model (SCM):** The relational graph implements a formal SCM. Interventions use Judea Pearl's graph surgery ($do(x)$), severing incoming dependency arrows ([ADR-040](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-040-twin-rollout-counterfactual-scm-kernel.md)).
* **Twin-Rollout Counterfactuals:** Factual and counterfactual trajectories share identical realized exogenous noise ($U$), isolating the net treatment effect without confounding noise divergence.
* **Automated Visual Trace Diffs:** Emits visual Mermaid diagrams showing exact divergent and invariant subgraphs ([`TraceDiff`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/provenance/trace.py)).

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.core.scm`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/core/scm.py) & [`ewm_engine.simulation.counterfactual`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/simulation/counterfactual.py).
* **Executable Example:** [`examples/counterfactual_warehouse_reroute.py`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/examples/counterfactual_warehouse_reroute.py).
* **Automated Contract Test:** [`tests/contract/test_scm_counterfactual_contract.py`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/tests/contract/test_scm_counterfactual_contract.py).

---

### Guarantee 5: Regulatory Compliance & Governance Readiness

> **Executive Statement:** *Documentation and technical evidence automatically export into regulatory frameworks (NIST AI RMF 1.0, EU AI Act Annex IV).*

#### Why This Matters to the Enterprise
The regulatory landscape (EU AI Act, ISO/IEC 42001, US Executive Orders) is imposing strict technical documentation and risk mitigation duties on high-stakes automated decision systems. Manual compliance engineering is slow and error-prone.

#### How EWM Enforces It
* **Automated Annex IV Technical Export:** Both [`ModelCard`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/cards/models.py) and [`ScenarioCard`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/cards/models.py) include `.to_annex_iv()` methods that export system parameters, assumptions, and constraint profiles directly into the EU AI Act Annex IV technical documentation structure.
* **NIST AI RMF 1.0 Alignment:** Formal mapping across GOVERN, MAP, MEASURE, and MANAGE functions ([ADR-039](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-039-responsible-ai-trust-and-anti-overclaiming-enforcement.md)).
* **Human-in-the-Loop Safeguards:** Epistemic gates block automated execution when scenario parameters violate common support or identifiability.

#### Shipped Evidence & Runnable Proof
* **Governance Guide:** [`docs/responsible-ai.md`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/responsible-ai.md) & [`docs/governance/compliance-mapping.md`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/governance/compliance-mapping.md).
* **Automated Unit Test:** [`tests/unit/test_cards.py`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/tests/unit/test_cards.py) (testing `to_annex_iv()` export compliance).

---

### Guarantee 6: Rigorous Decision-Backtesting & Regret Bounds

> **Executive Statement:** *Policies are ranked by decision quality under deep uncertainty, not merely historical curve-fitting.*

#### Why This Matters to the Enterprise
A model that achieves 98% forecasting accuracy on peacetime data will recommend catastrophic policies during unexpected supply-chain shocks or interest-rate spikes. Enterprises must optimize for decision robustness, not just rollout accuracy.

#### How EWM Enforces It
* **Decision-Focused Learning (`ewm decide`):** Ranks candidate actions across multi-scenario ensembles using robust decision metrics: **Minimax Regret**, **Satisficing / Domain Criterion**, and **Info-Gap Analysis** ([ADR-041](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-041-decision-focused-scenario-discovery.md)).
* **Scenario Discovery (PRIM/CART):** Automates the Patient Rule Induction Method to isolate the precise hyper-dimensional vulnerability subspace where a policy fails, preventing blind spots.
* **Statistical Hypothesis Testing:** Pairwise scenario comparisons use Welch’s t-test and Mann-Whitney U to flag statistically significant differences ($p < 0.05$) while suppressing noisy trivial gains.

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.decision.robustness`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/decision/robustness.py) & [`ewm_engine.decision.discovery`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/decision/discovery.py).
* **Executable CLI Command:** `ewm decide` & [`compare_scenarios()`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/comparison.py).
* **Automated Contract Test:** [`tests/contract/test_decision_contract.py`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/tests/contract/test_decision_contract.py).

---

## Summary Matrix: The Guarantees in Action

| Guarantee | Executive Risk Addressed | Programmatic Mechanism | Primary Shipped Anchor |
| :--- | :--- | :--- | :--- |
| **1. No Hallucination** | Impossible physical/financial plans | Inviolable conservation invariants | `ewm_engine.constraints.standard` |
| **2. Mandatory Uncertainty** | Concealed downside tail risk | BCa bootstrap + Conformal coverage | `ewm_engine.evaluation.uncertainty` |
| **3. Cryptographic Audit** | Regulatory fines & non-reproducibility | SHA-256 state hashes + PRNG trees | `ewm_engine.core.seed` |
| **4. Explainable Causality** | Ineffective correlational policies | SCM graph surgery + Twin-Rollout | `ewm_engine.core.scm` |
| **5. Regulatory Compliance** | EU AI Act & NIST non-compliance | Auto Annex IV & NIST RMF cards | `ewm_engine.cards.models` |
| **6. Decision-Backtesting** | Fragile plans under market shifts | Minimax Regret & PRIM discovery | `ewm_engine.decision.robustness` |
