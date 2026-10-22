# Six Decision-Grade Guarantees

> *"Enterprise executives do not buy neural weights or speculative forecasts; they buy decision-grade capability: auditability, certified uncertainty, inviolable physical reality, and regulatory defensibility."*

---

## Executive Summary

Most enterprise software projects exploring "AI world models" stall at the transition from laboratory prototype to capital allocation. The barrier is almost never raw predictive accuracy — it is the absence of **decision-grade guarantees**. 

Large language models hallucinate impossible inventories; unconstrained neural networks violate mass conservation; and black-box point forecasts hide existential tail risk from the Board of Directors.

Enterprise World Model Engine (`ewm-engine`) provides **six testable, architecturally enforced guarantees** designed to satisfy the Chief Executive, Chief Operating Officer, Chief Financial Officer, and Chief Risk Officer.

> [!NOTE]
> **Maturity & Scope Notice**: While core simulation primitives (state immutability, conservation constraints, deterministic seed trees) are hardened and gated by automated CI, advanced modules (such as neural residual dynamics, SBI calibration, and GNN propagation) are marked **Beta / Experimental** under [SPEC 0](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/spec0-policy.md). EWM does not claim uniform production-readiness across experimental subpackages.

---

## The Six Guarantees

```
                   ┌────────────────────────────────────────────────────────┐
                   │               Decision-Grade Guarantees                │
                   └────────────────────────────────────────────────────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
1. Conservation Guardrail   2. Mandatory Uncertainty    3. Cryptographic Audit
   (Bounded Invariants)        (Conformal & E-Values)      (Deterministic Replay)
     │                           │                           │
     ▼                           ▼                           ▼
4. Explainable Causality    5. Regulatory Compliance    6. Decision-Backtesting
   (SCM & Mermaid Traces)      (NIST RMF & EU AI Act)      (Minimax Regret Bounds)
```

---

### Guarantee 1: Enforced Conservation Laws Within Machine Precision

> **Executive Statement:** *Physical resources, ledger balances, and contractual constraints are bounded by explicit algebraic invariants evaluated at every transition.*

#### Why This Matters to the Enterprise
Generative AI models and deep neural networks are unconstrained token or function approximators. In supply chain planning, an LLM agent or unconstrained regressor will casually ship goods from a non-existent warehouse or generate positive cash flow without double-entry debits.

#### How EWM Enforces It
EWM separates **KNOWN explicit physical rules** from **LEARNED probabilistic dynamics**. Physical conservation laws (mass balance, capacity limits, double-entry accounting) are hard algebraic invariants evaluated before and after every transition:
* Any action or state transition that violates an invariant triggers immediate rollout rejection (`TrajectoryStatus.INVALID`).
* In hybrid systems, learned neural networks are restricted to modeling residual dynamics over a hard conservation layer ([ADR-042](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-042-grey-box-conservation-flywheel.md)).
* Invariant violations are continuously tracked and scored via the **Physics-IQ Guardrail** ([ADR-044](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-044-causal-correctness-metamorphic-evaluation.md)).

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.constraints.standard`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/constraints/standard.py) & [`ewm_engine.experimental.conservation`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/experimental/conservation.py).
* **Executable Acceptance Fixture:** [`examples/minimal_warehouse/run.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/examples/minimal_warehouse/run.py) and `reproduce_results.py --card seir` (verifying $S + E + I + R = N$ within IEEE-754 floating-point precision $\le 10^{-12}$, observed max residual $4.55 \times 10^{-13}$).
* **Automated Contract Test:** [`tests/contract/test_grey_box_contract.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/tests/contract/test_grey_box_contract.py).

---

### Guarantee 2: Mandatory Uncertainty Quantification (No Bare Point Estimates)

> **Executive Statement:** *The engine refuses to deliver a single speculative number without certified confidence intervals and out-of-support warnings.*

#### Why This Matters to the Enterprise
Point estimates ("revenue will grow 8.2%") hide catastrophic downside tails. A model that is 99% confident in a 1% probability disaster is a corporate liability. High-stakes capital allocation demands honest dispersion bounds.

#### How EWM Enforces It
* **Mandatory Dispersion & 6-Part Uncertainty Taxonomy:** The public API rejects bare point predictions via [`ScenarioForecastResult`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/trust.py). Every distribution and delta carries an explicit `UncertaintyTaxonomyBreakdown` (separating `aleatoric`, `parameter`, `structural_model`, `runtime_context`, `monte_carlo_error`, and unestimated `model_discrepancy_vs_reality`), explicit $P_{\text{model}}$ estimands, and non-parametric bootstrap intervals (`percentile`, `basic`, `normal`, `studentized`, `bca`).
* **Finite-Sample Marginal Conformal Bands & Online ACI:** Simulation rollouts can be calibrated to finite-sample $(1-\alpha)$ **marginal** prediction coverage intervals under exchangeability (`ConformalIntervalPredictor`), with slice-conditional coverage audits and online Adaptive Conformal Inference (`AdaptiveConformalInference`) under distribution shift.
* **Disambiguated Safe-Testing e-Processes & Confounding Sensitivity E-Values:** Sequential monitoring uses anytime-valid **safe-testing e-processes** (`SafeTestingEProcess`, $\mathbb{E}_{H_0}[e_t \mid \mathcal{F}_{t-1}] \le 1$; Grünwald et al., 2024) for optional stopping, strictly separated from observational **confounding Sensitivity E-values** (`SensitivityEValueResult`, VanderWeele & Ding, 2017).
* **Epistemic Refusal (RWM-U):** When scenario parameters enter out-of-distribution (OOD) territory where historical support is zero, the engine issues structured `WideUncertaintyWarning` and down-weights the policy during optimization ([ADR-039](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-039-responsible-ai-trust-and-anti-overclaiming-enforcement.md)).

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.evaluation.uncertainty`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/evaluation/uncertainty.py) & [`ewm_engine.evaluation.abstention`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/evaluation/abstention.py).
* **Executable Proof:** See the CI-generated hero asset [`assets/hero-proof-dark.svg`](../assets/hero-proof-dark.svg) showing empirical conformal bands.
* **Automated Contract Test:** [`tests/contract/test_calibration_contract.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/tests/contract/test_calibration_contract.py).

---

### Guarantee 3: Cryptographic Auditability & Tiered Reproducibility

> **Executive Statement:** *Any simulation run, scenario branch, or recommendation can be replayed bit-for-bit on the certified pure-Python/NumPy kernel (`BITWISE_REPRODUCIBLE`) from an immutable cryptographic hash and seed manifest.*

#### Why This Matters to the Enterprise
When external regulators, internal audit committees, or litigators demand to know why a $50M inventory write-down was authorized, "the model weights updated in the cloud" is an unacceptable defense. Decisions must be tied to deterministic evidence.

#### How EWM Enforces It
* **Immutable State Fingerprinting:** Every [`WorldState`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/state.py) calculates an SHA-256 fingerprint from canonical JSON byte representations, enforcing state immutability.
* **Hierarchical PRNG Seed Trees:** Uses NumPy 64-bit `SeedSequence` trees to isolate stochastic branches. Branching a counterfactual scenario cannot leak entropy or disturb parallel Monte Carlo trajectories ([ADR-012](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-012-determinism-reproducibility-and-branch-isolation.md)).
* **Supply-Chain Provenance:** Packaged with SLSA Level 2+ provenance attestations, pinned dependencies, and hermetic build recipes ([ADR-034](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-034-supply-chain-security-provenance-and-vulnerability-management.md)).

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.core.seed`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/seed.py) & [`ewm_engine.provenance.trace`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/provenance/trace.py).
* **Executable Acceptance Fixture:** [`tests/examples/test_documented_examples.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/tests/examples/test_documented_examples.py) (`test_readme_quickstart_executes`).
* **Automated Invariant Test:** [`tests/unit/test_hermetic_and_clock.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/tests/unit/test_hermetic_and_clock.py).

---

### Guarantee 4: Explainable Causality & Transparent Ripple Effects

> **Executive Statement:** *Recommendations reveal the exact structural mechanism behind an outcome — who caused what, through which path, and under what assumptions.*

#### Why This Matters to the Enterprise
Correlation-based models suggest associations that collapse under policy changes (Goodhart's Law / Lucas Critique). Decision-makers need to see the causal chain of transmission before pulling organizational levers.

#### How EWM Enforces It
* **Native Structural Causal Model (SCM):** The relational graph implements a formal SCM. Interventions use Judea Pearl's graph surgery ($do(x)$), severing incoming dependency arrows ([ADR-040](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-040-twin-rollout-counterfactual-scm-kernel.md)).
* **Twin-Rollout Counterfactuals:** Factual and counterfactual trajectories share identical realized exogenous noise ($U$), isolating the net treatment effect without confounding noise divergence.
* **Automated Visual Trace Diffs:** Emits visual Mermaid diagrams showing exact divergent and invariant subgraphs ([`TraceDiff`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/provenance/trace.py)).

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.core.scm`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/scm.py) & [`ewm_engine.simulation.counterfactual`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/simulation/counterfactual.py).
* **Executable Example:** [`examples/counterfactual_warehouse_reroute.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/examples/counterfactual_warehouse_reroute.py).
* **Automated Contract Test:** [`tests/contract/test_scm_counterfactual_contract.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/tests/contract/test_scm_counterfactual_contract.py).

---

### Guarantee 5: Regulatory Compliance & Governance Readiness

> **Executive Statement:** *Documentation and technical evidence automatically export into regulatory frameworks (NIST AI RMF 1.0, EU AI Act Annex IV).*

#### Why This Matters to the Enterprise
The regulatory landscape (EU AI Act, ISO/IEC 42001, US Executive Orders) is imposing strict technical documentation and risk mitigation duties on high-stakes automated decision systems. Manual compliance engineering is slow and error-prone.

#### How EWM Enforces It
* **Automated Annex IV Technical Export:** Both [`ModelCard`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/cards/models.py) and [`ScenarioCard`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/cards/models.py) include `.to_annex_iv()` methods that export system parameters, assumptions, and constraint profiles directly into the EU AI Act Annex IV technical documentation structure.
* **NIST AI RMF 1.0 Alignment:** Formal mapping across GOVERN, MAP, MEASURE, and MANAGE functions ([ADR-039](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-039-responsible-ai-trust-and-anti-overclaiming-enforcement.md)).
* **Human-in-the-Loop Safeguards:** Epistemic gates block automated execution when scenario parameters violate common support or identifiability.

#### Shipped Evidence & Runnable Proof
* **Governance Guide:** [`docs/responsible-ai.md`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/responsible-ai.md) & [`docs/governance/compliance-mapping.md`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/governance/compliance-mapping.md).
* **Automated Unit Test:** [`tests/unit/test_native_cards.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/tests/unit/test_native_cards.py) (testing `to_annex_iv()` export compliance).

---

### Guarantee 6: Rigorous Decision-Backtesting & Regret Bounds

> **Executive Statement:** *Policies are ranked by decision quality under deep uncertainty, not merely historical curve-fitting.*

#### Why This Matters to the Enterprise
A model that achieves 98% forecasting accuracy on peacetime data will recommend catastrophic policies during unexpected supply-chain shocks or interest-rate spikes. Enterprises must optimize for decision robustness, not just rollout accuracy.

#### How EWM Enforces It
* **Decision-Focused Learning (`ewm decide`):** Ranks candidate actions across multi-scenario ensembles using robust decision metrics: **Minimax Regret**, **Satisficing / Domain Criterion**, and **Info-Gap Analysis** ([ADR-041](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-041-decision-focused-scenario-discovery.md)).
* **Scenario Discovery (PRIM/CART):** Automates the Patient Rule Induction Method to isolate the precise hyper-dimensional vulnerability subspace where a policy fails, preventing blind spots.
* **Statistical Hypothesis Testing:** Pairwise scenario comparisons use Welch’s t-test and Mann-Whitney U to flag statistically significant differences ($p < 0.05$) while suppressing noisy trivial gains.

#### Shipped Evidence & Runnable Proof
* **Feature Module:** [`ewm_engine.decision.robustness`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/decision/robustness.py) & [`ewm_engine.decision.discovery`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/decision/discovery.py).
* **Executable CLI Command:** `ewm decide` & [`compare_scenarios()`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/evaluation/comparison.py).
* **Automated Contract Test:** [`tests/contract/test_decision_contract.py`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/tests/contract/test_decision_contract.py).

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
