# Enterprise Simulation & Decision Maturity Model

> *"Organizations evolve from fragile spreadsheet intuition to continuously calibrated digital twins. This 5-level framework helps executive leadership assess their current decision maturity and plan their technological journey."*

---

## 1. The 5-Level Maturity Ladder

```
      Level 5: Continuously-Optimized
      (Calibrated Digital Twin, Active Inference, Self-Whitening Grey-Box Flywheel)
                        ▲
                        │
      Level 4: Prescriptive & Counterfactual
      (Pearl SCM Kernel, Twin-Rollout do(x), PRIM Scenario Discovery, Minimax Regret)
                        ▲
                        │
      Level 3: Simulated & Stochastic
      (Monte Carlo Ensembles, BCa Confidence Bands, What-If Branching, Invariant Checks)
                        ▲
                        │
      Level 2: Modeled & Constrained
      (Explicit Typed State, Relational Graphs, Hard Accounting Conservation Rules)
                        ▲
                        │
      Level 1: Ad-Hoc & Heuristic
      (Static Spreadsheets, Flaw of Averages, Uncoupled Departmental Guesswork)
```

---

## 2. Detailed Dimension Breakdown

| Dimension | Level 1: Ad-Hoc | Level 2: Modeled | Level 3: Simulated | Level 4: Prescriptive | Level 5: Continuously-Optimized |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Decision Paradigm** | Reactive intuition & static spreadsheets | Deterministic rule checking & business logic | Stochastic Monte Carlo what-if analysis | Causal counterfactual optimization & regret bounds | Active inference & receding-horizon control (MPC) |
| **State Representation** | Unlinked Excel cells, siloed databases | Typed relational graph entities (`WorldState`) | Stateful discrete-event trajectory history | Structural Causal Model (SCM) causal DAG | Hybrid heterogeneous graph state + neural residual embeddings |
| **Physics & Conservation** | Ignored; formulas easily violate physical limits | Hard algebraic invariant pre/post gates | Rollouts violating invariants auto-invalidated | Invariants enforced across factual + counterfactual | Hard architectural conservation layer (KKT/hPINN) |
| **Causal Maturity** | Observational correlation ($P(Y \mid X)$) | Rule-based associations | Scenario parameter sweeps | Pearl's Ladder Rung 2 & 3 ($do(x)$, Twin-Rollout) | Causal graph learning + symbolic equation discovery |
| **Uncertainty Quantification** | Single-point estimates ("flaw of averages") | Manual worst-case / best-case manual tables | BCa bootstrap CIs & quantile distributions | Conformal prediction & sequential E-values | Amortized Bayesian calibration (SBI) with TARP coverage |
| **Governance & Audit** | No version control; overwriting formulas | Git-tracked declarative YAML specs (`WSL`) | Seed-isolated deterministic PRNG replay | Mitchell Model Cards & auto EU AI Act Annex IV | SLSA L2+ cryptographic provenance on live telemetry |

---

## 3. The Evolutionary Journey

### Level 1 $\rightarrow$ Level 2: From Spreadsheets to Typed Models
* **The Symptom:** Different business units use conflicting Excel formulas; supply plans forecast negative inventory; cash flow forecasts do not reconcile with ERP balances.
* **The EWM Transition:** Adopt declarative World Specification Language ([WSL](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/concepts/wsl.md)) and [`WorldState`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/core/state.py). Define explicit entities, resources, and hard conservation constraints.
* **Immediate Value:** Eliminates impossible plans before simulation begins.

### Level 2 $\rightarrow$ Level 3: From Static Models to Stochastic Simulation
* **The Symptom:** Management evaluates plans under a single expected outcome; unexpected market volatility causes severe covenant or SLA breaches.
* **The EWM Transition:** Deploy [`SimulationEngine`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/simulation/engine.py) to run 1,000-sample Monte Carlo ensembles with seeded determinism. Apply [`compare_scenarios()`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/evaluation/comparison.py) to compute empirical quantiles ($p05, p50, p95$).
* **Immediate Value:** Replaces point guesses with certified distribution envelopes, exposing hidden downside tail risks.

### Level 3 $\rightarrow$ Level 4: From What-If to Causal Prescriptive Action
* **The Symptom:** Decision-makers run dozens of what-if scenarios but cannot determine which parameter combinations cause failure, or confuse correlation with policy efficacy.
* **The EWM Transition:** Promote graph state to a Structural Causal Model ([ADR-040](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-040-twin-rollout-counterfactual-scm-kernel.md)). Execute Twin-Rollout counterfactuals ($do(x)$). Run [`ewm decide`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/decision/discovery.py) using PRIM scenario discovery to isolate vulnerability subspaces and rank policies via Minimax Regret.
* **Immediate Value:** Mathematically guarantees that interventions work under identical historical shocks, bounding strategic regret.

### Level 4 $\rightarrow$ Level 5: Towards the Calibrated Digital Twin
* **The Symptom:** Simulated model parameters drift away from actual enterprise telemetry over time, requiring manual re-tuning.
* **The EWM Transition:** Implement amortized Simulation-Based Inference (`ewm calibrate`, [ADR-043](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-043-amortized-calibration-certified-coverage.md)) to continuously calibrate parameters against operational data with certified coverage. Deploy receding-horizon control (MPC) and the neuro-symbolic flywheel ([ADR-042](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/docs/adr/ADR-042-grey-box-conservation-flywheel.md)).
* **Immediate Value:** Creates a self-correcting, auditable Digital Twin of the Organization (DTO).

---

## 4. Executive Self-Assessment Diagnostic

Answer these five questions to benchmark your organization's decision maturity:

1. **How are physical and financial rules enforced in your forward forecasts?**
   * *A: Handled by individual spreadsheet authors (Level 1)*
   * *B: Enforced as hard, code-level invariant gates that reject invalid states (Level 2+)*
2. **When planning a major initiative, what kind of forecast does leadership review?**
   * *A: A single point estimate or best/base/worst table (Level 1–2)*
   * *B: A calibrated Monte Carlo probability distribution with explicit confidence intervals (Level 3+)*
3. **Can you prove that an outcome was caused by your policy rather than external luck?**
   * *A: No, we rely on observational regression or post-hoc correlation (Level 1–3)*
   * *B: Yes, we isolate causal interventions using structural models and counterfactual surgery (Level 4+)*
4. **How do you find the operational conditions under which your strategy fails?**
   * *A: Manual trial-and-error what-if guessing (Level 1–3)*
   * *B: Automated hyper-dimensional scenario discovery (PRIM/CART) (Level 4+)*
5. **How are model assumptions documented and audited for regulatory compliance?**
   * *A: Scattered in slide decks and internal memos (Level 1–2)*
   * *B: Cryptographically sealed with automated EU AI Act / NIST RMF exports (Level 4–5)*
