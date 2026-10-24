# Internal Pre-Registration & Analysis Plan (`EWM-PREREG-2026-10-23-LADDER`)

> [!IMPORTANT]
> **Timestamped Internal Pre-Registration (`2026-10-23T15:32:15-03:00`):**
> This analysis plan was locked prior to executing the comparative ladder and ablation study. Its machine-readable counterpart is committed at [`docs/results/artifacts/pre_registration_comparative_ladder.json`](../results/artifacts/pre_registration_comparative_ladder.json). No hyperparameters, thresholds, or structural equations are tuned to favor EWM; negative findings where simpler baselines match or beat EWM are first-class deliverables.

---

## 1. Primary Capabilities & Distinct Evidence Standards

The evaluation strictly separates four capabilities and **never collapses them into a single scalar score**:

| Capability | Primary Metrics | Evidence Standard & Gate |
| :--- | :--- | :--- |
| **1. Prediction** | 1-step `call_volume_mae`, `net_cash_mae`, `crps`, 90% interval coverage ($\hat{\text{Cov}}_{0.90}$), and `calibration_score` | Evaluated strictly on bitemporal point-in-time views ($\text{event\_time} \le t_{\text{dec}} \land \text{knowledge\_time} \le t_{\text{dec}}$). Admits `STATISTICAL_ASSOCIATION`. |
| **2. Simulation** | Multi-step rollout `trajectory_nmae`, error growth slope $\gamma_H$, `internal_coherence_score` ($\mathcal{C}_{\text{pres}}$), `reality_fidelity_score`, and parameter elasticity | Distinguishes **internal coherence** (conservation & bound satisfaction) from **external reality fidelity** (agreement with ground truth). |
| **3. Intervention** | `intervention_effect_mae`, `intervention_effect_bias`, `sign_recovery_rate` across $do(\tau_{\text{auth}}, N_{\text{agents}}, \text{SMS})$ | Gated by `validate_dependency_for_query(..., DependencyQueryKind.INTERVENTION)`. Models with `STATISTICAL_ASSOCIATION` are refused (`refusal_gate_triggered = True`); ungated plug-in error is also recorded diagnostically. |
| **4. Planning / Decision** | `decision_regret_usd` ($U^*(\pi^*) - U^*(\hat{\pi})$), `policy_rank_spearman_rho`, and `sla_constraint_violation_steps` | Evaluated against the ground-truth optimal feasible policy $\pi^*$ under staffing capacity and SLA constraints. |

---

## 2. Comparative Ladder & Decision Baselines

All models are evaluated on identical inputs, `SeedSequence`-derived evaluation seeds ($M=10$ full / $M=5$ smoke), and bitemporal splits:

1. **Rung 1 (`rung1_simple_statistical`):** Historical rolling-mean and `AR(1)` linear autoregressive baseline (`STATISTICAL_ASSOCIATION`).
2. **Rung 2 (`rung2_standard_ml_gbt`):** Gradient Boosted Regression Trees (`GradientBoostedStumpEnsemble`) on raw point-in-time state features (`STATISTICAL_ASSOCIATION`).
3. **Rung 3 (`rung3_ml_engineered_features`):** Gradient Boosted Regression Trees + domain-engineered features (lagged decline velocity, queue utilization ratio, nonlinear interactions, rolling volatility) (`STATISTICAL_ASSOCIATION`).
4. **Rung 4 (`rung4_explicit_dynamics_predict_only`):** Explicit-dynamics queueing and cohort simulator **without** causal `do()` intervention surgery (models passive state evolution under observational policy correlation).
5. **Rung 5 (`rung5_full_ewm`):** Full EWM with 4-layer structural cascade dynamics, `do()` intervention operator, runtime `ContextSnapshot` ($K_t$) discovery, $\Pi_{\mathcal{C}}$ feasible projection, and split-conformal calibration.
6. **Four Architectural Decision Baselines:** `rules-only`, `learned-only`, `static hybrid`, and `hybrid + runtime context discovery`.

---

## 3. Pre-Registered Hypotheses & Negative Outcomes

| ID | Category | Pre-Registered Hypothesis |
| :--- | :--- | :--- |
| **`H1`** | Conditional Positive | `Rung 5 (Full EWM)` achieves lower decision regret than `Rungs 1–4` when policy interventions excite the 4-layer Fraud $\to$ Declines $\to$ Call-Center $\to$ Churn cascade under confounding fraud waves. |
| **`N1`** | **Negative Result for EWM** | `Rung 3 (ML + Engineered Features)` matches or beats `Rung 5 (Full EWM)` on **1-step observational prediction MAE** (`PREDICTION`) at lower compute and specification cost. |
| **`N2`** | **Negative Result for EWM** | `Rung 4 (Explicit Dynamics without Interventions / Conventional DES)` matches `Rung 5 (Full EWM)` on **passive trajectory simulation** (`SIMULATION`) under a fixed historical policy. |
| **`N3`** | **Prediction-vs-Decision Gap** | Improving 1-step observational prediction (`Rung 1 -> Rung 3`) does **not** reduce decision regret under policy confounding; conversely, a structural model with an unmodeled constant facility-cost offset has worse state prediction MAE yet achieves `$0.00` decision regret. |
| **`N4`** | **Negative Result for EWM** | In stationary single-layer regimes without mid-rollout context shifts or cascade overloads, `Rung 5` is **not cost-justified** ($>3\times$ latency and $>4\times$ specification complexity for `$0.00` regret reduction). |
| **`F1`** | **Construct Validity Cautionary Case** | Optimizing a misspecified objective (`obj_narrow_fraud_loss_minimization` or `obj_short_horizon_cash_myopia`) inside the exact ground-truth EWM (`Rung 5`) produces severe true enterprise decision regret ($>\$40{,}000$). |
