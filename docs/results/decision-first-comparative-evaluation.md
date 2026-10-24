# Decision-First Comparative Evaluation

> **Epistemic Status (`CONTROLLED_SYNTHETIC_KNOWN_MECHANISMS_NOT_REAL_WORLD_EFFICACY`):**
> Capability demonstrated in a controlled, synthetic environment with known mechanisms and ground-truth parameters — NOT proven efficacy in a real operation.
>
> **Pre-Registration Lock:** `EWME-PREREG-20261023-LADDER` ([`docs/scientific/pre_registration_comparative_ladder.md`](../scientific/pre_registration_comparative_ladder.md)) | **Master Seed:** `42` | **Evaluated Seeds ($M$):** `10` (`[3885454580, 3581871814, 3100435380, 1970475875, 2832263878, 545717655, 616287349, 2828675968, 2503035058, 3353190387]`) | **Report Fingerprint:** `0553d5286516e8cc565a9aecde654c46ac0ca3c72f1adf53bbe46527b453428b`

---

## 1. Executive Summary & Falsifiable Thesis Test

This report operationalizes the central empirical question of the **Enterprise World Model Engine**:

> *Under which conditions does explicitly modeling the dynamics of a corporate system let you decide better than simply predicting its outcomes?*

Rather than collapsing evaluation into a single composite benchmark score, we separate and measure **four distinct capabilities**—**Prediction**, **Simulation**, **Intervention**, and **Planning/Decision**—across a 5-rung comparative ladder, 4 architectural baselines, and 6 single-component ablations on identical bitemporal splits (`BitemporalLedger.as_of(t)`).

### Key Findings at a Glance

1. **Headline Result — The Prediction-vs-Decision Gap (Direction 1: Better Prediction $\not\Rightarrow$ Better Decision):** Climbing from **Rung 1 (Simple Statistical)** to **Rung 3 (ML + Engineered Features)** reduces 1-step call-arrival forecast MAE by **91.5%** (`25.60` $\to$ `2.18`), yet produces **\$0.00 improvement in decision regret** (regret remains **\$124,283.57** per episode vs. **Rung 5 Full EWM's \$0.00**). Observational feature engineering improves 1-step forecasting but cannot identify the multi-layer operational cascade triggered by an aggressive fraud-rule intervention $\text{do}(\tau = 0.40)$.
2. **Headline Result — The Prediction-vs-Decision Gap (Direction 2: Moderate State Error $\not\Rightarrow$ Bad Decision):** Conversely, **Arch Rules-Only** (which omits a constant \$450/step fixed facility overhead) suffers **13.3x higher net-cash prediction MAE** than **Rung 3** (`\$8,023.66` vs. `\$604.06`), yet achieves **\$0.00 decision regret** (`1.000` Spearman policy rank correlation) because a constant additive state offset leaves all marginal action-utility gradients intact.
3. **First-Class Negative Results (`N1`-`N4`):** Standard ML with engineered lag features (**Rung 3**) matches or beats **Rung 5 (Full EWM)** on 1-step observational prediction at $4\times$ lower specification burden (`N1`); conventional predict-only queueing DES (**Rung 4**) achieves 100% internal coherence and matches **Rung 5** on passive rollouts without interventions (`N2`); residual lag features and risk-neutral UQ add zero decision-regret benefit in ablation (`N3`); and in stationary single-layer regimes, **Rung 5** is not cost-justified (`N4`).
4. **Objective Construct Validity (Cautionary Case):** Even inside the **exact ground-truth EWM (Rung 5)** with zero dynamics error, optimizing a narrow departmental KPI (`obj_narrow_fraud_loss_minimization`) selects the aggressive uncoordinated policy ($\tau = 0.40$, 32 agents), causing **10 SLA breach steps** and **>\$22,000 in true enterprise decision regret** (`F1`).

---

## 2. Section A — The Four Separated Capabilities & Evidence Standards

| Capability | Query Class | Evidence Standard | Reported Metrics (Never Collapsed) |
| :--- | :--- | :--- | :--- |
| **1. Prediction** | `observational_prediction` ($P(O_{t+1} \mid \mathcal{I}_{\tau_t})$) | Point-in-time bitemporal split (`as_of(t)`) + proper scoring + empirical coverage | Call Arrival MAE, Call RMSE, Net Cash MAE (\$), CRPS, 90% Empirical Coverage, 2-bin Quantile ACE |
| **2. Simulation** | `state_evolution` ($P(S_{1:H} \mid S_0, K_{0:H}, E_{1:H})$) | Multi-step rollout vs. known mechanism + **Internal Coherence vs. Reality Fidelity** separation + parameter sensitivity | Internal Coherence Pass Rate (conservation/bounds), External Trajectory NMAE, Error Growth Slope $\gamma_H$, Parameter Sensitivity Index |
| **3. Intervention** | `interventional_do` ($P(S_{1:H} \mid \text{do}(a), K_t)$) | Ground-truth ATE recovery gated by `DependencyEvidenceNature` & `validate_dependency_for_query` | Evidence Nature, Causal Refusal Gate status, Diagnostic Ungated Utility ATE MAE (\$), Wait ATE MAE (s), Relative Bias |
| **4. Planning / Decision** | `decision_planning` ($\arg\max_{\pi \in \Pi_{\mathcal{C}}} \mathbb{E}[U(\pi)]$) | Constrained policy regret against ground-truth optimal policy $\pi^*$ | Modal Selected Policy, Decision Regret (\$), Spearman Policy Rank Correlation $\rho$, Constraint Violation Rate |

---

## 3. Section B — Comparative Ladder & Architectural Baselines

### 3.1 The 5-Rung Comparative Ladder (Identical Bitemporal Splits & Seeds)

#### Capability 1: Prediction (1-Step Point-in-Time Forecasting & Calibration)

| Rung / Model | Call Arrival MAE (95% CI) | Call RMSE | Net Cash MAE ($) | Call CRPS | 90% Coverage | Calibration ACE | MC SE / Model Spread |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rung 1: Simple Statistical (Rolling Mean + AR(1))** | `25.60 +/- 0.22` | `27.86` | `$8,088.4` | `19.97` | `26.0%` | `0.639` | `0.12` / `19.01` |
| **Rung 2: Standard ML (Gradient Boosted Trees on Raw Features)** | `8.44 +/- 0.23` | `9.73` | `$3,680.4` | `5.77` | `78.0%` | `0.121` | `0.13` / `12.97` |
| **Rung 3: ML + Engineered Domain Features (Lag Velocity & Utilization)** | `2.18 +/- 0.31` | `2.73` | `$604.1` | `1.57` | `86.7%` | `0.113` | `0.17` / `4.20` |
| **Rung 4: Explicit-Dynamics World Model WITHOUT Interventions (Predict-Only DES)** | `3.79 +/- 0.40` | `4.31` | `$521.2` | `2.79` | `58.7%` | `0.307` | `0.21` / `4.05` |
| **Rung 5: Full EWM (4-Layer Dynamics + Causal do() + Context K_t + Conformal UQ)** | `3.32 +/- 0.37` | `3.85` | `$521.2` | `2.37` | `73.3%` | `0.190` | `0.19` / `4.50` |

#### Capability 2: Simulation (Internal Coherence vs. External Trajectory Fidelity)

| Rung / Model | Internal Coherence Pass Rate | External Trajectory NMAE (95% CI) | Error Growth Slope | Param Sensitivity Index | Coherence vs. Reality Fidelity Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Rung 1: Simple Statistical (Rolling Mean + AR(1))** | `0.0%` | `1.052 +/- 0.004` | `0.0837` | `0.000` | INTERNALLY INCOHERENT: Violates closed-pool customer or queue conservation (coherence pass rate=0%, external NMAE=1.052). |
| **Rung 2: Standard ML (Gradient Boosted Trees on Raw Features)** | `0.0%` | `1.092 +/- 0.004` | `0.0873` | `0.000` | INTERNALLY INCOHERENT: Violates closed-pool customer or queue conservation (coherence pass rate=0%, external NMAE=1.092). |
| **Rung 3: ML + Engineered Domain Features (Lag Velocity & Utilization)** | `0.0%` | `1.092 +/- 0.004` | `0.0873` | `0.000` | INTERNALLY INCOHERENT: Violates closed-pool customer or queue conservation (coherence pass rate=0%, external NMAE=1.092). |
| **Rung 4: Explicit-Dynamics World Model WITHOUT Interventions (Predict-Only DES)** | `100.0%` | `0.597 +/- 0.004` | `0.0595` | `0.650` | INTERNALLY COHERENT YET EXTERNALLY BIASED: Achieves 100% internal conservation/bound coherence, but suffers external trajectory NMAE=0.597 due to missing causal intervention/shift dynamics. |
| **Rung 5: Full EWM (4-Layer Dynamics + Causal do() + Context K_t + Conformal UQ)** | `100.0%` | `0.109 +/- 0.002` | `0.0060` | `0.650` | COHERENT & HIGH-FIDELITY: Satisfies 100% internal conservation invariants and tracks external reality (trajectory NMAE=0.109). |

#### Capability 3: Intervention (Causal `do()` ATE Recovery & Refusal Gate)

| Rung / Model | Evidence Nature | Causal Gate Refused? | Diagnostic Ungated Utility ATE MAE ($) | Diagnostic Ungated Wait ATE MAE (s) | Diagnostic ATE Relative Bias |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rung 1: Simple Statistical (Rolling Mean + AR(1))** | `statistical_association` | **YES (Refused)** | `$72,678.06` | `59.55s` | `117.0%` |
| **Rung 2: Standard ML (Gradient Boosted Trees on Raw Features)** | `statistical_association` | **YES (Refused)** | `$72,678.06` | `55.55s` | `117.0%` |
| **Rung 3: ML + Engineered Domain Features (Lag Velocity & Utilization)** | `statistical_association` | **YES (Refused)** | `$72,426.25` | `55.55s` | `116.6%` |
| **Rung 4: Explicit-Dynamics World Model WITHOUT Interventions (Predict-Only DES)** | `statistical_association` | **YES (Refused)** | `$72,678.06` | `59.55s` | `117.0%` |
| **Rung 5: Full EWM (4-Layer Dynamics + Causal do() + Context K_t + Conformal UQ)** | `experimentally_supported` | **NO (Admitted)** | `$0.00` | `0.00s` | `0.0%` |

#### Capability 4: Planning / Decision (Constrained Policy Regret & Rank Correlation)

| Rung / Model | Modal Selected Policy | Decision Regret USD (95% CI) | Median (IQR) | Policy Rank Corr (Spearman) | Constraint Violation Rate | Cohen's $d_z$ vs Rung 5 | Holm $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rung 1: Simple Statistical (Rolling Mean + AR(1))** | `myopic_aggressive_tau_0.40_agents_32` | `$124283.57 +/- 557.18` | `$124,347.3` (`$1,210.7`) | `-0.300` | `0.0%` | `135.95` | `0.0000` |
| **Rung 2: Standard ML (Gradient Boosted Trees on Raw Features)** | `myopic_aggressive_tau_0.40_agents_32` | `$124283.57 +/- 557.18` | `$124,347.3` (`$1,210.7`) | `-0.300` | `0.0%` | `135.95` | `0.0000` |
| **Rung 3: ML + Engineered Domain Features (Lag Velocity & Utilization)** | `myopic_aggressive_tau_0.40_agents_32` | `$124283.57 +/- 557.18` | `$124,347.3` (`$1,210.7`) | `-0.300` | `0.0%` | `135.95` | `0.0000` |
| **Rung 4: Explicit-Dynamics World Model WITHOUT Interventions (Predict-Only DES)** | `myopic_aggressive_tau_0.40_agents_32` | `$124283.57 +/- 557.18` | `$124,347.3` (`$1,210.7`) | `-0.300` | `0.0%` | `135.95` | `0.0000` |
| **Rung 5: Full EWM (4-Layer Dynamics + Causal do() + Context K_t + Conformal UQ)** | `coordinated_ewm_tau_0.50_agents_42_sms` | `$0.00 +/- 0.00` | `$0.0` (`$0.0`) | `1.000` | `0.0%` | `0.00` | `ref` |

### 3.2 The 4 Architectural Baselines (Cross-Referenced with Enterprise Suite A-L)

#### Capability 1: Prediction (1-Step Point-in-Time Forecasting & Calibration)

| Rung / Model | Call Arrival MAE (95% CI) | Call RMSE | Net Cash MAE ($) | Call CRPS | 90% Coverage | Calibration ACE | MC SE / Model Spread |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arch Baseline: Rules-Only (Deterministic Structural + Fixed Overhead Offset)** | `19.78 +/- 0.15` | `22.18` | `$8,023.7` | `19.78` | `0.0%` | `0.900` | `0.08` / `0.00` |
| **Arch Baseline: Learned-Only (Observational Ridge Regressor)** | `4.50 +/- 0.47` | `5.44` | `$1,211.4` | `3.32` | `68.0%` | `0.256` | `0.26` / `5.50` |
| **Arch Baseline: Static Hybrid (Structural + Residual, Fixed Context K_0)** | `3.79 +/- 0.40` | `4.31` | `$521.2` | `2.79` | `58.7%` | `0.307` | `0.21` / `4.05` |
| **Arch Baseline: Hybrid + Runtime Context Discovery (Full EWM Equivalent)** | `3.32 +/- 0.37` | `3.85` | `$521.2` | `2.37` | `73.3%` | `0.190` | `0.19` / `4.50` |

#### Capability 2: Simulation (Internal Coherence vs. External Trajectory Fidelity)

| Rung / Model | Internal Coherence Pass Rate | External Trajectory NMAE (95% CI) | Error Growth Slope | Param Sensitivity Index | Coherence vs. Reality Fidelity Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Arch Baseline: Rules-Only (Deterministic Structural + Fixed Overhead Offset)** | `100.0%` | `0.430 +/- 0.003` | `0.0417` | `0.650` | INTERNALLY COHERENT YET EXTERNALLY BIASED: Achieves 100% internal conservation/bound coherence, but suffers external trajectory NMAE=0.430 due to missing causal intervention/shift dynamics. |
| **Arch Baseline: Learned-Only (Observational Ridge Regressor)** | `0.0%` | `1.092 +/- 0.004` | `0.0873` | `0.000` | INTERNALLY INCOHERENT: Violates closed-pool customer or queue conservation (coherence pass rate=0%, external NMAE=1.092). |
| **Arch Baseline: Static Hybrid (Structural + Residual, Fixed Context K_0)** | `100.0%` | `0.434 +/- 0.003` | `0.0409` | `0.650` | INTERNALLY COHERENT YET EXTERNALLY BIASED: Achieves 100% internal conservation/bound coherence, but suffers external trajectory NMAE=0.434 due to missing causal intervention/shift dynamics. |
| **Arch Baseline: Hybrid + Runtime Context Discovery (Full EWM Equivalent)** | `100.0%` | `0.109 +/- 0.002` | `0.0060` | `0.650` | COHERENT & HIGH-FIDELITY: Satisfies 100% internal conservation invariants and tracks external reality (trajectory NMAE=0.109). |

#### Capability 3: Intervention (Causal `do()` ATE Recovery & Refusal Gate)

| Rung / Model | Evidence Nature | Causal Gate Refused? | Diagnostic Ungated Utility ATE MAE ($) | Diagnostic Ungated Wait ATE MAE (s) | Diagnostic ATE Relative Bias |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arch Baseline: Rules-Only (Deterministic Structural + Fixed Overhead Offset)** | `business_rule_mechanism` | **NO (Admitted)** | `$0.00` | `0.00s` | `0.0%` |
| **Arch Baseline: Learned-Only (Observational Ridge Regressor)** | `statistical_association` | **YES (Refused)** | `$72,678.06` | `59.55s` | `117.0%` |
| **Arch Baseline: Static Hybrid (Structural + Residual, Fixed Context K_0)** | `experimentally_supported` | **NO (Admitted)** | `$3,728.51` | `3.45s` | `6.0%` |
| **Arch Baseline: Hybrid + Runtime Context Discovery (Full EWM Equivalent)** | `experimentally_supported` | **NO (Admitted)** | `$0.00` | `0.00s` | `0.0%` |

#### Capability 4: Planning / Decision (Constrained Policy Regret & Rank Correlation)

| Rung / Model | Modal Selected Policy | Decision Regret USD (95% CI) | Median (IQR) | Policy Rank Corr (Spearman) | Constraint Violation Rate | Cohen's $d_z$ vs Rung 5 | Holm $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arch Baseline: Rules-Only (Deterministic Structural + Fixed Overhead Offset)** | `coordinated_ewm_tau_0.50_agents_42_sms` | `$0.00 +/- 0.00` | `$0.0` (`$0.0`) | `1.000` | `0.0%` | `0.00` | `1.0000` |
| **Arch Baseline: Learned-Only (Observational Ridge Regressor)** | `myopic_aggressive_tau_0.40_agents_32` | `$124283.57 +/- 557.18` | `$124,347.3` (`$1,210.7`) | `-0.300` | `0.0%` | `135.95` | `0.0000` |
| **Arch Baseline: Static Hybrid (Structural + Residual, Fixed Context K_0)** | `coordinated_ewm_tau_0.50_agents_42_sms` | `$0.00 +/- 0.00` | `$0.0` (`$0.0`) | `1.000` | `0.0%` | `0.00` | `1.0000` |
| **Arch Baseline: Hybrid + Runtime Context Discovery (Full EWM Equivalent)** | `coordinated_ewm_tau_0.50_agents_42_sms` | `$0.00 +/- 0.00` | `$0.0` (`$0.0`) | `1.000` | `0.0%` | `0.00` | `1.0000` |

---

## 4. Section B2 — Enterprise Benchmark Suite De-Scripting Audit (Families A-L)

Every metric and `phenomenon_surfaced` narrative across all 12 families (`A`-`L`) in `benchmarks/families/enterprise_families.py` was audited and de-scripted so that 100% of published numbers in [`docs/results/artifacts/enterprise_benchmark_suite.json`](artifacts/enterprise_benchmark_suite.json) and [`docs/research/enterprise-benchmark-suite.md`](../research/enterprise-benchmark-suite.md) are computed at runtime from actual state/trajectory executions:

| Benchmark Family | Metric / Field | Before Classification | After Classification | Source Line (`enterprise_families.py`) |
| :--- | :--- | :--- | :--- | :---: |
| **A (DirectDynamics)** | `CAUSAL_CORRECTNESS` | `SCRIPTED (1.0 else 0.65)` | `MEASURED (_compute_causal_correctness)` | `L242 -> L298` |
| **A (DirectDynamics)** | `CALIBRATION` | `SCRIPTED (1.0 else 0.75)` | `MEASURED (_compute_empirical_calibration)` | `L245 -> L301` |
| **A (DirectDynamics)** | `LONG_HORIZON_ERROR_GROWTH` | `SCRIPTED (mae * 0.1)` | `MEASURED (_compute_error_growth_slope)` | `L246 -> L304` |
| **A (DirectDynamics)** | `DECISION_REGRET` | `SCRIPTED (mae * 0.5)` | `MEASURED (_compute_policy_regret_from_predictions)` | `L249 -> L305` |
| **A (DirectDynamics)** | `ABSTENTION_QUALITY` | `SCRIPTED (1.0 literal)` | `MEASURED (_compute_abstention_f1)` | `L250 -> L310` |
| **A (DirectDynamics)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE ('MAE > 2.0')` | `MEASURED (dynamic f-string from rules_mae)` | `L305 -> L393` |
| **B (HiddenWorkflowCascades)** | `TRACE_PROVENANCE_COMPLETENESS` | `SCRIPTED (0.0/0.85/0.95/1.0)` | `MEASURED (_compute_trace_completeness)` | `L470 -> L570` |
| **B (HiddenWorkflowCascades)** | `CALIBRATION & ERROR_GROWTH` | `SCRIPTED (0.88/0.35, mae*0.15)` | `MEASURED (_compute_empirical_calibration / slope)` | `L524 -> L571` |
| **B (HiddenWorkflowCascades)** | `DECISION_REGRET & ABSTENTION` | `SCRIPTED (mae*0.4, 0.90/0.20)` | `MEASURED (policy regret & abstention F1)` | `L530 -> L588` |
| **B (HiddenWorkflowCascades)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE` | `MEASURED (dynamic f-string from learned_mae/fid)` | `L593 -> L702` |
| **C (RuntimeConfiguration)** | `pre_shift_mae & TRACE` | `SCRIPTED (0.25/1.35, 1.0/0.5/0.0)` | `MEASURED (k0_errors mean & trace edges)` | `L761/L824 -> L896` |
| **C (RuntimeConfiguration)** | `CALIBRATION / GROWTH / REGRET` | `SCRIPTED (0.90/0.40, mae*0.2, mae*0.5)` | `MEASURED (empirical calibration, slope, regret)` | `L833 -> L900` |
| **C (RuntimeConfiguration)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE ('MAE > 38')` | `MEASURED (dynamic f-string from static_mae/ctx_mae)` | `L908 -> L1043` |
| **D (DeploymentShift)** | `TRACE / CALIBRATION / GROWTH` | `SCRIPTED (1.0/0.5/0.0, 0.85/0.35, *0.25)` | `MEASURED (trace edges, steady calibration, slope)` | `L1113/L1161 -> L1519` |
| **D (DeploymentShift)** | `DECISION_REGRET & ABSTENTION` | `SCRIPTED (steady*0.6, 0.90/0.30)` | `MEASURED (policy regret & abstention F1)` | `L1167 -> L1540` |
| **D (DeploymentShift)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE` | `MEASURED (dynamic f-string from peak/steady/rec)` | `L1242 -> L1713` |
| **E (PartialObservability)** | `TRACE / GROWTH / RECOVERY` | `SCRIPTED (0.95/0.45, mae*0.18, 1.0/H)` | `MEASURED (trace records, slope, step threshold)` | `L1445/L1466 -> L1942` |
| **E (PartialObservability)** | `DECISION_REGRET & ABSTENTION` | `SCRIPTED (mae*0.45, 0.85/0.35)` | `MEASURED (policy regret & abstention F1)` | `L1471 -> L1950` |
| **E (PartialObservability)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE ('>75%')` | `MEASURED (dynamic f-string from rel_red_e%)` | `L1534 -> L2054` |
| **F (LongHorizonBusinessOp)** | `LEARNED_ONLY rollout step` | `SCRIPTED (-0.35 / +2.5 drift offset)` | `MEASURED (quadratic regressor extrapolation)` | `L1695 -> L2216` |
| **F (LongHorizonBusinessOp)** | `CALIBRATION / REGRET / TRACE` | `SCRIPTED (0.86/0.50, *0.5, 1.0/0.7/0.0)` | `MEASURED (interval coverage, cash regret, trace)` | `L1757 -> L2301` |
| **F (LongHorizonBusinessOp)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE` | `MEASURED (dynamic f-string from slopes)` | `L1831 -> L2409` |
| **G (MultiAgentEconomy)** | `LEARNED_ONLY clearing vector` | `SCRIPTED (+ [0.12, 0.08, -0.05])` | `MEASURED (3 independent per-actor regressors)` | `L1979 -> L2558` |
| **G (MultiAgentEconomy)** | `CALIBRATION / GROWTH / REGRET` | `SCRIPTED (0.88/0.45, *0.12, *0.6)` | `MEASURED (empirical calibration, slope, regret)` | `L2023 -> L2627` |
| **G (MultiAgentEconomy)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE` | `MEASURED (dynamic f-string from pass rates)` | `L2097 -> L2732` |
| **H (CausalCounterfactual)** | `CALIBRATION / GROWTH / REGRET` | `SCRIPTED (0.90/0.30, *0.1, *0.5)` | `MEASURED (empirical calibration, slope, regret)` | `L2281 -> L2902` |
| **H (CausalCounterfactual)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE ('MAE > 70')` | `MEASURED (dynamic f-string from learned_cf_h)` | `L2354 -> L3002` |
| **I (ConservationStress)** | `LEARNED_ONLY vault transfer` | `SCRIPTED (vb + req * 1.04)` | `MEASURED (unconstrained polynomial ridge)` | `L2946 -> L3267` |
| **I (ConservationStress)** | `GROWTH / REGRET / TRACE` | `SCRIPTED (mae*0.1, mae*0.5, 1.0/0.0)` | `MEASURED (slope, vault regret, trace records)` | `L2981 -> L3292` |
| **J (UncertaintyCalibration)** | `LEARNED/STATIC pred_sigma` | `SCRIPTED (homo_std*0.65, sigma*0.78)` | `MEASURED (unmodified homo_std & K_0 sigma)` | `L3158 -> L3529` |
| **J (UncertaintyCalibration)** | `CAUSAL / GROWTH / REGRET` | `SCRIPTED (0.90, mae*0.1, cov_err*15)` | `MEASURED (causal score, slope, q90 safety regret)` | `L3209 -> L3560` |
| **J (UncertaintyCalibration)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE ('~55%', '~75%')` | `MEASURED (dynamic f-string from empirical_coverage)` | `L3288 -> L3669` |
| **K (OODRegimeShift)** | `CAUSAL / GROWTH / REGRET` | `SCRIPTED (0.75/0.40, *0.2, *0.5)` | `MEASURED (causal score, slope, fallback regret)` | `L3450 -> L3846` |
| **K (OODRegimeShift)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE` | `MEASURED (dynamic f-string from ctx_ood_mae_k)` | `L3525 -> L3938` |
| **L (DecisionQuality)** | `Policy selection & pred_err` | `SCRIPTED ('pi_speculative', 2.1+noise)` | `MEASURED (confounded ridge fit + argmax + Spearman)` | `L3639 -> L4095` |
| **L (DecisionQuality)** | `CALIBRATION / GROWTH / ABST` | `SCRIPTED (0.88/0.45, *0.1, 0.90/0.30)` | `MEASURED (empirical calibration, slope, abstention)` | `L3688 -> L4156` |
| **L (DecisionQuality)** | `phenomenon_surfaced` | `ASSERTED NARRATIVE ('~2.4', '129.3')` | `MEASURED (dynamic f-string from measured MAE/regret)` | `L3764 -> L4251` |

---

## 5. Section C — Headline Result: The Prediction-vs-Decision Gap

The comparative evaluation demonstrates **both directions** of decoupling between observational prediction accuracy and downstream decision quality:

| Direction | Comparison | 1-Step Prediction Error | Realized Decision Regret (\$) | Policy Rank Correlation | Mechanism Explanation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Direction 1: Better Prediction $\not\Rightarrow$ Better Decision** | **Rung 1** $\to$ **Rung 3** | Call MAE drops **`25.60` $\to$ `2.18`** (`-91.5%`) | Regret unchanged: **`\$124,283.57` $\to$ `\$124,283.57`** (`\$0.00` gain) | `0.300` $\to$ `0.300` | Lag velocities and utilization ratios improve 1-step observational fit but do not identify the Layer-2/3/4 cascade of $\text{do}(\tau=0.40)$. |
| **Direction 2: Moderate State Error $\not\Rightarrow$ Bad Decision** | **Arch Rules-Only** vs. **Rung 3** | Net Cash MAE is **`\$8,023.66`** vs. Rung 3's **`\$604.06`** | Rules-Only regret is **`\$0.00`** vs. Rung 3's **`\$124,283.57`** | **`1.000`** vs. `0.300` | An unmodeled constant facility overhead (`+\$450/step`) shifts level predictions but cancels out in action-utility differences $U(\pi_1) - U(\pi_2)$. |

- **Direction 1 Synthesis:** Improving 1-step observational prediction from Rung 1 (call MAE=25.60) to Rung 3 (ML + Engineered Features, call MAE=2.18, a 91.5% error reduction) yields $0.00 reduction in decision regret (regret remains $124,283.57/episode vs Rung 5's $0.00), because observational feature engineering cannot identify the Layer-2/3/4 cascade of do(tau).
- **Direction 2 Synthesis:** Conversely, Arch Rules-Only suffers moderate state prediction error on net cash (MAE=$8,023.66/step from an unmodeled $450/step fixed facility overhead, vs Rung 3's $604.06/step), yet achieves $0.00 decision regret (rank correlation=1.00) because a constant additive offset preserves exact interventional action-utility gradients.

---

## 6. Section D — Single-Component Ablation Study on Rung 5

Each ablation removes **one** architectural component from **Rung 5 (Full EWM)** while holding all seeds, bitemporal splits, and candidate policies fixed. Critically, Ablations 4 and 6 expose components that add complexity without improving expected-utility decision regret:

### 6.1 Ablation Capability Breakdown (Compared Against Rung 5 Reference)

#### Capability 1: Prediction (1-Step Point-in-Time Forecasting & Calibration)

| Rung / Model | Call Arrival MAE (95% CI) | Call RMSE | Net Cash MAE ($) | Call CRPS | 90% Coverage | Calibration ACE | MC SE / Model Spread |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ablation 1: Rung 5 minus Multi-Layer Cascade Dynamics (Layer-1 Only)** | `3.79 +/- 0.40` | `4.31` | `$521.2` | `2.74` | `63.3%` | `0.261` | `0.21` / `4.50` |
| **Ablation 2: Rung 5 minus Causal do() Intervention Surgery** | `3.79 +/- 0.40` | `4.31` | `$521.2` | `2.74` | `63.3%` | `0.261` | `0.21` / `4.50` |
| **Ablation 3: Rung 5 minus Runtime Context Discovery (K_t Locked to K_0)** | `3.79 +/- 0.40` | `4.31` | `$521.2` | `2.74` | `63.3%` | `0.261` | `0.21` / `4.50` |
| **Ablation 4: Rung 5 minus Split-Conformal Uncertainty Quantification** | `3.79 +/- 0.40` | `4.31` | `$521.2` | `3.79` | `0.0%` | `0.900` | `0.21` / `0.00` |
| **Ablation 5: Rung 5 minus Feasible Manifold Projection (Pi_C Disabled)** | `3.79 +/- 0.40` | `4.31` | `$521.2` | `2.74` | `63.3%` | `0.261` | `0.21` / `4.50` |
| **Ablation 6: Rung 5 minus Engineered Residual Lag Features (Zero-Benefit Check)** | `3.79 +/- 0.40` | `4.31` | `$521.2` | `2.74` | `63.3%` | `0.261` | `0.21` / `4.50` |

#### Capability 2: Simulation (Internal Coherence vs. External Trajectory Fidelity)

| Rung / Model | Internal Coherence Pass Rate | External Trajectory NMAE (95% CI) | Error Growth Slope | Param Sensitivity Index | Coherence vs. Reality Fidelity Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Ablation 1: Rung 5 minus Multi-Layer Cascade Dynamics (Layer-1 Only)** | `100.0%` | `1.231 +/- 0.004` | `0.0917` | `0.042` | INTERNALLY COHERENT YET EXTERNALLY BIASED: Achieves 100% internal conservation/bound coherence, but suffers external trajectory NMAE=1.231 due to missing causal intervention/shift dynamics. |
| **Ablation 2: Rung 5 minus Causal do() Intervention Surgery** | `100.0%` | `0.597 +/- 0.004` | `0.0595` | `0.650` | INTERNALLY COHERENT YET EXTERNALLY BIASED: Achieves 100% internal conservation/bound coherence, but suffers external trajectory NMAE=0.597 due to missing causal intervention/shift dynamics. |
| **Ablation 3: Rung 5 minus Runtime Context Discovery (K_t Locked to K_0)** | `100.0%` | `0.434 +/- 0.003` | `0.0409` | `0.650` | INTERNALLY COHERENT YET EXTERNALLY BIASED: Achieves 100% internal conservation/bound coherence, but suffers external trajectory NMAE=0.434 due to missing causal intervention/shift dynamics. |
| **Ablation 4: Rung 5 minus Split-Conformal Uncertainty Quantification** | `100.0%` | `0.109 +/- 0.002` | `0.0060` | `0.650` | COHERENT & HIGH-FIDELITY: Satisfies 100% internal conservation invariants and tracks external reality (trajectory NMAE=0.109). |
| **Ablation 5: Rung 5 minus Feasible Manifold Projection (Pi_C Disabled)** | `0.0%` | `0.107 +/- 0.002` | `0.0057` | `0.650` | INTERNALLY INCOHERENT: Violates closed-pool customer or queue conservation (coherence pass rate=0%, external NMAE=0.107). |
| **Ablation 6: Rung 5 minus Engineered Residual Lag Features (Zero-Benefit Check)** | `100.0%` | `0.109 +/- 0.002` | `0.0060` | `0.650` | COHERENT & HIGH-FIDELITY: Satisfies 100% internal conservation invariants and tracks external reality (trajectory NMAE=0.109). |

#### Capability 3: Intervention (Causal `do()` ATE Recovery & Refusal Gate)

| Rung / Model | Evidence Nature | Causal Gate Refused? | Diagnostic Ungated Utility ATE MAE ($) | Diagnostic Ungated Wait ATE MAE (s) | Diagnostic ATE Relative Bias |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ablation 1: Rung 5 minus Multi-Layer Cascade Dynamics (Layer-1 Only)** | `business_rule_mechanism` | **NO (Admitted)** | `$72,678.06` | `59.55s` | `117.0%` |
| **Ablation 2: Rung 5 minus Causal do() Intervention Surgery** | `statistical_association` | **YES (Refused)** | `$72,678.06` | `59.55s` | `117.0%` |
| **Ablation 3: Rung 5 minus Runtime Context Discovery (K_t Locked to K_0)** | `experimentally_supported` | **NO (Admitted)** | `$3,728.51` | `3.45s` | `6.0%` |
| **Ablation 4: Rung 5 minus Split-Conformal Uncertainty Quantification** | `experimentally_supported` | **NO (Admitted)** | `$0.00` | `0.00s` | `0.0%` |
| **Ablation 5: Rung 5 minus Feasible Manifold Projection (Pi_C Disabled)** | `experimentally_supported` | **NO (Admitted)** | `$0.00` | `0.00s` | `0.0%` |
| **Ablation 6: Rung 5 minus Engineered Residual Lag Features (Zero-Benefit Check)** | `experimentally_supported` | **NO (Admitted)** | `$0.00` | `0.00s` | `0.0%` |

#### Capability 4: Planning / Decision (Constrained Policy Regret & Rank Correlation)

| Rung / Model | Modal Selected Policy | Decision Regret USD (95% CI) | Median (IQR) | Policy Rank Corr (Spearman) | Constraint Violation Rate | Cohen's $d_z$ vs Rung 5 | Holm $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ablation 1: Rung 5 minus Multi-Layer Cascade Dynamics (Layer-1 Only)** | `myopic_aggressive_tau_0.40_agents_32` | `$124283.57 +/- 557.18` | `$124,347.3` (`$1,210.7`) | `-0.300` | `0.0%` | `135.95` | `0.0000` |
| **Ablation 2: Rung 5 minus Causal do() Intervention Surgery** | `myopic_aggressive_tau_0.40_agents_32` | `$124283.57 +/- 557.18` | `$124,347.3` (`$1,210.7`) | `-0.300` | `0.0%` | `135.95` | `0.0000` |
| **Ablation 3: Rung 5 minus Runtime Context Discovery (K_t Locked to K_0)** | `coordinated_ewm_tau_0.50_agents_42_sms` | `$0.00 +/- 0.00` | `$0.0` (`$0.0`) | `1.000` | `0.0%` | `0.00` | `1.0000` |
| **Ablation 4: Rung 5 minus Split-Conformal Uncertainty Quantification** | `coordinated_ewm_tau_0.50_agents_42_sms` | `$0.00 +/- 0.00` | `$0.0` (`$0.0`) | `1.000` | `0.0%` | `0.00` | `1.0000` |
| **Ablation 5: Rung 5 minus Feasible Manifold Projection (Pi_C Disabled)** | `infeasible_overcapacity_tau_0.50_agents_150_sms` | `$71090.28 +/- 150.26` | `$71,159.7` (`$314.7`) | `1.000` | `100.0%` | `281.85` | `0.0000` |
| **Ablation 6: Rung 5 minus Engineered Residual Lag Features (Zero-Benefit Check)** | `coordinated_ewm_tau_0.50_agents_42_sms` | `$0.00 +/- 0.00` | `$0.0` (`$0.0`) | `1.000` | `0.0%` | `0.00` | `1.0000` |

---

## 7. Section E — Pre-Registered Negative Results (`N1`-`N4`)

In accordance with pre-registration [`EWME-PREREG-20261023-LADDER`](../scientific/pre_registration_comparative_ladder.md), negative results where simpler baselines match or beat the Full EWM are reported with equal prominence:

| ID | Negative Hypothesis | Capability Scope | Confirmed? | Measured Evidence | Scientific & Architectural Takeaway |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **`N1`** | **1-Step Observational Prediction Parity/Superiority of ML + Engineered Features** | `Prediction (1-step point-in-time forecasting)` | **YES** | Rung 3 (ML + Engineered Features) achieves 1-step call-arrival MAE=2.18 and net-cash MAE=$604.06, matching/beating Rung 5 (call MAE=3.32, net-cash MAE=$521.24) at lower specification burden (3 vs 12 components). | For pure 1-step observational forecasting under stationary conditions, standard ML with lagged domain features suffices; building an Enterprise World Model is not justified if the only requirement is short-horizon observational point prediction. |
| **`N2`** | **Passive State Simulation Sufficiency of Predict-Only Queueing DES (Rung 4)** | `Simulation (passive rollout without causal interventions)` | **YES** | Rung 4 (Predict-Only DES) achieves 100.0% internal conservation/capacity coherence (identical to Rung 5's 100.0%) and matches Rung 5 on pre-shift passive state evolution. | When no policy intervention do(a) or configuration shift occurs, a conventional discrete-event queueing simulator (Rung 4) is internally coherent and sufficient. |
| **`N3`** | **Prediction-vs-Decision Decoupling & Zero-Decision-Benefit Components** | `Prediction -> Decision translation & Ablations` | **YES** | Rung 1 -> Rung 3 cuts 1-step call MAE by 91.5% with $0.00 regret reduction; ablating engineered lag features (Ablation 6, regret=$0.00) and conformal UQ under risk-neutral expected utility (Ablation 4, regret=$0.00) causes 0.00 change in decision regret. | Better 1-step observational fit does not imply better decisions under confounding/cascades, and components that improve residual smoothing or interval coverage add zero decision value unless the decision rule is sensitive to tail risk or action gradients. |
| **`N4`** | **Stationary Single-Layer Regime Where Full EWM Is NOT Cost-Justified** | `Operational Cost / Viability Boundary` | **YES** | In the stationary single-layer regime (no secondary workflow/call-center cascade, fixed K_0), Rung 1 and Rung 2 achieve $0.00 decision regret with 1-2 specification components, whereas Rung 5 requires 12 specification components for $0.00 incremental regret reduction. | Full EWM is only cost-justified when cross-layer cascades, hard capacity constraints, and policy interventions/shifts interact; in stationary single-layer tasks, simple baselines win on benefit-per-unit-cost. |

---

## 8. Section F — Objective Construct Validity & Cautionary Misspecified-Objective Cases

A causally valid, zero-error Enterprise World Model **cannot protect an organization from a misspecified objective function**. Below, we hold the world model fixed at the exact ground-truth **Rung 5 (Full EWM)** and vary only the objective function optimized by the planner:

| Objective ID | Objective Name | Misspecified Proxy? | Selected Policy Inside Exact EWM (Rung 5) | Layer-1 Margin minus Fraud (\$) | Realized SLA Breach Steps | Realized Churned Customers | True Enterprise Utility (\$) | True Enterprise Decision Regret (\$) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `obj_balanced_enterprise_utility` | **Balanced Multi-Layer Enterprise Utility (Reference)** | **NO** | `coordinated_ewm_tau_0.50_agents_42_sms` | `\$491,910.31` | `0.0` | `515.2` | `\$339,327.32` | **`\$0.00`** |
| `obj_narrow_fraud_loss_minimization` | **Cautionary Case A: Narrow Fraud-Team Proxy (Layer-1 Margin minus Chargebacks)** | **YES** | `myopic_aggressive_tau_0.40_agents_32` | `\$472,138.84` | `10.0` | `1507.8` | `\$215,638.17` | **`\$123,689.15`** |
| `obj_short_horizon_cash_myopia` | **Cautionary Case B: Short-Horizon Cost-Center Myopia (Omits Deferred Churn LTV)** | **YES** | `myopic_aggressive_tau_0.40_agents_32` | `\$472,138.84` | `10.0` | `1507.8` | `\$215,638.17` | **`\$123,689.15`** |
| `obj_sla_constrained_multi_objective` | **SLA-Constrained Multi-Objective Utility (Zero SLA Breach Hard Constraint)** | **NO** | `coordinated_ewm_tau_0.50_agents_42_sms` | `\$491,910.31` | `0.0` | `515.2` | `\$339,327.32` | **`\$0.00`** |

### Construct-Validity Threats Documented

- 1. Departmental Silo Proxy Gaming (Goodhart's Law): Optimizing Layer-1 fraud chargeback reduction in isolation externalizes massive costs onto Contact-Center Operations (SLA breaches) and Customer Retention (churn LTV).
- 2. Horizon Truncation Myopia: Evaluating policies over a 1-step cash window omits delayed customer churn reconciliation (6-step audit lag) and multi-step queue backlog accumulation.
- 3. Unpriced Contractual & Reputational Externalities: Omitting nonlinear SLA breach thresholds ($2,200/step above 120s wait) makes under-staffed policies appear artificially profitable.
- 4. Risk-Neutral Expectation Collapse: Optimizing expected mean utility ignores tail-risk queue blowups unless explicit SLA/CVaR constraints are enforced in the objective.

---

## 9. Section G — Cost, Complexity & Operational Viability ("Benefit per Unit Cost")

| Rung / Baseline | Mean Latency (ms) | Peak Memory (KB) | Spec Burden (Components) | Cascade Regret Reduction vs Rung 1 (\$) | Regret Reduction per ms Latency (\$/ms) | Regret Reduction per Spec Component (\$/comp) | Stationary Single-Layer Regret (\$) | Cost-Justified in Stationary Regime? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rung 1: Simple Statistical (Rolling Mean + AR(1))** | `2.14 ms` | `10.6 KB` | `1` | `\$0.00` | `\$0.00/ms` | `\$0.00` | `\$0.00` | **YES** |
| **Rung 2: Standard ML (Gradient Boosted Trees on Raw Features)** | `33.42 ms` | `42.4 KB` | `2` | `\$0.00` | `\$0.00/ms` | `\$0.00` | `\$0.00` | **YES** |
| **Rung 3: ML + Engineered Domain Features (Lag Velocity & Utilization)** | `163.01 ms` | `64.0 KB` | `3` | `\$0.00` | `\$0.00/ms` | `\$0.00` | `\$0.00` | **NO (Over-Engineered)** |
| **Rung 4: Explicit-Dynamics World Model WITHOUT Interventions (Predict-Only DES)** | `1.90 ms` | `48.2 KB` | `6` | `\$0.00` | `\$0.00/ms` | `\$0.00` | `\$0.00` | **NO (Over-Engineered)** |
| **Rung 5: Full EWM (4-Layer Dynamics + Causal do() + Context K_t + Conformal UQ)** | `58.67 ms` | `88.0 KB` | `12` | `\$124,283.57` | `\$2,118.51/ms` | `\$10,356.96` | `\$0.00` | **NO (Over-Engineered)** |
| **Arch Baseline: Rules-Only (Deterministic Structural + Fixed Overhead Offset)** | `1.69 ms` | `48.1 KB` | `8` | `\$124,283.57` | `\$73,752.67/ms` | `\$15,535.45` | `\$0.00` | **NO (Over-Engineered)** |
| **Arch Baseline: Learned-Only (Observational Ridge Regressor)** | `2.74 ms` | `25.3 KB` | `2` | `\$0.00` | `\$0.00/ms` | `\$0.00` | `\$0.00` | **YES** |
| **Arch Baseline: Static Hybrid (Structural + Residual, Fixed Context K_0)** | `1.83 ms` | `48.2 KB` | `10` | `\$124,283.57` | `\$67,808.54/ms` | `\$12,428.36` | `\$0.00` | **NO (Over-Engineered)** |
| **Arch Baseline: Hybrid + Runtime Context Discovery (Full EWM Equivalent)** | `58.60 ms` | `88.0 KB` | `12` | `\$124,283.57` | `\$2,121.04/ms` | `\$10,356.96` | `\$0.00` | **NO (Over-Engineered)** |

---

## 10. Reproducibility & Verification Commands

```bash
# Full 10-seed comparative evaluation + artifact regeneration
python scripts/run_comparative_evaluation.py --seed 42 --update-artifacts

# Fast 5-seed smoke verification
python scripts/run_comparative_evaluation.py --seed 42 --smoke

# Run unit & benchmark self-tests
pytest tests/unit/test_comparative_evaluation.py tests/benchmark/test_enterprise_benchmark_suite.py -q
```
