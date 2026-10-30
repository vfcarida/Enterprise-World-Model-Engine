# Fraud Cascade TCO Analysis

**Date**: 2026-10-30  
**Scenario ID**: `fraud_declines_call_center_cascade_v1` (`EWME-PREREG-20261023-LADDER`)  
**Epistemic Label**: `CONTROLLED_SYNTHETIC_KNOWN_MECHANISMS_NOT_REAL_WORLD_EFFICACY`  
**Methodology**: [Cost-Benefit Protocol for World Modeling Adoption](../methodology/cost_benefit_protocol.md)  

---

## 1. Scenario Description

An enterprise payment platform adjusts its fraud-blocking score threshold $\tau \in \{0.40, 0.55, 0.70\}$ across a 4-layer operational cascade:
1. **Fraud Detection Layer**: Lowering $\tau$ to `0.40` blocks more fraudulent transactions but increases false-positive legitimate transaction declines.
2. **Customer Friction & Call-Center Arrival Layer**: Legitimate customers whose transactions are declined call support with a lagged arrival distribution, creating an endogenous call-volume surge.
3. **Call-Center Queueing & SLA Layer**: When call arrivals exceed agent service capacity (`M/M/c` queue saturation), hold times spike and SLA violation rates breach the `5.0%` contractual ceiling (`18.12%` 10-seed mean SLA violation; `15.63%` in conservative single-calibration baseline).
4. **Enterprise Net Margin & Churn Layer**: SLA penalties, agent overtime, and false-decline customer churn overwhelm the direct fraud-loss savings of $\text{do}(\tau = 0.40)$.

---

## 2. Cost Estimate: EWM Approach (`Rung 5 Full EWM`)

### 2.1 Modeling Effort

| Activity | Hours | Hourly Rate ($/hr) | Cost ($) | Scope & Deliverable |
| :--- | :---: | :---: | :---: | :--- |
| **Scenario Design & Causal DAG** | `40` | `$100` | `$4,000` | 4-layer causal graph, intervention set $\mathcal{A}$, and enterprise utility specification |
| **Schema Definition** | `16` | `$100` | `$1,600` | Typed `WorldState`, `BitemporalLedger`, and `ContextSnapshot` ($K_t$) |
| **Constraint Specification** | `24` | `$100` | `$2,400` | Cohort conservation laws ($\le 10^{-9}$) and SLA capacity constraints |
| **Dynamics Calibration** | `32` | `$100` | `$3,200` | Structural queue/churn equations + projected GBT residual dynamics |
| **Testing & Conformance** | `24` | `$100` | `$2,400` | Leakage certificates, ablation suite, and property/metamorphic tests |
| **Subtotal: Modeling Effort** | **`136` hours** | **`$100/hr`** | **`$13,600`** | ~3.4 person-weeks (senior data scientist) |

### 2.2 Data Preparation

| Activity | Hours | Hourly Rate ($/hr) | Cost ($) | Scope & Deliverable |
| :--- | :---: | :---: | :---: | :--- |
| **Historical Fraud & Decline Extraction** | `16` | `$100` | `$1,600` | Point-in-time transaction and decline log extraction |
| **Call-Center Queue Telemetry Processing** | `8` | `$100` | `$800` | Agent staffing, arrival lag, and abandonment series |
| **Bitemporal Pipeline Setup** | `16` | `$100` | `$1,600` | `BitemporalLedger.as_of(t)` ingestion and zero-leakage verification |
| **Subtotal: Data Preparation** | **`40` hours** | **`$100/hr`** | **`$4,000`** | 1.0 person-week |

### 2.3 Compute Resources

- **Development & CI Executions**: `50` runs $\times$ `2` min (`4.82s` full reproduction harness; `~97s` extended 10-seed evaluation) $\times$ `$0.05`/CPU-hour $\approx$ `$0.08`
- **Production Counterfactual Rollouts**: `1,000` what-if scenarios/year $\times$ `2` min $\times$ `$0.05`/CPU-hour $\approx$ `$1.67`/year
- **Subtotal: Compute Resources**: **`~$2/year`** (deterministic CPU core; no mandatory GPU or LLM token cost)

### 2.4 Maintenance

- **Quarterly Drift Recalibration & Audit**: `8` hours/quarter $\times$ `4` quarters/year = **`32` hours/year (`$3,200/year`)**

**Total First-Year Cost**: `$13,600` (modeling) + `$4,000` (data) + `$2` (compute) + `$3,200` (maintenance) = **`$20,802`**

---

## 3. Cost Estimates: Comparison Baselines

### 3.1 Excel / Spreadsheet Baseline

- **Modeling Effort**: Spreadsheet design (`24` hours, `$2,400`) + formula validation (`8` hours, `$800`) = `32` hours (`$3,200`)
- **Maintenance**: Manual formula updates (`16` hours/year, `$1,600/year`)
- **Total First-Year Cost**: `$3,200` + `$1,600` = **`$4,800`**

### 3.2 Rules-Only Heuristic (`Rung 1` / `arch_rules_only`)

- **Modeling Effort**: Threshold rule script (`16` hours, `$1,600`) + validation (`8` hours, `$800`) = `24` hours (`$2,400`)
- **Maintenance**: Ad-hoc threshold adjustments (`16` hours/year, `$1,600/year`)
- **Total First-Year Cost**: `$2,400` + `$1,600` = **`$4,000`**

### 3.3 ML-Only Observational Predictor (`Rung 3` GBT with Lag Features)

- **Modeling Effort**: Feature engineering (`32` hours, `$3,200`) + GBT training/tuning (`24` hours, `$2,400`) + pipeline tests (`16` hours, `$1,600`) = `72` hours (`$7,200`)
- **Data Preparation**: `32` hours (`$3,200`)
- **Maintenance**: Retraining under drift (`32` hours/year, `$3,200/year`)
- **Total First-Year Cost**: `$7,200` + `$3,200` + `$3,200` = **`$13,600`**

---

## 4. Benefit Quantification

### 4.1 Per-Episode Decision Improvement (under $P_{\text{model}}$)

- **EWM (`Rung 5 Full EWM`) Decision Regret**: `$0.00` (`0.00%` SLA violation; selects optimal threshold $\text{do}(\tau = 0.55)$)
- **Baseline Decision Regret (`Rung 1`–`Rung 3` / Myopic Policy)**:
  - **Conservative Baseline Benchmark**: `$113,864.26` per episode (`15.63%` SLA violation) $\Rightarrow$ **`$113,864` regret reduction per episode**
  - **Canonical 10-Seed Comparative Ladder (`comparative_evaluation_ladder.json`)**: `$124,283.57` per episode (`18.12%` SLA violation) $\Rightarrow$ **`$124,283.57` regret reduction per episode**
  - **Flagship Point-in-Time Scenario (`fraud_cascade_flagship.json`)**: `$124,774.05` per episode $\Rightarrow$ **`$124,774.05` regret reduction per episode**

### 4.2 Annualized Scenario Application

Assuming **1 major fraud-policy what-if recalibration per quarter (`4/year`)**:
- **Conservative Annual Decision Improvement**: `4` $\times$ `$113,864` = **`$455,456/year`**
- **Canonical 10-Seed Annual Decision Improvement**: `4` $\times$ `$124,283.57` = **`$497,134.28/year`**

*(Note: All benefit figures are evaluated under the synthetic simulation distribution $P_{\text{model}}$; realized production savings depend on real-world mechanism fidelity.)*

---

## 5. ROI Analysis

### 5.1 Primary ROI Summary (`EWM` vs. Baselines)

| Model / Baseline | First-Year TCO | Annual Decision Benefit ($P_{\text{model}}$) | Net First-Year Value | ROI | Payback Period |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EWM (`Rung 5`, Conservative `$113.9K`/ep)** | **`$20,802`** | **`$455,456`** | **`+$434,654`** | **`2,089%`** | **`0.05` years (`18` days)** |
| **EWM (`Rung 5`, Canonical 10-Seed `$124.3K`/ep)** | **`$20,802`** | **`$497,134`** | **`+$476,332`** | **`2,290%`** | **`0.04` years (`15` days)** |
| **Excel / Spreadsheet Baseline** | `$4,800` | `$0` (cannot simulate queue cascade) | `-$4,800` | **`0%`** (`-100%` net) | Never (no regret reduction) |
| **Rules-Only Heuristic (`Rung 1`)** | `$4,000` | `$0` (`$124,283.57` regret/episode) | `-$4,000` | **`0%`** (`-100%` net) | Never |
| **ML-Only Lag Predictor (`Rung 3` GBT)** | `$13,600` | `$0` (`91.49%` lower 1-step MAE, but `$0` regret reduction) | `-$13,600` | **`0%`** (`-100%` net) | Never |

**Explicit Headline Metrics**:
- **Total First-Year Cost**: `$20,802` (`176` initial engineering hours + `32` maintenance hours + `$2` compute)
- **Annual Benefit**: `$455,456/year` (conservative) to `$497,134/year` (10-seed canonical mean) under $P_{\text{model}}$
- **ROI:** `2,089%` (conservative baseline) / `2,290%` (10-seed canonical mean)
- **Payback Period**: `0.05` years (`18` days / `0.6` months conservative; `15` days canonical)
- **Break-Even Threshold**: EWM breaks even in Year 1 whenever total annual regret reduction exceeds **`$20,802/year`** (`$5,200.50` per quarterly policy decision, or just **`4.57%`** of the simulated per-episode cascade loss).

---

## 6. Sensitivity Analysis

### 6.1 Single-Parameter Perturbations ($\pm 20\%$ Around Conservative Base Case)

| Assumption | Base Case | Low (`-20%`) | High (`+20%`) | Impact on ROI (Range) |
| :--- | :---: | :---: | :---: | :---: |
| **Annual Decision Improvement ($)** | `$455,456` | `$364,365` | `$546,547` | **`1,652% – 2,527%`** |
| **Initial Modeling Effort (Hours / $)** | `136h` (`$13,600`) | `109h` (`$10,880`) | `163h` (`$16,320`) | **`1,836% – 2,419%`** |
| **Data Preparation Effort (Hours / $)** | `40h` (`$4,000`) | `32h` (`$3,200`) | `48h` (`$4,800`) | **`2,008% – 2,177%`** |
| **Annual Maintenance (Hours/Year)** | `32h/y` (`$3,200`) | `26h/y` (`$2,560`) | `38h/y` (`$3,840`) | **`2,024% – 2,159%`** |
| **Loaded Hourly Rate ($/Hour)** | `$100/hr` | `$80/hr` (`-20%`) | `$120/hr` (`+20%`) | **`1,725% – 2,637%`** |

### 6.2 Multi-Parameter Stress Scenarios

| Scenario | Hourly Rate | Total Hours (Init + Maint) | Decision Frequency | Annual Benefit ($P_{\text{model}}$) | First-Year TCO | Net ROI (%) | Payback |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Optimistic (`-20%` cost, `+20%` benefit)** | `$80/hr` | `166h` | `4/year` | `$546,547` | `$13,282` | **`4,015%`** | `9` days |
| **Base Conservative** | `$100/hr` | `208h` | `4/year` | `$455,456` | `$20,802` | **`2,089%`** | `18` days |
| **Base Canonical (10-Seed Mean)** | `$100/hr` | `208h` | `4/year` | `$497,134` | `$20,802` | **`2,290%`** | `15` days |
| **Adverse (`+20%` hours, `-20%` benefit)** | `$100/hr` | `250h` | `4/year` | `$364,365` | `$24,962` | **`1,360%`** | `25` days |
| **Novice Team (`2.5x` hours, `$150/hr`, `1/yr` decision)** | `$150/hr` | `520h` | `1/year` | `$113,864` | `$78,002` | **`46%`** | `8.2` months |

**Conclusion**: Across all single-factor $\pm 20\%$ perturbations, first-year ROI remains above **`1,650%`**, and even under the extreme novice-team stress test (`2.5x` hours at `$150/hr` with only a single policy decision per year), EWM remains net-positive (`+46%` ROI, `8.2`-month payback).

---

## 7. Limitations and Caveats

1. **Synthetic Scenario Caveat (`CONTROLLED_SYNTHETIC_KNOWN_MECHANISMS_NOT_REAL_WORLD_EFFICACY`)**:
   - The `$113,864.26` / `$124,283.57` per-episode regret reduction is measured inside a controlled synthetic benchmark with known ground-truth equations, **not** from a live production deployment.
2. **$P_{\text{model}}$ vs. $P_{\text{real}}$ Structural Gap**:
   - All benefits are conditional on the declared causal graph capturing the true operational bottleneck. As shown in **Cautionary Case A (`obj_narrow_fraud_loss_minimization`)**, if an organization deploys `Rung 5` with a misspecified single-silo objective, policy regret remains `$123,689.15` (`$124,283.57` 10-seed mean), yielding **0% ROI** despite paying the full `$20,802` EWM TCO.
3. **Engineering Experience Assumptions**:
   - The `136`-hour modeling estimate assumes a senior data scientist or simulation engineer fluent in Python and typed state-space modeling; teams building their first structural simulation should budget `2x–3x` initial hours (`270–400` hours).
4. **Organizational Alignment & Change Management**:
   - Realizing the counterfactual benefit requires joint governance between Fraud Risk and Customer Support operations so that threshold changes recommended by the world model are actually enacted.
5. **Amortization Across Subsequent Scenarios**:
   - Approximately `60%` of the initial investment (`BitemporalLedger` pipeline, `WorldState` schemas, and CI verification harness) is reusable across adjacent operational cascades.

---

## 8. Decision Recommendation

- **Adopt EWM (`Rung 5`) when**:
  - Cross-silo operational cascades (fraud $\to$ support, inventory $\to$ fulfillment, pricing $\to$ churn) create non-linear capacity breaches where expected annual regret exceeds **`$50,000/year`** (yielding payback `< 5` months).
  - Bitemporal historical telemetry is available to calibrate transition equations and verify zero lookahead leakage.
- **Retain Excel / Rules / Standard ML (`Rung 1`–`Rung 3`) when**:
  - Annual decision stakes are `< $10,000/year`, dynamics are single-layer and stationary (`Negative Result N4`), or the task is purely observational 1-step forecasting without policy interventions (`Negative Result N1`).
