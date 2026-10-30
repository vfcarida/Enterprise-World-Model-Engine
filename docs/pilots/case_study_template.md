# Enterprise World Model Engine — Standardized Pilot Case Study Template

**Template Version**: 1.0.0  
**Protocol Reference**: [`docs/pilots/pilot_program_protocol.md`](pilot_program_protocol.md)  
**TCO Methodology Reference**: [`docs/methodology/cost_benefit_protocol.md`](../methodology/cost_benefit_protocol.md)  

---

## Metadata Header

- **Case Study ID**: `PILOT-[FIN|HLT|LOG]-XX`
- **Partner Organization**: `[Organization Name or Anonymized Industry Descriptor]`
- **Industry Vertical**: `[Finance / Healthcare / Logistics]`
- **Pilot Duration**: `[Start Date YYYY-MM-DD]` to `[End Date YYYY-MM-DD]` (`12 months`)
- **Publication Track**: `[Track A: Open Named | Track B: Anonymized Open Metrics]`
- **IRB / Governance Reference**: `[IRB Exemption ID / NHSR Reference]`
- **EWM Engine Version**: `v1.5.0`

---

## 1. Background

### 1.1 Organization and Operational Context
- **Organizational Scale**: `[Transaction volume / bed capacity / SKU & warehouse count]`
- **Operational Problem**: `[Describe the multi-stage cascading bottleneck where single-stage reactive heuristics or point forecasts fail]`
- **Baseline Decision Process**: `[Describe the existing heuristic, spreadsheet, or tabular ML rule used in production prior to the pilot]`

### 1.2 Decision Stakes and Business Objective
- **Annualized Financial Exposure**: `$\text{[X]}/year` (must exceed the `$\$50,000/\text{yr}` qualification threshold)
- **Primary SLA / Safety Constraint**: `[e.g., Call-center wait time <= 180s | ED boarding time <= 4h | Cold-chain spoilage <= 0.5%]`
- **Why Simpler Tools Were Insufficient**: `[Explain why static MILP or tabular GBT could not capture delayed feedback, conservation invariants, or runtime context shifts K_t]`

---

## 2. Scenario Specification (`S_t, K_t, A_t, \mathcal{T}`)

### 2.1 State Space ($S_t$) and Bitemporal Data Schema
- **Source Systems & Connectors**: `[e.g., SAPDataConnector, SalesforceDataConnector, ServiceNowDataConnector, or Parquet]`
- **Historical Calibration Window**: `[YYYY-MM-DD to YYYY-MM-DD]` (`N` records)
- **Temporal Holdout Window**: `[YYYY-MM-DD to YYYY-MM-DD]` (`M` records; `LeakageCertificate(passed=True)`)

| State Variable / Pool | Type | Physical / Accounting Meaning | Conservation / Boundary Constraint |
| :--- | :--- | :--- | :--- |
| `[var_1]` | `float64` | `[Description]` | `[e.g., Non-negativity >= 0]` |
| `[var_2]` | `float64` | `[Description]` | `[e.g., Exact conservation sum = Total]` |

### 2.2 Runtime Context ($K_t$), Actions ($A_t$), and Causal Graph ($\mathcal{G}$)
- **Runtime Context Snapshot ($K_t$)**: `[Policy parameters, staffing rosters, or tariff schedules tracked in ContextSnapshot]`
- **Candidate Interventions ($\text{do}(A_t)$)**: `[List the 10+ counterfactual policy interventions evaluated in Phase 3]`
- **Causal Evidence Tagging (`DependencyEvidenceNature`)**:
  - `STRUCTURAL_CONSERVATION`: `[Count]` edges
  - `EXPERIMENTAL_INTERVENTION`: `[Count]` edges
  - `OBSERVATIONAL_CONDITIONAL`: `[Count]` edges

---

## 3. Total Cost of Ownership (TCO) Analysis

*Computed using [`docs/methodology/cost_benefit_protocol.md`](../methodology/cost_benefit_protocol.md) and [`docs/methodology/tco_template.csv`](../methodology/tco_template.csv).*

### 3.1 Four-Component Cost Breakdown

| Cost Component | Engineering Hours | Fully-Loaded Rate (`$/hr`) | Direct Infrastructure / License (`$`) | Year-1 Cost (`$`) | Ongoing Annual Cost (`$/yr`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Development ($C_{\text{dev}}$)** | `[hrs]` | `$[rate]` | `$0` | `$[cost]` | `$0` |
| **2. Compute ($C_{\text{compute}}$)** | — | — | `$[cost]` | `$[cost]` | `$[cost]` |
| **3. Data Integration ($C_{\text{data}}$)** | `[hrs]` | `$[rate]` | `$[cost]` | `$[cost]` | `$[cost]` |
| **4. Maintenance ($C_{\text{maint}}$)** | `[hrs/yr]` | `$[rate]` | `$0` | `$[cost]` | `$[cost]` |
| **Total TCO** | **`[Total hrs]`** | — | **`$[Total infra]`** | **`$[Year-1 Total]`** | **`$[Ongoing Total]`** |

---

## 4. Empirical Results (Holdout & Counterfactual Evaluation)

### 4.1 Five-Rung Comparative Ladder on Temporal Holdout

| Rung | Method | Trajectory MAE | Constraint Preservation | Mean Decision Regret (`$`) | Regret Reduction vs. Baseline (`95% Bootstrap CI`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Rung 1** | Linear `AR(1)` Baseline | `[val]` | `[val]` | `$[val]` | — |
| **Rung 2** | Tabular `GBT` Predictor | `[val]` | `[val]` | `$[val]` | `$[delta] ([low, high])` |
| **Rung 3** | Sequence `Lag-GBT` | `[val]` | `[val]` | `$[val]` | `$[delta] ([low, high])` |
| **Rung 4** | Unconstrained Simulator | `[val]` | `[val]` | `$[val]` | `$[delta] ([low, high])` |
| **Rung 5** | **Full EWM Engine** | **`[val]`** | **`1.0000`** | **`$[val]`** | **`$[delta] ([low, high])`** |

### 4.2 Summary Financial ROI and Payback Metrics

- **Annualized Regret Reduction ($B_{\text{regret}}$)**: `$\text{[X]}/year`
- **Annualized SLA Penalty Avoidance ($B_{\text{SLA}}$)**: `$\text{[Y]}/year`
- **Annualized Risk Mitigation ($B_{\text{risk}}$)**: `$\text{[Z]}/year`
- **Total Gross Annual Benefit**: `$\text{[Total Benefit]}/year`
- **Year-1 Net Benefit**: `$\text{[Net Benefit]}`
- **Year-1 ROI (%)**: `[ROI]%`
- **Payback Period**: `[Days / Months]`
- **3-Year NPV ($r = 10\%$)**: `$\text{[NPV]}`

---

## 5. Limitations, Negative Findings, and Epistemic Boundaries

### 5.1 Model Discrepancy ($P_{\text{model}}$ vs. $P_{\text{real}}$)
- **Observed Holdout Discrepancy**: `[Report empirical residual between simulated trajectories and observed production outcomes during Phase 3 shadow mode]`
- **TARP Calibration Error**: `[Report TARP coverage error before and after split-conformal calibration]`

### 5.2 Negative Findings (When Simpler Baselines Sufficed)
- **Stationary Regime Performance**: `[Document any sub-scenarios or stationary periods where Rung 2/3 tabular baselines matched or outperformed Rung 5, consistent with Negative Result N1]`

### 5.3 Abstention Gate Triggers (`"DO_NOT_USE_ME_HERE"`)
- **Out-of-Support Perturbations**: `[Document how many times ValidityLimitAbstentionGate or EpistemicRefusalError fired during shadow evaluation and what unmodeled regime shift triggered it]`
- **Cross-Domain Generalizability**: `[Summarize which state/constraint patterns transfer to peer organizations and which require partner-specific recalibration]`
