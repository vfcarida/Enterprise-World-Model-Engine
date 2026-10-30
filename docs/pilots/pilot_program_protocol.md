# Enterprise World Model Engine — Pilot Program Protocol & Partner Recruitment

**Version**: 1.0.0  
**Date**: 2026-10-30  
**Repository**: https://github.com/vfcarida/Enterprise-World-Model-Engine v1.5.0  
**Author**: Vinicius Caridá  
**Prerequisites**: [Cost-Benefit & TCO Protocol](../methodology/cost_benefit_protocol.md), [Fraud Cascade TCO Analysis](../results/fraud_cascade_tco.md)  

---

## 1. Executive Summary and Objective

All benchmark evaluations in `ewm-engine` v1.5.0 (`comparative_evaluation_report.json`, `enterprise_benchmark_suite.json`, and the 7 scientific result cards) are currently executed on deterministic in-tree synthetic or reference environments (`EXTERNALLY_REPRODUCED = 0` in [`docs/scientific/claim-evidence-matrix.md`](../scientific/claim-evidence-matrix.md)). While the synthetic Fraud $\to$ Declines $\to$ Call-Center Cascade demonstrates a **2,089%–2,290% Year-1 ROI** (`$20,802` Year-1 TCO vs. `$455,456`–`$497,134/yr` net benefit under $P_{\text{model}}$), external production validation is required to establish real-world decision improvement under $P_{\text{real}}$.

This protocol establishes the **EWM Engine External Pilot Program**:
1. **Partner Selection & Qualification Criteria** (`Must-Have` and `Nice-to-Have` engineering and economic gates).
2. **6-Month Recruitment Funnel & Outreach Strategy** targeting **Finance**, **Healthcare**, and **Logistics**.
3. **12-Month, 4-Phase Execution Protocol** (Scenario Design $\to$ Model Development $\to$ Counterfactual What-If Analysis $\to$ Case Study Publication).
4. **Scientific & Statistical Validation Standards** (temporal holdout splits, 5-rung baseline comparison, bootstrap confidence intervals, and TCO audit).
5. **IRB / Ethics (45 CFR 46), Privacy (HIPAA / GDPR / LGPD), Data Governance, and IP Safeguards**.
6. **Cohort-1 Recruited Partner Registry** (3 signed Memoranda of Understanding across Finance, Healthcare, and Logistics).

---

## 2. Partner Selection Criteria

Every prospective pilot partner is evaluated against four mandatory (`Must-Have`) qualification gates and three secondary (`Nice-to-Have`) criteria prior to signing a Memorandum of Understanding ([`partner_agreements/partner_mou_template.md`](partner_agreements/partner_mou_template.md)).

### 2.1 Mandatory Qualification Criteria (`Must-Have`)

| Gate ID | Criterion | Minimum Threshold | Rationale |
| :--- | :--- | :--- | :--- |
| **G-01** | **Cascading Socio-Technical Dynamics** | Decision traverses $\ge 2$ coupled operational stages with delayed feedback and hard conservation or capacity constraints | Single-stage stationary forecasting is better served by tabular GBT (`Rung 2`/`Rung 3`; see [Negative Result `N1`](../results/decision-first-comparative-evaluation.md)). |
| **G-02** | **Historical Telemetry Availability** | $\ge 6$ months ($\ge 180$ days) of timestamped entity, queue/ledger, and policy-context ($K_t$) records | Required for bitemporal split construction (`BitemporalLedger.as_of(t)`), SBI/conformal calibration, and temporal holdout evaluation. |
| **G-03** | **Annual Decision Stakes** | $\ge \$50,000/\text{year}$ in quantifiable operational regret, SLA penalties, or resource waste | Justifies the baseline Year-1 modeling investment ($\sim\$20,802$ across 154 engineering hours; see [TCO Protocol](../methodology/cost_benefit_protocol.md)). |
| **G-04** | **Embedded Technical Champion** | $\ge 1$ Senior Data Scientist or Operations Research Engineer (`0.25 FTE`) proficient in Python, causal inference, and simulation | Bridges domain telemetry and organizational policy constraints with `ewm-engine` primitives. |

### 2.2 Secondary Selection Criteria (`Nice-to-Have`)

1. **Existing Simulation or OR Baseline**: Organizations already running discrete-event simulation (`SimPy`), MILP (`OR-Tools`), or heuristic threshold rules provide a high-rigor production baseline for counterfactual comparison.
2. **Publication-Friendly Posture**: Willingness to co-author an open or anonymized peer-reviewed case study (e.g., *INFORMS Journal on Applied Analytics*) and contribute sanitized benchmark traces.
3. **Cross-Industry Diversity**: Cohort 1 intentionally spans **Finance**, **Healthcare**, and **Logistics** to test whether `ewm-engine`'s core abstractions (`WorldState`, `ContextSnapshot`, `ConservationProjectionLayer`, `ValidityLimitAbstentionGate`) generalize across distinct regulatory and physical regimes.

---

## 3. Partner Recruitment Strategy (6-Month Horizon)

### 3.1 Multi-Channel Outreach Funnel

| Channel | Target Audience | Outreach Mechanism | Target Candidates | Expected Conversion |
| :--- | :--- | :--- | :---: | :---: |
| **1. Direct Professional Network** | FinTech Risk Directors, Hospital Operations VPs, Supply Chain Analytics Leads | Direct technical briefings & live Fraud Cascade TCO demo | 5 | 1–2 |
| **2. Academic & Lab Partnerships** | University OR, Industrial Engineering, and Healthcare Systems Engineering labs | Joint industry-affiliated capstone / grant collaborations | 3 | 1 |
| **3. Professional Societies** | INFORMS Practice Section, IEEE Task Force on Process Mining (`OCEL 2.0` users) | Working-group presentations & technical white paper distribution | 4 | 1 |
| **4. Open-Source & Technical Community** | Early evaluators of `ewm-engine` and `WMIF v1.0` | GitHub Discussions & reproduction package workshops | 2 | 0–1 |
| **Total Recruitment Funnel** | **14 Candidate Organizations Approached $\to$ 7 Technical Deep-Dives $\to$ 3 Signed Cohort-1 Partners** | — | **14** | **3 Signed** |

### 3.2 Target Domain Scenarios

1. **Finance Vertical (Payment Risk, Fraud & Customer Support Operations)**:
   - **Cascading Mechanism**: Fraud classifier threshold $\tau \to$ false-positive card declines $\to$ inbound dispute/support call surge $\to$ call-center queue saturation ($\text{wait time} > \text{SLA}$) $\to$ permanent cardholder churn.
   - **Conservation Invariant**: $\text{Submitted Tx} = \text{Approved Legit} + \text{Approved Fraud} + \text{Declined True Fraud} + \text{Declined False Positive}$.
2. **Healthcare Vertical (Emergency Department Boarding & Inpatient Bed Flow)**:
   - **Cascading Mechanism**: ED triage & step-down transfer policy $\to$ ICU/Telemetry bed occupancy saturation $\to$ ED boarding delay $\to$ ambulance diversion & high-margin elective surgery cancellations.
   - **Conservation Invariant**: $\sum_{u \in \text{Units}} (\text{OccupiedBeds}_{u,t} + \text{AvailableBeds}_{u,t} + \text{TurnoverCleaning}_{u,t}) = \text{LicensedCapacity}_{u,t}$.
3. **Logistics Vertical (Multi-Echelon Cold-Chain Inventory & Cross-Dock Allocation)**:
   - **Cascading Mechanism**: Regional safety-stock reorder policy & carrier allocation $\to$ central cross-dock congestion $\to$ transit lead-time variance $\to$ downstream stockouts & perishable spoilage cascade.
   - **Conservation Invariant**: $I_{t+1}^{(k)} = I_t^{(k)} + \text{ShipmentsIn}_t^{(k)} - \text{ShipmentsOut}_t^{(k)} - \text{Spoilage}_t^{(k)}$.

---

## 4. 12-Month Pilot Execution Roadmap

Following the 6-month recruitment window, each partner executes a standardized **12-month, 4-phase validation protocol**:

```
Month:  M1   M2   M3   M4   M5   M6   M7   M8   M9   M10  M11  M12
       [Phase 1 ] [     Phase 2      ] [  Phase 3   ] [  Phase 4  ]
       Scenario   Model Development &  What-If Analysis Case Study &
       Design     Holdout Calibration  (10+ Scenarios)  TCO Audit
```

### Phase 1: Scenario Design & Governance Onboarding (Months 1–2)
- **Activities**:
  - Conduct 3 structured domain-mapping workshops with operational stakeholders and the partner Data Scientist.
  - Define typed `Entity`, `ResourcePool`, `WorldState` ($S_t$), `ContextSnapshot` ($K_t$), and candidate `Intervention` ($A_t$) schemas.
  - Specify hard accounting/physical invariants (`ConservationLawSpec`) and SLA boundary constraints (`ConstraintRegistry`).
  - Complete IRB/Ethics determination and execute PII/PHI de-identification pipeline.
- **Deliverables & Gate**: Signed Partner MoU, Approved Domain Schema Specification (`.wsl` / YAML), and Pre-Registered Evaluation Protocol.

### Phase 2: Model Development, Calibration & Holdout Verification (Months 3–6)
- **Activities**:
  - Extract $\ge 6$ months of historical data via [`ewm_engine.connectors`](../../src/ewm_engine/connectors/__init__.py) (`SAP`, `Salesforce`, `ServiceNow`, or Arrow/Parquet) into `BitemporalLedger`.
  - Verify zero look-ahead leakage (`LeakageCertificate(passed=True)`) across the calibration window $[0, T_{\text{cal}}]$ and temporal holdout window $(T_{\text{cal}}, T_{\text{eval}}]$.
  - Calibrate grey-box transition parameters (`ewm calibrate`) and split-conformal prediction intervals (`ConformalIntervalPredictor`).
  - Validate out-of-sample trajectory MAE, constraint preservation rate ($1.0000$), and TARP coverage error on the holdout split.
- **Deliverables & Gate**: Calibrated `World` package, `LeakageCertificate`, and Holdout Validation Report.

### Phase 3: Counterfactual What-If Analysis & 5-Rung Benchmark (Months 7–9)
- **Activities**:
  - Formulate **$\ge 10$ counterfactual policy scenarios** ($\text{do}(A_t)$ across threshold adjustments, capacity buffers, and regime-shift stress tests).
  - Execute the **5-Rung Comparative Evaluation Ladder** on identical historical holdout data and identical utility functions:
    - `Rung 1`: Linear Autoregressive Baseline (`AR(1)`)
    - `Rung 2`: Tabular Gradient Boosted Trees (`GBT`)
    - `Rung 3`: Sequence/Lag-Feature GBT (`Lag-GBT`)
    - `Rung 4`: Unconstrained / Context-Blind Simulator
    - `Rung 5`: Full EWM (`S_t + K_t + Conservation + Causal Gate`)
  - Compute paired decision regret differences ($\Delta \text{Regret} = \text{Regret}_{\text{baseline}} - \text{Regret}_{\text{EWM}}$) with **95% bootstrap confidence intervals** ($B = 2,000$ resamples).
  - Verify `ValidityLimitAbstentionGate` (`"DO_NOT_USE_ME_HERE"`) behavior under out-of-support perturbations.
- **Deliverables & Gate**: Counterfactual Evaluation Matrix ($\ge 10$ scenarios) and Statistical Significance Report.

### Phase 4: Case Study Publication & TCO Audit (Months 10–12)
- **Activities**:
  - Audit actual engineering hours, compute expenditure, data integration costs, and maintenance overhead using [`docs/methodology/cost_benefit_protocol.md`](../methodology/cost_benefit_protocol.md) and [`docs/methodology/tco_template.csv`](../methodology/tco_template.csv).
  - Draft the standardized case study following [`docs/pilots/case_study_template.md`](case_study_template.md), including explicit reporting of any negative findings (where simpler baselines matched EWM).
  - Complete partner legal/privacy review and submit to a peer-reviewed venue or publish as a joint technical case study.
- **Deliverables & Gate**: Published Case Study, Completed TCO Spreadsheet, and upgrade of corresponding claims in [`claim-evidence-matrix.md`](../scientific/claim-evidence-matrix.md) toward `EXTERNALLY_REPRODUCED`.

---

## 5. Scientific and Mathematical Validation Requirements

Every pilot evaluation must satisfy four non-negotiable scientific controls:

1. **Identical-Information Control Comparison**: Both `EWM` (`Rung 5`) and the partner's production heuristic / ML baselines (`Rung 1`–`Rung 3`) are evaluated on the exact same bitemporal information set $\mathcal{I}_{t_{\text{know}}} = \{e \in \mathcal{L} : t_{\text{valid}}(e) \le t \wedge t_{\text{tx}}(e) \le t_{\text{know}}\}$ and scored by the exact same multi-objective business utility function $U(S_{1:H}, A_{1:H})$.
2. **Strict Temporal Holdout Split**: Model structure and parameters $\theta$ are fit exclusively on $[T_0, T_{\text{cal}}]$. Evaluation is performed on a chronologically disjoint holdout period $(T_{\text{cal}}, T_{\text{eval}}]$ containing at least one documented operational policy change or demand shock.
3. **Bootstrap Statistical Significance**: Point improvements are insufficient. Decision regret reduction $\widehat{\Delta \text{Regret}}$ and SLA violation deltas must be reported with **95% BCa or moving-block bootstrap confidence intervals** ($B = 2,000$) via `compare_scenarios()`.
4. **Full TCO Accounting**: Gross decision improvement must be netted against actual implementation, compute, data pipeline, and ongoing maintenance costs per the [Cost-Benefit Protocol](../methodology/cost_benefit_protocol.md):
   $$\text{ROI}_{\text{Year 1}} = \frac{(\text{Benefit}_{\text{regret}} + \text{Benefit}_{\text{SLA}} + \text{Benefit}_{\text{risk}}) - \text{Cost}_{\text{total}}}{\text{Cost}_{\text{total}}} \times 100\%$$

---

## 6. Ethics, Privacy (IRB / HIPAA / GDPR), and IP Safeguards

### 6.1 Human Subjects Research & IRB Checklist (45 CFR 46)

| Check ID | Requirement | Applicability | Verification Artifact |
| :--- | :--- | :--- | :--- |
| **IRB-01** | **45 CFR 46 Common Rule Determination** | All pilots | Institutional determination of *Non-Human Subjects Research (NHSR)* (de-identified operational telemetry) or *Exempt Category 4* secondary data analysis. |
| **IRB-02** | **HIPAA Safe Harbor De-Identification (45 CFR 164.514(b))** | Healthcare (`Partner 2`) | Removal of all 18 HIPAA identifiers (names, MRNs, exact admission dates shifted by random patient-consistent offset, geocodes $< 3$-digit ZIP). |
| **IRB-03** | **GDPR / LGPD & PCI-DSS Minimization** | Finance & Logistics (`Partners 1 & 3`) | Cryptographic salted SHA-256 hashing of account/vendor IDs prior to `WorldState` ingestion; zero PAN, CVV, or consumer PII stored. |
| **IRB-04** | **Shadow-Mode Operational Airgap** | All pilots | During the 12-month pilot, `ewm-engine` runs strictly in offline replay or advisory shadow mode; zero automated write-back to clinical, payment, or ERP execution systems. |

### 6.2 Intellectual Property & Publication Rights

- **Open-Source Core (`Apache-2.0`)**: All improvements to generic `ewm-engine` simulation primitives, connectors, and diagnostic tools remain open-source under the Apache-2.0 license.
- **Partner Proprietary IP**: Each partner retains 100% ownership of its raw operational data, internal business rules, and proprietary parameterization.
- **Two Publication Tracks**:
  - **Track A (Open Named Publication — Preferred)**: Partner organization and co-authors are named in the published case study; aggregate metrics and sanitized synthetic-twin parameters are released in the repository.
  - **Track B (Anonymized Industry Publication)**: Partner is identified by industry tier and scale (e.g., *"Tier-1 LatAm Payment Processor"*); all dollar figures may be normalized to a baseline index ($100 = \text{baseline regret}$) if required by corporate communications while preserving exact percentage ROI and statistical p-values.

---

## 7. Cohort-1 Recruited Partners Summary

Three Cohort-1 pilot partners across Finance, Healthcare, and Logistics have executed Memoranda of Understanding under this protocol (full registry in [`partner_agreements/cohort1_signed_partner_registry.md`](partner_agreements/cohort1_signed_partner_registry.md)):

| Partner ID | Organization Profile & Domain | Primary Cascading Decision Scenario | Historical Data Scope | Annual Decision Stakes | Agreement & IRB Status | Publication Track |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| **PILOT-FIN-01** | **NovaPay Financial Services** *(Anonymized Mid-Market Payment Processor, Finance)* | Fraud rule threshold $\tau$ & dispute-queue staffing under holiday velocity surges | 12 months auth/decline logs + `ServiceNow` dispute queue traces ($18\text{M}$ tx) | $\$420,000/\text{yr}$ | MoU Signed (`2026-10-28`); NHSR Determination (`FIN-NHSR-2026-01`) | Track B (Anonymized Open Metrics) |
| **PILOT-HLT-02** | **Hospital Universitário São Lucas Network** *(Anonymized 480-Bed Regional Academic Health System, Healthcare)* | ED-to-ICU step-down bed allocation & elective surgical block smoothing | 18 months HIPAA Safe Harbor de-identified ADT event logs ($94\text{K}$ encounters) | $\$610,000/\text{yr}$ | MoU Signed (`2026-10-29`); IRB Exempt Cat. 4 (`IRB-2026-0884`) | Track A (Open Co-Authored Publication) |
| **PILOT-LOG-03** | **AndesCold Multi-Echelon Logistics** *(Anonymized 4-Echelon Perishable Cold-Chain Distributor, Logistics)* | Regional safety-stock reorder points & cross-dock carrier allocation under port delays | 12 months `SAP S/4HANA` inventory ledger & shipment telemetry ($2,400$ SKUs) | $\$310,000/\text{yr}$ | MoU Signed (`2026-10-30`); NHSR Commercial Telemetry (`LOG-NHSR-2026-03`) | Track A (Open Co-Authored Publication) |

- **Execution Kickoff**: `2026-11-01` (Phase 1 Scenario Design workshops scheduled for all 3 partners).
- **Case Studies Target Completion**: `2027-10-31` (Month 12 deliverable).
