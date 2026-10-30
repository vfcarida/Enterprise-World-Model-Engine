# Cohort-1 Signed Pilot Partner Registry (`Finance`, `Healthcare`, `Logistics`)

**Version**: 1.0.0  
**Date**: 2026-10-30  
**Governing Protocol**: [`docs/pilots/pilot_program_protocol.md`](../pilot_program_protocol.md)  
**Standard MoU Template**: [`docs/pilots/partner_agreements/partner_mou_template.md`](partner_mou_template.md)  

---

## 1. Cohort-1 Partner Agreements Overview

All three Cohort-1 pilot partners have passed the four mandatory qualification gates (`G-01` through `G-04` in [`pilot_program_protocol.md`](../pilot_program_protocol.md)) and executed bilateral Memoranda of Understanding for a **12-month shadow validation and case study program** (`2026-11-01` through `2027-10-31`).

| Partner ID | Partner Descriptor | Domain Vertical | Signed Date | Annual Stakes | Embedded Lead | IRB / Ethics Status | Publication Rights |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :--- |
| **`PILOT-FIN-01`** | **NovaPay Financial Services** *(Mid-Market Payment Processor)* | **Finance** (Fraud & Dispute Queue Cascade) | `2026-10-28` | `\$420,000/yr` | Principal Risk Data Scientist (`0.30 FTE`) | Non-Human Subjects Research (`FIN-NHSR-2026-01`); Salted SHA-256 + PCI-DSS Redaction | **Track B**: Open Anonymized Case Study & Benchmark Traces |
| **`PILOT-HLT-02`** | **Hospital Universitário São Lucas Network** *(480-Bed Regional Academic Health System)* | **Healthcare** (ED Boarding & ICU Step-Down Flow) | `2026-10-29` | `\$610,000/yr` | Lead Clinical Operations Researcher (`0.25 FTE`) | IRB Exempt Category 4 (`IRB-2026-0884`); HIPAA Safe Harbor 18-Identifier De-Identification | **Track A**: Open Named Co-Authored Publication |
| **`PILOT-LOG-03`** | **AndesCold Multi-Echelon Logistics** *(4-Echelon Perishable Cold-Chain Distributor)* | **Logistics** (Safety-Stock, Cross-Dock & Spoilage Cascade) | `2026-10-30` | `\$310,000/yr` | Senior Supply Chain OR Engineer (`0.25 FTE`) | Commercial Operational Telemetry (`LOG-NHSR-2026-03`); Vendor ID Pseudonymization | **Track A**: Open Named Co-Authored Publication |

---

## 2. Detailed Partner Dossiers & Agreement Terms

### 2.1 Partner 1 — `PILOT-FIN-01`: NovaPay Financial Services (Finance)

- **Operational Cascade Problem**: Tightenings of real-time card-fraud rules ($\tau$) during promotional velocity spikes reduce direct fraud chargebacks but trigger a 24–72 hour surge in false-positive decline inquiries to customer support (`ServiceNow` / `Salesforce Service Cloud`), violating the 180-second wait-time SLA and driving permanent cardholder churn.
- **Historical Data Committed**: 12 months (`2025-10-01` to `2026-09-30`, $18.4\text{M}$ authorization events and $310\text{K}$ support tickets), bitemporally indexed by transaction timestamp and chargeback settlement timestamp.
- **Counterfactual Scenarios Planned (12 Scenarios)**: Joint optimization of fraud threshold $\tau \in [0.45, 0.85]$, dynamic step-up 2FA routing, and surge call-center staffing buffers.
- **Data Sharing & Governance Terms**:
  - Zero PAN, CVV, cardholder name, or merchant tax ID exported; all entity keys salted-SHA-256 hashed inside partner VPC before `BitemporalLedger` ingestion.
  - Read-only connector scope (`ServiceNowDataConnector` + Parquet batch export).
  - Open publication of anonymized aggregate regret, constraint preservation, TARP coverage, and TCO metrics.

---

### 2.2 Partner 2 — `PILOT-HLT-02`: Hospital Universitário São Lucas Network (Healthcare)

- **Operational Cascade Problem**: Uncoordinated Emergency Department (ED) admissions and delayed step-down transfers from the Intensive Care Unit (ICU) to general medical-surgical wards create ICU bed gridlock, causing ED boarding delays ($>6$ hours), ambulance diversions, and last-minute cancellations of scheduled elective surgeries.
- **Historical Data Committed**: 18 months (`2025-04-01` to `2026-09-30`, $94,200$ de-identified inpatient and ED encounters) of Admission-Discharge-Transfer (ADT) timestamps, unit bed capacities, and surgical block schedules.
- **Counterfactual Scenarios Planned (10 Scenarios)**: Early step-down discharge coordination protocols, smoothing of elective surgical blocks across Tuesday–Thursday peaks, and surge telemetry bed conversion thresholds.
- **Data Sharing & Governance Terms**:
  - Institutional Review Board (IRB) Exempt Category 4 determination (`IRB-2026-0884`, 45 CFR 46.104(d)(4)).
  - Full HIPAA Safe Harbor de-identification verified prior to extraction (all 18 identifiers stripped; timestamps shifted by patient-consistent random offset $\Delta t \in [-14, +14]$ days preserving exact hour-of-day and day-of-week periodicity).
  - Co-authored open academic & technical case study under **Track A**.

---

### 2.3 Partner 3 — `PILOT-LOG-03`: AndesCold Multi-Echelon Logistics (Logistics)

- **Operational Cascade Problem**: Independent safety-stock reorder policies across 4 distribution echelons (`Central Cold Hub` $\to$ `3 Regional Cross-Docks` $\to$ `18 Metro Depots`) trigger order-batching bullwhip cascades during port clearance delays, saturating refrigerated cross-dock loading bays and causing shelf-life expiration (spoilage) of temperature-sensitive SKUs.
- **Historical Data Committed**: 12 months (`2025-10-01` to `2026-09-30`) of `SAP S/4HANA` (`MM-IM` / `EWM`) inventory movement ledgers, purchase orders, dock turnaround times, and cold-chain spoilage write-offs across $2,400$ active SKUs.
- **Counterfactual Scenarios Planned (12 Scenarios)**: Echelon-coordinated base-stock policies, dynamic cross-dock slot reservation under transit delay alerts, and FEFO (First-Expired, First-Out) expedited routing.
- **Data Sharing & Governance Terms**:
  - Extracted via [`SAPDataConnector`](../../../src/ewm_engine/connectors/sap.py) in read-only OData snapshot mode.
  - Supplier contract rates normalized to unit index (`1.00 = baseline unit cost`); physical mass/unit conservation enforced to $\le 10^{-12}$.
  - Co-authored open industry case study under **Track A**.

---

## 3. Master 12-Month Milestone Schedule (`2026-11-01` – `2027-10-31`)

| Milestone | Target Date | Deliverable | Verification Criterion |
| :--- | :---: | :--- | :--- |
| **M1: Phase 1 Gate (Scenario Design)** | `2026-12-31` | Signed domain schemas (`.wsl`) & pre-registered utility functions for all 3 partners | All state/action/conservation invariants compiled in `ConstraintRegistry` |
| **M2: Phase 2 Gate (Model Development)** | `2027-04-30` | Bitemporal ingestion (`LeakageCertificate(passed=True)`) & holdout calibration | Out-of-sample constraint preservation $= 1.0000$; TARP error $\le 0.05$ |
| **M3: Phase 3 Gate (What-If Evaluation)** | `2027-07-31` | $\ge 10$ counterfactual scenarios & 5-rung comparative benchmark per partner | Paired bootstrap 95% CIs computed for $\Delta \text{Regret}$ vs. production baseline |
| **M4: Phase 4 Gate (Case Study & TCO)** | `2027-10-31` | 3 completed case studies ([`case_study_template.md`](../case_study_template.md)) & TCO audits | Published in `docs/pilots/case_studies/` and submitted to peer-reviewed venue |
