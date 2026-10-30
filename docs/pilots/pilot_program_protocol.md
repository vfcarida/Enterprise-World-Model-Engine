# Enterprise World Model Engine — Pilot Program Protocol

**Version**: 1.0.0  
**Date**: 2026-10-30  
**Author**: Vinicius Caridá  
**Prerequisites**: [Cost-Benefit Protocol](../methodology/cost_benefit_protocol.md), [Fraud Cascade TCO Analysis](../results/fraud_cascade_tco.md)  

---

## 1. Objective and Scope

This protocol governs external **12-week shadow-mode and advisory pilots** of `ewm-engine` across three target socio-technical verticals where multi-layer cascades and hard conservation/capacity constraints dominate single-step prediction accuracy:

1. **Partner Vertical A — Financial Operations & Payment Risk**: Fraud threshold $\to$ false-positive declines $\to$ support queue saturation & customer churn cascade.
2. **Partner Vertical B — Healthcare Inpatient & ED Capacity Flow**: Triage admission policy $\to$ ICU/ward bed occupancy $\to$ boarding delay & elective surgery cancellation cascade.
3. **Partner Vertical C — Multi-Echelon Logistics & Cold-Chain Fulfillment**: Reorder buffer policy $\to$ warehouse dock congestion $\to$ bullwhip amplification & spoilage cascade.

---

## 2. Target Partner Profiles & Recruitment Criteria

| Vertical | Target Partner Profile | Minimum Historical Data Requirement | Primary Counterfactual Intervention ($A_t$) | Target Annual Stake ($\Delta \text{Regret}$) |
| :--- | :--- | :--- | :--- | :---: |
| **A. Payment Risk & Support** | Fintech / issuer processing $\ge 5\text{M}$ tx/month with in-house support queues | 6 months of timestamped auth, decline, and call-center queue logs | Fraud rule threshold $\text{do}(\tau)$ & agent staffing allocation | $\ge \$100\text{K}/\text{year}$ |
| **B. Healthcare Operations** | Regional hospital network ($\ge 300$ beds) with EHR event logs | 12 months of de-identified ADT (Admission-Discharge-Transfer) events | Step-down transfer timing & elective block scheduling | $\ge \$150\text{K}/\text{year}$ |
| **C. Supply Chain & Logistics** | Multi-warehouse distributor ($\ge 3$ echelons, $\ge 1,000$ SKUs) | 12 months of ERP order, shipment, and inventory ledger snapshots | Safety-stock reorder points & cross-dock expediting | $\ge \$100\text{K}/\text{year}$ |

---

## 3. 12-Week Phased Execution Protocol

| Phase | Weeks | Activities | Deliverable & Gate |
| :--- | :---: | :--- | :--- |
| **Phase 0: Governance, IRB & TCO Baseline** | Weeks 1–2 | Execute Partner MoU & Data Governance Checklist; establish baseline TCO using [`cost_benefit_protocol.md`](../methodology/cost_benefit_protocol.md) | Signed MoU, De-Identification Certificate, Baseline Regret Estimate |
| **Phase 1: Connector Ingestion & Bitemporal Alignment** | Weeks 3–5 | Ingest historical snapshots via `ewm_engine.connectors` (`SAP`, `Salesforce`, `ServiceNow`, or Parquet); verify `BitemporalLedger.as_of(t)` zero-leakage certificate | `LeakageCertificate(passed=True)` on partner holdout split |
| **Phase 2: 5-Rung Comparative Ladder Backtest** | Weeks 6–8 | Evaluate Rung 1 (`AR(1)`), Rung 2 (`Tabular GBT`), Rung 3 (`Lag GBT`), Rung 4 (`Predict-Only Sim`), and Rung 5 (`Full EWM`) on historical regime shifts | Pre-registered comparative evaluation report (`ComparativeEvaluationReport`) |
| **Phase 3: Shadow-Mode Advisory Rollout** | Weeks 9–11 | Run Rung 5 in read-only shadow mode alongside production decision-makers; log `"DO_NOT_USE_ME_HERE"` abstentions and TARP coverage | Shadow-mode calibration & regret scorecard |
| **Phase 4: Post-Pilot TCO & Go/No-Go Review** | Week 12 | Populate [`tco_template.csv`](../methodology/tco_template.csv) with actual partner engineering hours and measured counterfactual regret reduction | Joint Case Study & `EXTERNALLY_REPRODUCED` Evidence Upgrade |

---

## 4. IRB, Privacy, and Operational Safeguards

1. **Strict Read-Only Shadow Execution**: During the 12-week pilot, `ewm-engine` operates exclusively in offline bitemporal replay and read-only shadow advisory mode. No automated write-back to production ERP, CRM, or clinical systems is permitted.
2. **PII / PHI De-Identification**: All entity identifiers must be cryptographically salted and hashed prior to ingestion into `WorldState`. Healthcare pilots require HIPAA Safe Harbor de-identification and institutional IRB exemption certification (see [`partner_agreements/partner_mou_template.md`](partner_agreements/partner_mou_template.md)).
3. **Mandatory Negative-Result Reporting**: If Rung 1–3 baselines match or outperform Rung 5 on the partner's decision objective (as in Negative Results `N1` and `N4`), the pilot report explicitly recommends retaining the simpler baseline.
