# Pilot Partner Agreement, IRB & Data Governance Template

**Version**: 1.0.0  
**Date**: 2026-10-30  
**Scope**: Standard Memorandum of Understanding (MoU) and IRB/Governance Checklist for `ewm-engine` 12-week shadow pilots  

---

## 1. Partner Memorandum of Understanding (MoU) Summary

- **Parties**: Enterprise World Model Engine Maintainer (`Vinicius Caridá`) and Pilot Partner Organization (`[Partner Legal Name]`, Vertical: `[Finance / Healthcare / Logistics]`).
- **Pilot Mode**: Non-invasive, read-only bitemporal replay and shadow advisory evaluation (`12` weeks).
- **License & IP**: Core `ewm-engine` remains Apache-2.0 licensed. Partner retains 100% ownership of proprietary telemetry and business rules; anonymized aggregate benchmark metrics (`MAE`, `Decision Regret`, `Constraint Preservation`, `TARP Error`, and engineering TCO hours) may be published in the repository's `Claim-to-Evidence Registry` upon joint sign-off.

---

## 2. IRB & Data Governance Checklist (Mandatory Prior to Phase 1)

- [ ] **C-01 (De-Identification)**: All customer, patient, or vendor identifiers removed or salted-SHA-256 pseudonymized prior to `BitemporalLedger` ingestion.
- [ ] **C-02 (HIPAA / GDPR / PCI Compliance)**: For Healthcare (`Vertical B`), HIPAA Safe Harbor 18-identifier stripping verified; for Payment Risk (`Vertical A`), PAN/PII fields excluded from state schemas.
- [ ] **C-03 (IRB Determination)**: Institutional Review Board (IRB) exemption or Quality Improvement (QI) non-human-subjects determination logged (`Reference ID: [ ]`).
- [ ] **C-04 (Read-Only Airgap)**: Connectors configured with read-only scopes (`OAuth2` / API token read-only); zero production write-back endpoints enabled.
- [ ] **C-05 (Pre-Registered Evaluation Protocol)**: Candidate action set $\mathcal{A}$, utility function $U(S_{1:H})$, and baseline rungs (`Rung 1`–`Rung 5`) locked prior to holdout evaluation.

---

## 3. Vertical-Specific Scope Schedules

### Schedule A — Financial Operations & Payment Risk Pilot
- **Source Systems**: Payment gateway authorization logs + CRM/contact-center queue telemetry (`Salesforce Service Cloud` / `ServiceNow`).
- **Conservation Invariant**: $\text{Total Submitted} = \text{Approved} + \text{Declined}_{\text{TrueFraud}} + \text{Declined}_{\text{FalsePositive}}$ ($\le 10^{-9}$).

### Schedule B — Healthcare Capacity & Patient Flow Pilot
- **Source Systems**: ADT bed-management event stream (`HL7`/`FHIR` export or tabular CSV/Parquet).
- **Conservation Invariant**: $\sum_{\text{unit}} \text{OccupiedBeds}_t + \text{AvailableBeds}_t = \text{LicensedCapacity}_t$ ($\le 10^{-12}$).

### Schedule C — Multi-Echelon Logistics & Supply Chain Pilot
- **Source Systems**: ERP inventory & purchase-order ledger (`SAP S/4HANA OData` / `MM-IM`).
- **Conservation Invariant**: $I_{t+1}^{(k)} = I_t^{(k)} + \text{ShipmentsIn}_t^{(k)} - \text{ShipmentsOut}_t^{(k)} - \text{Spoilage}_t^{(k)}$ ($\le 10^{-12}$).
