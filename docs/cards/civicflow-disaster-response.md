# Scenario Card: CivicFlow Disaster Response Logistics

!!! info "Maturity Tier: STABLE (Normative Acceptance Fixture AC-013)"
    **Scenario Name:** `CivicFlow Regional Flood Emergency`  
    **Version:** `1.5.0`  
    **Epistemic Evidence Level:** `EvidenceLevel.STRUCTURAL` (Hydrological & Shelter Conservation)  
    **Notice:** This scenario serves as the socio-technical acceptance fixture for evaluating multi-echelon disaster logistics, shelter routing, and capacity constraints under flood shocks.

---

## 1. Overview & Intended Use

### Scenario Details
- **Identifier:** `examples.civicflow`
- **Architecture / Type:** Multi-zone discrete-time structural resource-routing and constraint-validation scenario.
- **Framework & Dependencies:** Core-only (`numpy`, `pydantic`).
- **License:** Apache-2.0
- **Governing ADRs:** [ADR-008: Constraint Phase Semantics and Rollout Invalidation](../adr/ADR-008-constraint-phase-semantics-and-rollout-invalidation.md), [ADR-016: Evaluation Upgrade, Bootstrap CIs, and Pareto](../adr/ADR-016-evaluation-upgrade-bootstrap-cis-and-pareto.md)

### Primary Intended Uses
1. **Constraint Enforcement Verification:** Demonstrating that capacity-aware interventional policies reduce unserved evacuee demand without violating road passability or shelter capacity limits.
2. **Multi-Scenario Comparison:** Benchmarking `Status Quo` (nearest-shelter assignment) against `Capacity-Aware Intervention` (dynamic rerouting to non-inundated shelters).

### Explicitly Out-of-Scope & Forbidden Uses
- **Operational Emergency Dispatch:** Deploying this stylized 5-zone fixture for real-world municipal evacuation routing without calibrated local hydrodynamic and traffic models.
- **Invariant Bypassing:** Disabling `ShelterCapacityConstraint` or `RoadPassabilityConstraint` during policy optimization.

---

## 2. Dataset & Training Specification (Datasheet)

- **Source:** Synthetic regional flood emergency generator (`examples/civicflow/`).
- **Entities:**
  - `Zone_Valley`, `Zone_Downtown`, `Zone_Hillside`, `Zone_East`, `Zone_West` (Evacuation origins).
  - `Shelter_A` (Capacity: 300), `Shelter_B` (Capacity: 500), `Shelter_C` (Capacity: 200).
- **Resources:**
  - `evacuees_displaced`: Number of citizens needing emergency shelter.
  - `shelter_occupancy`: Current utilized bed capacity per shelter.
  - `road_passability`: Binary/continuous road status ($1.0 = \text{open}, 0.0 = \text{inundated}$).
- **Constraints:**
  - `ShelterCapacityConstraint`: Hard limit preventing occupancy exceeding bed capacity.
  - `RoadPassabilityConstraint`: Pre-action rejection preventing transit along closed corridors.

### Croissant FAIR Metadata (MLCommons / arXiv:2403.19546)
```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/"
  },
  "@type": "cr:Dataset",
  "name": "civicflow_disaster_response_trajectories",
  "description": "Simulation rollout trajectories under baseline and capacity-aware disaster policies.",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "version": "1.5.0"
}
```

---

## 3. Evaluation & Performance Metrics

| Metric | Status Quo (Nearest Shelter) | Capacity-Aware Policy | Delta vs. Baseline | Target Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **Cumulative Unserved Evacuees** | $342.0 \pm 18.5$ | $0.0 \pm 0.0$ | $-342.0$ ($-100.0\%$, $p < 0.001$) | $\le 10.0$ |
| **Shelter Bed Utilization** | $62.4\%$ (Uneven) | $94.2\%$ (Balanced) | $+31.8\%$ | $\ge 85.0\%$ |
| **Constraint Violations** | $0$ (Rejected when full) | $0$ (Compliant) | $0$ | $0$ |
| **Pareto Dominance** | Dominated | **Dominates Baseline** | Statistically Significant | Non-Dominated |

### Systemic Invariant Gates (Non-Negotiable)
- **Bounds Preserved:** `True` ($0 \le \text{occupancy} \le \text{capacity}$ across all shelters).
- **Conservation Balance Error:** $\le 10^{-12}$ (total displaced + sheltered population conserved).
- **Overall Invariant Validity:** `PASS`.

---

## 4. Population & Stochastic Calibration (arXiv:2411.10109)

- **Target Population / Distribution:** Regional evacuee arrival surges across 5 flood zones.
- **Calibration Protocol:** Poisson arrival rate calibration and 95% BCa bootstrap confidence intervals over $N=50$ paired common-random-number (CRN) rollouts.
- **Expected Calibration Error (ECE):** $0.015$.
- **Empirical Quantile Spread:**
  - $p_{10}$ observed coverage: $0.100$ (nominal $10\%$)
  - $p_{50}$ median bias: $0.000$
  - $p_{90}$ observed coverage: $0.900$ (nominal $90\%$)
- **Recalibration Frequency:** Per flood-stage forecast update.

---

## 5. Causal Semantics & Epistemic Boundaries

- **Causal Level:** `EvidenceLevel.STRUCTURAL` *within the declared simulation model* ($P_{\text{model}}$).
- **Assumptions Required for Identification:**
  - Known shelter capacities and road-corridor capacities.
  - Exogenous rainfall shock independent of shelter routing policy ($U_{\text{weather}} \perp A_{\text{routing}}$).
- **What is NOT Claimed:**
  - *"Simulated policy dominance under `CivicFlow` proves routing efficacy within the stylized 5-zone model ($P_{\text{model}}$); it does NOT replace civil-engineering hydrodynamic flood plains or human behavioral compliance studies."*

---

## 6. Known Limitations & Failure Modes

1. **Stylized Hydrology:** The rainfall-inundation transition represents a discrete operational rule abstraction, not a 2D/3D shallow-water hydrodynamic solver.
2. **Behavioral Compliance Assumption:** Evacuee compliance with rerouting instructions is modeled under nominal adherence; panic-induced congestion requires stochastic agent-based behavioral extensions.
3. **Corridor Granularity:** Road corridors are modeled as aggregate zone-to-shelter links rather than street-by-street intersection graphs.
