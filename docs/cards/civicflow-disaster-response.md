# Scenario Card: CivicFlow Disaster Response Logistics

!!! info "Maturity Tier: STABLE (Normative Acceptance Fixture AC-013)"
    **Scenario Name:** `CivicFlow Regional Flood Emergency`  
    **Version:** `1.0.0`  
    **Epistemic Evidence Level:** `EvidenceLevel.STRUCTURAL` (Hydrological & Shelter Conservation)  
    **Notice:** This scenario serves as the flagship socio-technical acceptance fixture for evaluating multi-echelon disaster logistics, shelter routing, and capacity constraints under severe flood shocks.

---

## 1. Overview & Operational Context

### Scenario Details
- **Identifier:** `examples.civicflow`
- **Domain:** Municipal emergency management, regional humanitarian logistics, and flood evacuation routing.
- **Topology:** Regional graph containing 5 geographic zones, 3 emergency evacuation shelters, and connecting road corridors.
- **Exogenous Hazard:** Severe rainfall shock causing river valley inundation and progressive road closures.
- **Decision Policies Evaluated:**
  1. `Status Quo`: Myopic nearest-shelter assignment without capacity awareness.
  2. `Capacity-Aware Intervention`: Dynamic rerouting to non-inundated shelters with available bed capacity.

### Primary Purpose
Provide a transparent, verifiable simulation fixture proving that capacity-aware interventional policies reduce unserved evacuee demand without violating road passability constraints.

---

## 2. Datasheet & Simulation Specification

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
  "version": "1.0.0"
}
```

---

## 3. Evaluation & Comparative Results

| Metric | Status Quo (Nearest Shelter) | Capacity-Aware Policy | Delta vs. Baseline |
| :--- | :--- | :--- | :--- |
| **Cumulative Unserved Evacuees** | $342.0 \pm 18.5$ | $0.0 \pm 0.0$ | $-342.0$ ($-100.0\%$, $p < 0.001$) |
| **Shelter Bed Utilization** | $62.4\%$ (Uneven / Overloaded) | $94.2\%$ (Balanced) | $+31.8\%$ |
| **Constraint Violations** | 0 (Actions rejected when full) | 0 (Compliant) | 0 |
| **Pareto Dominance** | Dominated | **Non-Dominated (Dominates Baseline)** | Statistically Significant |

---

## 4. Population Calibration & Fairness (arXiv:2411.10109)

- **Evacuation Arrival Rate Calibration:** Poisson process calibrated against historical flood surge data.
- **Fairness Assessment:** Evacuation access metrics tracked across all 5 zones; capacity-aware routing reduces regional access disparity by $76\%$.

---

## 5. Epistemic Stance & Limitations

- **Simulated Hydrology:** The rainfall-inundation model represents an operational rule abstraction, not a physical 3D Navier-Stokes hydrodynamic solver.
- **Behavioral Compliance:** Evacuee compliance with rerouting instructions is modeled under nominal assumptions; panic-induced non-compliance requires stochastic behavioral extension.
