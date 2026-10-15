# Calibration Report: calib_CalibratedWarehouse_1792073607

- **WorldSpec:** `CalibratedWarehouse`
- **Data Hash:** `a6df65c1355263de...`
- **Status:** `CERTIFIED`
- **Decision Usable:** `True`
- **Evidence Level:** `predictive` (Observational calibrated)

## 1. Coverage Certification (TARP & Posterior-SBC)

| Metric | Value | Threshold | Verdict |
| :--- | :--- | :--- | :--- |
| TARP Max Discrepancy | `0.2500` | `0.2800` | PASS |
| TARP KS Test p-value | `0.0765` | `0.0100` | PASS |
| SBC Rank Uniformity p | `0.0009` | `0.0100` | FAIL |
| Misspecification Score | `0.29 std` | `3.50 std` | PASS |

## 2. Posterior Distribution over Parameters

| Parameter | Posterior Mean | Std | Median | 90% Credible Interval |
| :--- | :--- | :--- | :--- | :--- |
| `initial_inventory` | 85.005 | 0.047 | 85.004 | [84.928, 85.087] |
| `depletion_rate` | 15.006 | 0.079 | 15.003 | [14.879, 15.141] |

## 3. Epistemic Integrity

> [!NOTE]
> **Calibrated $\neq$ Causally Identified:** A calibrated simulation accurately reproduces observed time series data under declared assumptions, but does not identify unconfounded counterfactual effects without exogenous variation or structural backdoor admissibility (Structural Causal Model).