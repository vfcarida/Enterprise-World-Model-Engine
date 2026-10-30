# When Does Explicitly Modeling Enterprise Dynamics Beat Predicting Outcomes? A Pre-Registered Comparative Evaluation

**Author**: Vinicius Caridá  
**Affiliation**: Enterprise World Model Engine Project  
**Version**: 1.5.0 Preprint Manuscript  
**Target Venues**: *INFORMS Journal on Computing* / *ACM/IEEE ICSE* / *NeurIPS Datasets & Benchmarks Track*  
**Pre-Registration ID**: `EWME-PREREG-20261023-LADDER`  
**Reproduction Manifest SHA-256**: `618142f9eab6b9e04bad9f3903313fd12525672f9844ea82f9d05d9bdcc160c6` (`CERTIFIED_RELEASE`)  

---

## Abstract

Supervised machine learning models excel at estimating observational associations $P(Y_{t+1} \mid X_t)$ in stationary regimes, yet operational decision-makers must choose interventions $\text{do}(A_t = a)$ in coupled socio-technical systems governed by conservation laws, queue capacities, and runtime configuration shifts. Rather than claiming that world models universally outperform simpler baselines, we ask a conditional scientific question: *Under which structural and economic conditions does explicitly modeling system dynamics, constraints, and runtime context ($K_t$) reduce decision regret compared to statistical and machine-learning baselines?* We introduce `ewm-engine`, an open-source, deterministic CPU simulation and counterfactual evaluation engine implementing a formal 6-part state separation $(S_t, O_t, B_t, P_t, K_t, A_t)$, KKT nullspace conservation projection, and fail-closed causal identifiability gates. Across a pre-registered 5-rung comparative ladder ($M=10$ bitemporal splits) on a 4-layer **Fraud $\to$ Declines $\to$ Call-Center Cascade**, climbing from `Rung 1` (`AR(1)`) to `Rung 3` (`Gradient Boosted Trees with Lag Features`) reduces 1-step call-arrival forecast MAE by **91.49%** (`25.60` $\to$ `2.18`), yet yields **\$0.00 reduction in policy regret** (`$124,283.57` per episode vs. `Rung 5 Full EWM`'s **\$0.00**). Conversely, we report four first-class negative results (`N1`–`N4`) demonstrating where simpler baselines win, alongside a Total Cost of Ownership (TCO) protocol (`$20,802` first-year cost, `2,089%–2,290%` simulated ROI when quarterly decision stakes exceed `$5,200.50`).

---

## 1. Formal Enterprise World Model Semantics (ADR-053)

We define an Enterprise World Model as a 10-tuple $\mathcal{W} = \langle \mathcal{S}, \mathcal{A}, \mathcal{E}, \mathcal{K}, \mathcal{O}, \mathcal{T}, \mathcal{C}, \mathcal{U}, \mathcal{Q}, \mathcal{P} \rangle$ with per-step 6-part state $(S_t, O_t, B_t, P_t, K_t, A_t)$:

$$S_{t+1} = f_{\text{known}}(S_t, A_t, E_t \mid K_t) + \mathcal{P}_{\mathcal{C}}\bigl(g_\theta(S_t, A_t, K_t)\bigr)$$

where $\mathcal{P}_{\mathcal{C}}(r) = r - A^T(AA^T)^{-1}Ar$ projects learned residuals onto the nullspace of linear conservation invariants $A S_{t+1} = b$ within IEEE-754 double-precision tolerance ($\le 10^{-12}$).

---

## 2. Empirical Results Summary

### 2.1 Pre-Registered 5-Rung Comparative Ladder (`EWME-PREREG-20261023-LADDER`)

| Rung / Model | Spec Burden | 1-Step Call MAE | Net Cash MAE ($) | SLA Violation (%) | Decision Regret ($) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Rung 1`: Simple Statistical `AR(1)`** | `1` | `25.60` | `$2,301.84` | `18.12%` | `$124,283.57` |
| **`Rung 2`: Tabular GBT (`Raw PIT`)** | `2` | `8.44` | `$1,045.12` | `18.12%` | `$124,283.57` |
| **`Rung 3`: Lag-Engineered GBT (`N1` Winner)** | `3` | **`2.18`** | **`$604.06`** | `18.12%` | `$124,283.57` |
| **`Rung 4`: Predict-Only Queue Simulator (`N2`)** | `6` | `3.45` | `$1,412.50` | `18.12%` | `$124,283.57` |
| **`Rung 5`: Full Context-Aware EWM** | `11` (`12`) | `3.32` | `$812.40` | **`0.00%`** | **`$0.00`** |

### 2.2 Enterprise Benchmark Suite (Families A–L, ADR-054)

Across 12 synthetic enterprise failure-mode families (`A_DirectDynamics` through `L_DecisionQuality`):
- **Constraint Preservation**: `1.0000` for `hybrid + runtime context discovery` vs. `0.7847` for `learned-only`.
- **Deployment Shift Recovery (Family D)**: `2` steps post-shift recovery latency (`steady_state_post_shift_mae = 0.1579`) via `AuditMetadataContextDiscovery` vs. `>34.6` unrecovered MAE for static baselines.
- **Suite Mean Decision Regret**: `0.2105` vs. `46.6619` for `learned-only` (`171.3500` vs. `0.0000` on Family L).

---

## 3. Reproducibility & Artifact Availability

All results are bit-reproducible on CPU via `python scripts/reproduce.py --require-clean` (see [`docs/results/execution_verification_report.md`](../../docs/results/execution_verification_report.md) and [`benchmarks/measurement_classification.md`](../../benchmarks/measurement_classification.md)).
