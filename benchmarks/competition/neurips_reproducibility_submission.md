# NeurIPS / ICML Reproducibility Challenge Submission Package

**Submission Title**: Reproducing the Prediction-vs-Decision Gap in Context-Aware Enterprise World Models (`EWME-PREREG-20261023-LADDER`)  
**Repository Version**: `v1.5.0` (`CERTIFIED_RELEASE`)  
**Author**: Vinicius Caridá  

---

## 1. Claims Under Independent Evaluation

| Claim ID | Claim Statement | Verification Command | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **RC-01** | Climbing from `Rung 1` (`AR(1)`) to `Rung 3` (`Lag GBT`) reduces 1-step Call MAE by `91.49%` (`25.60` $\to$ `2.18`), yet yields `$0.00` improvement in decision regret (`$124,283.57` vs. `Rung 5` `$0.00`) | `python scripts/run_comparative_evaluation.py --seed 42 --smoke` | `rung1_to_rung3_mae_reduction_pct == 91.49`, `rung5_decision_regret_usd == 0.0` |
| **RC-02** | Context-aware hybrid world model (`K_t`) recovers from deployment shift on Family D in `2` steps (`MAE = 0.1579` vs. `>34.6` static baselines) while preserving `1.0000` hard constraints across Families A–L | `python -m benchmarks.enterprise_suite --seed 42` | `steps_until_recovery == 2`, `CONSTRAINT_PRESERVATION == 1.0` |
| **RC-03** | Full 7-card scientific suite, 3 WMIF conformance worlds, and `SoAStateTensor` roundtrip achieve `CERTIFIED_RELEASE` on a clean checkout | `python scripts/reproduce.py --require-clean` | Exit code `0`, `certification_status == "CERTIFIED_RELEASE"` |

---

## 2. Evaluator Checklist (NeurIPS Reproducibility Standards)

- **Code & Data Included**: 100% self-contained in repository; zero external proprietary datasets or API keys required.
- **Compute Requirements**: Runs on any standard CPU laptop/VM in `< 60` seconds (`4.82s` for `scripts/reproduce.py`).
- **Pre-Registered Protocol**: Locked in [`docs/scientific/pre_registration_comparative_ladder.md`](../../docs/scientific/pre_registration_comparative_ladder.md).
- **Negative Results Disclosed**: `N1` (`Rung 3` beats `Rung 5` on 1-step MAE), `N2` (`Rung 4` achieves `100%` passive coherence), `N3` (uncalibrated intervals under-cover), and `N4` (`Rung 1` achieves `$0` regret in stationary single-layer regime).
