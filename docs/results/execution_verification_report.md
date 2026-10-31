# Execution Verification Report

**Date**: 2026-10-30  
**Commit**: `1d4a102d8b2d71e5496659d453a3cfaa987da6c6` (`v1.5.0`)  
**Environment**: Windows 11 (`10.0.26300-SP0`), CPython `3.12.10`, NumPy `2.5.2`, Pydantic `2.13.4`  

---

## Execution Summary

- **Command**: `python scripts/reproduce.py --require-clean`
- **Exit Code**: `0`
- **Duration**: `4.82` seconds (`2026-10-30T17:59:56.370776+00:00` to `2026-10-30T18:00:00.046663+00:00` internal harness execution)
- **Verification Checks Passed**: `true`
- **Certification Status**: `CERTIFIED_RELEASE`
- **Certification Reason**: `All verification checks passed on a clean working tree at an immutable Git commit.`
- **Source Tree SHA-256 (`287` files)**: `b9d11f85e85b751498eacd1377be9429db0016395eadef156a344bdf8990e5c6`
- **Universal Run Manifest SHA-256**: `618142f9eab6b9e04bad9f3903313fd12525672f9844ea82f9d05d9bdcc160c6`

---

## Numerical Results

| Benchmark / Suite | Metric | Executed Value | README Claim / Tolerance | Match? |
| :--- | :--- | :--- | :--- | :---: |
| **SEIR Outbreak (`seir`)** | Max mass-balance conservation drift | `4.547473508864641e-13` | $\le 10^{-12}$ (`4.55e-13`) | ✅ |
| **SEIR Outbreak (`seir`)** | Peak infections & step | `183.34932704279635` (step `79`) | `183.349` at step `79` | ✅ |
| **SEIR Outbreak (`seir`)** | 90% split-conformal coverage | `1.0` | `1.00` ($\ge 0.90$) | ✅ |
| **SEIR Outbreak (`seir`)** | Physics-IQ conservation score | `1.0` | `1.0` | ✅ |
| **Beer Game (`beer_game`)** | Bullwhip amplification ratio | `3.8024999999999998` | `3.80` (`3.8025`) | ✅ |
| **Beer Game (`beer_game`)** | Retailer / Factory peak order | `8.3640` / `16.3098` | `8.364` / `16.310` | ✅ |
| **Beer Game (`beer_game`)** | Multi-tier conservation | `VERIFIED` | `VERIFIED` | ✅ |
| **World3 (`world3`)** | Industrial capital peak step | `99` | $t = 99$ | ✅ |
| **World3 (`world3`)** | Non-renewable resource depletion | `100.0%` (`0.0` remaining) | `100.0%` | ✅ |
| **Amortized SBI (`calibrated_posterior`)** | Mean TARP calibration error | `0.028333333333333353` | `0.0283` ($\le 0.05$) | ✅ |
| **Counterfactual SCM (`counterfactual_scm`)** | Factual / Counterfactual ending stock | `15.0` / `50.0` | `15.0` / `50.0` | ✅ |
| **Counterfactual SCM (`counterfactual_scm`)** | Coupled net treatment effect | `+35.0` | `+35.0` units | ✅ |
| **Decision Discovery (`decision_discovery`)** | PRIM box density & coverage | `0.9116279069767442` / `0.8` | $\ge 0.90$ density, $\ge 0.80$ coverage | ✅ |
| **Decision Discovery (`decision_discovery`)** | Minimax regret & dominant action | `"Robust"` | `"Robust"` | ✅ |
| **Induction Gate (`induction_validation`)** | Admitted / Refused / Open questions | `7` / `2` / `2` | `7` admitted, `2` refused, `2` open | ✅ |
| **Induction Gate (`induction_validation`)** | Causal refuters passed & `"causes"` emitted | `1.0` (`3/3`) & `0` strings | `1.0` pass rate, `0` causal strings | ✅ |
| **WMIF Conformance (`ref_seed_trivial`)** | Bit-exact state fingerprint | `5dbb3db473211cc7...` | `PASSED` (`seed=42`) | ✅ |
| **WMIF Conformance (`ref_closed_form_stochastic`)** | Bit-exact state fingerprint | `7a811e8a2a765dad...` | `PASSED` (`seed=1337`, `3` branches) | ✅ |
| **WMIF Conformance (`ref_constraint_stress`)** | Bit-exact state fingerprint | `c51a1b7ed371c374...` | `PASSED` (`seed=999`) | ✅ |
| **Batched Scale Tensor (`SoAStateTensor`)** | Roundtrip state fingerprint bit-identity | `89534de1e5536adc...` | `tensor_roundtrip_bit_identical = true` | ✅ |
| **Comparative Ladder (`EWME-PREREG-20261023-LADDER`)** | Rung 1 $\to$ Rung 3 Call MAE reduction | `91.49%` (`25.60` $\to$ `2.18`) | `91.49%` (`25.60` $\to$ `2.18`) | ✅ |
| **Comparative Ladder (`EWME-PREREG-20261023-LADDER`)** | Rung 1–3 vs. Rung 5 Decision Regret | `$124,283.57` vs. `$0.00` | `$124,283.57` vs. `$0.00` (`$0.00` reduction R1$\to$R3) | ✅ |
| **Flagship Fraud Cascade (`fraud_cascade_flagship`)** | Myopic vs. Cascade EWM Regret | `$124,774.05` vs. `$0.00` | `$124,774.05` vs. `$0.00` | ✅ |
| **Enterprise Benchmark Suite (Families A–L)** | Constraint preservation (`hybrid` vs. `learned-only`) | `1.0000` vs. `0.7847` | `1.0000` vs. `0.7847` | ✅ |
| **Enterprise Benchmark Suite (Family D)** | Post-shift recovery latency & steady-state MAE | `2` steps, `MAE = 0.1579` (pre `0.1536`, peak `37.3258`) | `2` steps, `MAE = 0.158` vs. `>34.6` unrecovered | ✅ |
| **Enterprise Benchmark Suite (Families A–L)** | Suite mean decision regret (`hybrid+K_t` vs. `learned-only`) | `0.2105` vs. `46.6619` | `0.2105` vs. `46.6619` | ✅ |
| **Validity Limits (`validity_limits_evaluation`)** | Performative utility erosion & policy regret | `$107,539.60` & `$48,793.29` | `$107,539.60` erosion, `$48,793.29` regret | ✅ |
| **Validity Limits (`validity_limits_evaluation`)** | Forecast invalidation & SLA breach horizon | `2` steps & `11` steps | `2` steps & `11` steps | ✅ |

---

## Environment Details

- **Package Version**: `ewm_engine 1.5.0`
- **Python**: `3.12.10` (`CPython`)
- **NumPy**: `2.5.2`
- **Pydantic**: `2.13.4`
- **PyYAML**: `6.0.3`
- **SciPy**: `1.18.0`
- **PyTorch**: `2.13.0`
- **Platform**: `Windows-11-10.0.26300-SP0` (`AMD64`)
- **Processor**: `Intel64 Family 6 Model 142 Stepping 10, GenuineIntel`
- **Lockfile SHA-256 (`uv.lock`)**: `e7f9c755d9cffca66771676793bb6970ecc5d4dddd0a977d5f6387f479e37543`

---

## Complete Output

```text
================================================================================
ENTERPRISE WORLD MODEL ENGINE: FULL SCIENTIFIC REPRODUCTION
Git: 1d4a102d8b (main) | Python: 3.12.10 | OS: Windows
================================================================================

[PHASE 1] Executing Scientific Result Cards Harness...
  [PASS] Card 'seir': PASSED
  [PASS] Card 'beer_game': PASSED
  [PASS] Card 'world3': PASSED
  [PASS] Card 'calibrated_posterior': PASSED
  [PASS] Card 'counterfactual_scm': PASSED
  [PASS] Card 'decision_discovery': PASSED
  [PASS] Card 'induction_validation': PASSED

[PHASE 2] Executing Standalone Conformance Reference Worlds...
  [PASS] Reference World 'ref_seed_trivial': Bit-Exact Conformance
  [PASS] Reference World 'ref_closed_form_stochastic': Bit-Exact Conformance
  [PASS] Reference World 'ref_constraint_stress': Bit-Exact Conformance

[PHASE 3] Verifying Batched Scale SoA State Tensor...
  [PASS] SoA State Tensor Bit-Identity: 89534de1e5536adc...

[PHASE 4] Regenerating Visual Gallery Artifacts...
  Verified 5 vector SVG artifacts in docs/results/artifacts

[PHASE 5] Verifying ADR-053/054 Enterprise Suite, Flagship Cascade & Comparative Evaluation...
  [PASS] Layer 'enterprise_benchmark_suite': PASSED
  [PASS] Layer 'fraud_cascade_flagship': PASSED
  [PASS] Layer 'comparative_evaluation_ladder': PASSED
  [PASS] Layer 'validity_limits_evaluation': PASSED

================================================================================
Universal Run Manifest SHA-256: 618142f9eab6b9e04bad9f3903313fd12525672f9844ea82f9d05d9bdcc160c6
Verification Checks Passed:     True
Certification Status:           CERTIFIED_RELEASE
Certification Reason:           All verification checks passed on a clean working tree at an immutable Git commit.
================================================================================

Manifest written to: reproduction_manifest.json
Docs copy synced to:  docs/results/reproduction_manifest.json
```

---

## Verification Conclusion

**PASS (`CERTIFIED_RELEASE`)** — Executing `python scripts/reproduce.py --require-clean` on a clean working tree at `v1.5.0` succeeded with exit code `0` in `4.82` seconds. All 7 scientific validation cards, all 3 WMIF conformance reference worlds, the `SoAStateTensor` roundtrip bit-identity check, all 5 visual gallery SVG artifacts, and all 4 ADR-053/054 enterprise evidence layers passed within declared IEEE-754 and statistical tolerances, upgrading `reproduction_manifest.json` from `REPRODUCIBLE_DIRTY_WORKTREE` to `CERTIFIED_RELEASE`.
