# Scientific Validation & Results Gallery

> **The Identity Claim**:  
> *"The only world-model engine scored on being right about its own uncertainty."*

---

## 1. Executive Summary

Most world models in artificial intelligence and simulation are evaluated purely on surface realism, pixel fidelity, or test-set Mean Squared Error (MSE). 

The **Enterprise World Model Engine (EWM)** operates under a fundamentally distinct scientific standard: **Honesty-Scored Validation**. Every benchmark is scored not merely on rollout accuracy, but on:
1. **Inviolable Conservation Laws (Physics-IQ):** Zero tolerance for hallucinated mass, energy, or money.
2. **Certified Coverage Calibration (TARP & Conformal):** Empirical coverage matching nominal confidence bounds.
3. **Causal Identifiability & Safe Testing (E-Values):** Refusal to promote unidentifiable observational correlations.
4. **Epistemic Abstention (RWM-U):** Automated down-weighting of ungrounded out-of-support extrapolations.

Every result card below is backed by a committed, seeded Python test harness and can be **reproduced bit-for-bit with a single command**.

```
                   The Six Scientific Validation Pillars
                                     │
     ┌──────────────────┬────────────┴────────────┬──────────────────┐
     ▼                  ▼                         ▼                  ▼
1. SEIR Outbreak    2. Beer Game Bullwhip     3. WORLD3 Model    4. SBI Calibration
   Bit-for-bit mass    Multi-tier inventory      Industrial         Certified TARP
   conservation.       amplification.            depletion.         coverage.
     │                                                                 │
     ▼                                                                 ▼
5. Counterfactual SCM (S01)                               6. Decision Discovery (S02)
   Twin-Rollout noise coupling.                              PRIM vulnerability subspace.
```

---

## 2. The Result Cards Gallery

| Benchmark Card | Scientific Domain & Provenance | Key Honesty Metric | Reproduction Command | Card Link |
| :--- | :--- | :--- | :--- | :--- |
| **SEIR Epidemic Wave** | Analytical epidemiology ODE benchmark ($N=1,000$) | **Mass Drift = 0.000000**<br>(Physics-IQ = 1.0) | `python scripts/reproduce_results.py --card seir` | [View Card](seir-outbreak.md) |
| **Beer Distribution Game** | Supply chain multi-tier delay & bullwhip (Sterman 1989) | **Bullwhip Ratio = 3.80**<br>(Inventory Conserved) | `python scripts/reproduce_results.py --card beer_game` | [View Card](beer-game-bullwhip.md) |
| **WORLD3 Global Limits** | Global resource dynamics (Meadows et al. 1972) | **Feedback Stability = 100%**<br>(No Numerical Drift) | `python scripts/reproduce_results.py --card world3` | [View Card](world3-resource-depletion.md) |
| **Amortized SBI Calibration** | Simulation-Based Inference (Papamakarios et al. 2019) | **TARP Error = 0.028**<br>(Certified Coverage) | `python scripts/reproduce_results.py --card calibrated_posterior` | [View Card](calibrated-posterior-s04.md) |
| **Twin-Rollout Counterfactual** | Causal SCM noise coupling (Richens & Everitt, ICLR 2024) | **Noise Coupling = 100%**<br>(Net ATE Isolated) | `python scripts/reproduce_results.py --card counterfactual_scm` | [View Card](counterfactual-scm-s01.md) |
| **Scenario Discovery (PRIM)** | Robust Decision Making & Regret (Friedman & Fisher 1999) | **Box Density = 91.2%**<br>(Coverage = 80.0%) | `python scripts/reproduce_results.py --card decision_discovery` | [View Card](decision-discovery-s02.md) |

---

## 3. Bit-for-Bit Universal Reproduction

To re-run the entire benchmark suite and verify the cryptographic state fingerprints across all six cards:

```bash
# Re-run all six scientific result cards in CI or locally
python scripts/reproduce_results.py --all
```

### Verified Run Manifest Fingerprint
```
Cryptographic Verification Fingerprint: sha256:554eb3c7988878e9f17b1274c5268a26cf3097d89b36e1210854b937aaf2d020
Overall Status: SUCCESS (All Cards Verified Bit-for-Bit)
```
