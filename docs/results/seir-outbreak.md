# Result Card: SEIR COVID-19 Epidemic Outbreak Benchmark

> **Identity & Scientific Domain**: Mathematical Epidemiology & Conservation Law Verification  
> **Benchmark Reference**: Analytical Kermack-McKendrick SEIR ODE Outbreak  
> **Reproduction Command**: `python scripts/reproduce_results.py --card seir`

---

## 1. Benchmark Overview & Physical Scenario

The Susceptible-Exposed-Infectious-Recovered (SEIR) model represents the quintessential epidemiological compartment system:
* Total Population: $N = 1,000.0$ individuals.
* Initial State: $S_0 = 990.0, E_0 = 10.0, I_0 = 0.0, R_0 = 0.0$.
* Parameters: Contact rate $\beta = 0.35$, incubation rate $\sigma = 0.20$, recovery rate $\gamma = 0.10$.
* Time Horizon: 80 steps ($\Delta t = 0.5$).

### The Non-Negotiable Invariant
At every step $t$, the total population must remain invariant:

$$S(t) + E(t) + I(t) + R(t) = N = 1,000.0$$

Any numerical solver or neural dynamics model that leaks or creates individuals fails the **Physics-IQ Guardrail**.

---

## 2. Verified Metrics & Honesty Scorecard

```json
{
  "card": "seir_outbreak",
  "steps": 80,
  "population": 1000.0,
  "peak_infections": 183.35,
  "peak_step": 79,
  "max_mass_balance_drift": 4.547e-13,
  "physics_iq_score": 1.0,
  "conformal_90_coverage": 1.0,
  "status": "PASSED"
}
```

* **Max Mass Balance Residual:** $4.547 \times 10^{-13}$ (below declared IEEE-754 tolerance $\le 10^{-12}$ across all 80 steps).
* **Physics-IQ Conservation Score:** $1.0 / 1.0$ ($100\%$ steps within tolerance).
* **Conformal Predictive Coverage:** $100\%$ containment within empirical 90% conformal bands.

---

## 3. Visual Overlay
 
![Analytical Ground Truth vs EWM Rollout](artifacts/ground_truth_overlay.svg)

Responsive hero assets:
* Dark theme: [`assets/hero-proof-dark.svg`](../assets/hero-proof-dark.svg)
* Light theme: [`assets/hero-proof-light.svg`](../assets/hero-proof-light.svg)

---

## 4. Single-Command Reproduction

```bash
python scripts/reproduce_results.py --card seir
# Or full reproduction suite:
python scripts/reproduce.py
```
