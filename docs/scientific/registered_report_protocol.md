# Pre-Registered Study Protocol: Deterministic World-Model Verification

> **Format**: Stage 1 Registered Report Protocol  
> **Study Title**: *Structural Causal Models vs. Unconstrained Neural Dynamics: An Empirical Audit of Invariant Preservation and Rollout Drift in Organizational Simulation*  
> **Principal Investigator**: Vinicius Caridá (`vfcarida@gmail.com`)  
> **Registration Platform**: Open Science Framework (OSF) / Center for Open Science  
> **Date of Pre-Registration**: October 2026  

---

## 1. Research Questions & Hypotheses

This protocol pre-registers an empirical audit comparing **Structural World Models (SWM)** with unconstrained neural baseline dynamics:

### Hypothesis 1: Mass-Balance Conservation (H1)
- **Claim**: Under algebraic nullspace projection $\mathcal{P}_{\text{null}}$, cumulative mass-balance drift across $H=80$ simulation steps remains bounded by numerical floating-point epsilon ($|\text{drift}| \le 10^{-12}$), whereas unconstrained neural dynamics exhibit super-linear drift ($|\text{drift}| > 1.0$).
- **Statistical Test**: Two-sided Kolmogorov-Smirnov test against zero-drift oracle.

### Hypothesis 2: Counterfactual Treatment Coupling (H2)
- **Claim**: Twin-Rollout exogenous noise abduction isolates the true causal intervention effect $\Delta \tau = \tau_{\text{cf}} - \tau_{\text{fact}}$ with variance strictly lower than decoupled independent Monte Carlo sampling ($\text{Var}(\Delta \tau_{\text{coupled}}) < 0.20 \cdot \text{Var}(\Delta \tau_{\text{uncoupled}})$).
- **Statistical Test**: Levene's test for equality of variance ($\alpha = 0.01$).

### Hypothesis 3: Posterior Coverage Calibration (H3)
- **Claim**: Amortized Simulation-Based Inference (SBI) gated by TARP coverage diagnostic satisfies nominal coverage within $|\Delta \alpha| \le 0.05$ across all credibility levels $\alpha \in [0.1, 0.9]$.
- **Statistical Test**: Uniform Kolmogorov-Smirnov goodness-of-fit on probability integral transform (PIT) values.

### Hypothesis 4: Causal Refusal Integrity (H4)
- **Claim**: When simulated observational data is contaminated with synthetic unobserved confounding ($U \to X, U \to Y$), the engine refuses point causal ranking with $100\%$ precision, rejecting auto-upgraded claims.
- **Metric**: Zero false-positive causal certificates on non-identifiable DAGs.

---

## 2. Experimental Data & Benchmarks

The pre-registered evaluation protocol evaluates six canonical dynamic systems:

1. **SEIR Epidemic System**: Kermack-McKendrick compartmental differential equations ($N = 1000$, $\beta = 0.35$, $\gamma = 0.10$).
2. **Beer Distribution Game**: Sterman (1989) 4-tier supply chain under customer order surge ($4 \to 8$ cases).
3. **WORLD3 Resource Model**: Meadows et al. global industrial and natural resource feedback loop ($t=0 \dots 120$).
4. **Synthetic Warehouse Logistics**: Relational logistics graph under worker absenteeism shocks.
5. **Bayesian Linear-Gaussian SBI**: Known ground-truth posterior for TARP diagnostic validation.
6. **OCEL 2.0 Process Event Log**: Synthetic organizational purchase-to-pay event log with injected unobserved confounders.

---

## 3. Sampling Plan & Stopping Rules

- **Random Number Generation**: NumPy `SeedSequence` with fixed initial base seed (`42`).
- **Sample Size**:
  - Point estimates: $N = 1000$ simulation runs per condition.
  - Multi-seed evaluations: $K = 20$ independent PRNG branches.
- **Stopping Rule**: All planned rollouts are executed to completion; optional stopping is strictly prohibited. Hypothesis testing uses anytime-valid E-values ($\mathbb{E}_{H_0}[E] \le 1$) to guarantee false-discovery rate control under sequential monitoring.

---

## 4. Analysis Pipeline & Quality Gates

The analysis is automated via `scripts/reproduce.py`:
- No manual data cleaning or subjective outlier exclusion.
- All code executed in a hermetic container via `Dockerfile` and `uv.lock`.
- Results automatically tabulated in `reproduction_manifest.json`.
- Deviations from pre-registered thresholds must be reported explicitly in Stage 2 submission.
