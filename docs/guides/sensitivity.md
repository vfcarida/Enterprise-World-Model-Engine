# Global Sensitivity Analysis: Sobol & Morris Methods

This guide demonstrates how to perform global sensitivity analysis (Sobol variance decomposition and Morris elementary effects screening) via SALib in `ewm_engine.experimentation`.

---

## 1. Prerequisites & Installation

Global sensitivity analysis requires the optional `[sensitivity]` extra:

```bash
pip install 'ewm-engine[sensitivity]'
```

---

## 2. Sobol Variance-Based Sensitivity Analysis

Sobol sensitivity analysis decomposes the total variance of the simulation outcome $V(Y)$ into contributions from individual parameters and their interactions:
- **First-Order Index ($S_1$):** Direct individual contribution of parameter $X_i$ to output variance.
- **Total-Order Index ($S_T$):** Total variance caused by $X_i$, including all interactions with other parameters.
- **Second-Order Index ($S_2$):** Interaction effects between specific pairs $(X_i, X_j)$.

```python
from ewm_engine.experimentation import (
    ParameterDef,
    ParameterSpace,
    GlobalSensitivityAnalyzer,
)

# 1. Define parameter space with continuous bounds
space = ParameterSpace(
    name="SupplyChainSensitivity",
    parameters=(
        ParameterDef(name="lead_time", type="continuous", bounds=(1.0, 10.0), default=5.0),
        ParameterDef(name="holding_cost", type="continuous", bounds=(0.5, 5.0), default=2.0),
        ParameterDef(name="demand_growth", type="continuous", bounds=(0.0, 0.20), default=0.05),
    ),
)

# 2. Define simulator evaluation callback (maps point dict -> float metric)
def simulation_metric(point: dict[str, float]) -> float:
    # Example non-linear response
    return point["lead_time"] * 2.5 + (point["demand_growth"] * 100.0) ** 1.5 + point["holding_cost"]

# 3. Execute Sobol analysis
analyzer = GlobalSensitivityAnalyzer(space)
sobol = analyzer.analyze_sobol(simulation_metric, n_samples=128, calc_second_order=True, seed=42)

print("Sobol First-Order Indices (S1):")
for name in sobol.names:
    print(f"  {name:<15}: S1 = {sobol.s1[name]:.3f} +/- {sobol.s1_conf[name]:.3f}")

print("\nSobol Total-Order Indices (ST):")
for name in sobol.names:
    print(f"  {name:<15}: ST = {sobol.st[name]:.3f} +/- {sobol.st_conf[name]:.3f}")
```

---

## 3. Morris Elementary Effects Screening

For models with dozens or hundreds of parameters, Sobol analysis can require thousands of runs. Morris screening provides a computationally inexpensive screening method:
- **$\mu^*$ (Mu Star):** Overall influence/importance of the parameter.
- **$\sigma$ (Sigma):** Degree of non-linearity or interaction with other parameters.

```python
morris = analyzer.analyze_morris(simulation_metric, n_trajectories=15, num_levels=4, seed=42)

print("Morris Screening Rankings (by mu*):")
ranked_params = sorted(morris.names, key=lambda n: morris.mu_star[n], reverse=True)
for name in ranked_params:
    print(f"  {name:<15}: mu* = {morris.mu_star[name]:.3f}, sigma = {morris.sigma[name]:.3f}")
```

> [!TIP]
> **Interpreting Results:** Parameters with high $\mu^*$ and low $\sigma$ act almost linearly. Parameters with high $\mu^*$ and high $\sigma$ have non-linear or synergistic interactions. Parameters with negligible $\mu^*$ can be safely fixed to constants to simplify subsequent optimization.
