# Uncertainty Quantification: Futures as Distributions

In complex enterprise environments, the future is not a deterministic prophecy. It is a **probability distribution over trajectories**.

---

## Why Single-Trajectory Rollouts Fail

Deterministic simulations output a single line:
$$\hat{Y}_{t+1:t+H}$$
In reality:
- Consumer demand fluctuates stochastically.
- Road closures occur with probabilistic severity.
- Supplier lead times exhibit fat-tailed delays.
- Worker productivity varies with shifts and operational congestion.

A policy that performs well under the *average* scenario may fail catastrophically in the 5th percentile tail (e.g., causing fatal stockouts, bankrupting cash flow, or violating safety constraints).

---

## Monte Carlo Rollouts in EWM Engine

EWM Engine uses reproducible pseudo-random seed spawning (`numpy.random.SeedSequence`) to execute $N$ independent rollout trajectories across an experimental horizon $H$:

$$\mathcal{T} = \{\tau_1, \tau_2, \dots, \tau_N\}$$

For any target metric $M$, the engine computes full distributional statistics:

- **Central Tendency**: Mean ($\mu$) and Median ($p_{50}$).
- **Spread & Dispersion**: Standard Deviation ($\sigma$) and Interquartile Range ($\text{IQR} = p_{75} - p_{25}$).
- **Tail Risks**:
  - $p_{05}$ (5th percentile lower bound)
  - $p_{95}$ (95th percentile upper bound)
  - $\text{CVaR}_{05}$ (Conditional Value-at-Risk / Expected Shortfall in worst 5% cases)
- **Constraint Violation Probability**:
  $$P(\text{Violation}) = \frac{1}{N} \sum_{k=1}^N \mathbb{I}(\text{Violations}(\tau_k) > 0)$$

---

## Model Predictive Control (MPC) and Re-grounding

Long open-loop rollouts will naturally diverge as stochastic variance compounds.

EWM Engine advocates a **receding-horizon execution loop**:
```
    Observe Current State
              |
              v
    Simulate Short Horizon (H=12)
              |
              v
    Select Interventional Action
              |
              v
    Observe New Real-World Reality
              |
              v
    Re-ground World State S_t+1
              |
              v
           Repeat
```
This balances anticipatory simulation with continuous empirical re-grounding.
