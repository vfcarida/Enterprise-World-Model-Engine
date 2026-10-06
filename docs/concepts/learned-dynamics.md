# Learned Dynamics & Empirical Evaluation

!!! warning "Experimental Subsystem (Horizon B, v1.2–v1.3)"
    Learned dynamics protocols, neural baselines, and evaluation harnesses reside strictly in `ewm_engine.experimental`.
    They require the optional `ml` extra (`pip install 'ewm-engine[ml]'`).
    Core simulation and public API contracts remain completely free of heavy machine learning dependencies.

---

## 1. The Epistemic Role of Learned Dynamics

The core design principle of the Enterprise World Model Engine is the **strict separation of what is KNOWN from what is LEARNED**:

- **What is KNOWN (Structural Core):** Hard constraints, physical conservation laws, accounting balance identities, capacity limits, and deterministic process graphs. These are modeled explicitly via `ConstraintEngine`, `RuleBasedDynamics`, and causal DAGs.
- **What is LEARNED (Empirical Residuals):** Complex human behavior, market demand elasticity, supplier lead time stochasticity, and organizational friction. These are modeled via data-driven approximations implementing `LearnedDynamics`.

In an organizational world model, machine learning models should **never replace known physics or accounting rules**. Instead, they estimate residual empirical components on top of structural baselines:

$$S_{t+1} = \text{StructuralDynamics}(S_t, A_t) + \text{LearnedResidual}(S_t, A_t, E_t)$$

---

## 2. The Research Anchor: Accuracy $\neq$ Systemic Understanding

A critical, empirically proven consensus of contemporary AI research (2024–2026) is that **high surface predictive accuracy does not imply systemic or causal understanding**:

- **Physics-IQ (arXiv:2501.09038):** Neural simulators that minimize Mean Squared Error (MSE) on one-step prediction frequently fail catastrophic invariant tests—predicting negative mass, energy creation from nothing, or teleportation when rolled out.
- **Sora Physical-Law Critique (arXiv:2411.02385):** Generative video and state models learn perceptual correlations but lack internal world representations that obey persistent physical invariants.
- **DreamerV3 (arXiv:2301.04104):** Heterogeneous scales across observations require scale-invariant transformations (symlog) to prevent high-magnitude signals from dominating gradient updates.

Because of this finding, **evaluating a learned dynamics model by Mean Absolute Error (MAE) or $R^2$ alone is scientifically irresponsible**. An enterprise model that predicts corporate profits within \$1,000 but permits negative inventory or balance sheet violations is worse than useless: it is dangerously deceptive.

---

## 3. The Mandatory Invariant Consistency Gate (ADR-019)

To prevent ungrounded models from passing evaluation, `experimental.dynamics_eval` enforces an **Invariant Consistency Gate** as a mandatory prerequisite for model acceptance.

```
Evaluated Transitions (S_t, A_t) ──► Dynamics Model ──► Predicted Next State S_{t+1}
                                                                  │
                    ┌─────────────────────────────────────────────┴───────────────────────────────────┐
                    ▼                                                                                 ▼
     Systemic Invariant Verification                                                 Empirical Prediction Error
  - Bound preservation [min, max]                                                 - Mean Absolute Error (MAE)
  - Resource non-negativity (c >= 0)                                              - Root Mean Squared Error (RMSE)
  - Conservation laws (sum delta = 0)                                             - Symlog scale-invariant error
                    │                                                                                 │
                    ▼                                                                                 ▼
        Invariant Passed? (True/False)                                                   Numeric Error Acceptable?
                    │                                                                                 │
                    └──────────────────────────────────────┬──────────────────────────────────────────┘
                                                           ▼
                                                Overall Model Validity
                                         (Requires Invariant Gate == PASS)
```

If a dynamics model generates transitions that violate declared bounds or conservation equations:
1. `report.invariants.passed` evaluates to `False`.
2. Every violation is categorized, counted, and reported in `report.invariants.violations`.
3. `report.overall_valid` is flagged as `False`, **regardless of how low the numeric MSE is**.

---

## 4. DreamerV3 Symlog Normalization

Enterprise systems present severe scale heterogeneity:
- Cash reserves: $\$10^7$
- Product unit prices: $\$10^2$
- Warehouse staffing: $10^1$
- Defect rate: $10^{-3}$

Standard min-max or z-score normalization fails when scenarios experience regime shifts or non-stationary growth. Following Hafner et al. (DreamerV3, 2023), the EWM Engine implements the **symlog** (symmetric logarithm) and **symexp** functions:

$$\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)$$

$$\text{symexp}(y) = \text{sign}(y) \cdot (\exp(|y|) - 1)$$

### Key Properties:
- Continuous, smooth, and invertible everywhere.
- Linear behavior around zero: $\text{symlog}(x) \approx x$ for $|x| \ll 1$.
- Logarithmic compression for large values: $\text{symlog}(x) \approx \text{sign}(x) \cdot \ln(|x|)$ for $|x| \gg 1$.
- Enables neural models to learn across monetary, operational, and headcount dimensions simultaneously without gradient explosion.

---

## 5. The Five Pillars of the Evaluation Harness

The evaluation harness in `ewm_engine.experimental.dynamics_eval` evaluates any `DynamicsModel` across five orthogonal criteria:

### 1. Invariant Integrity (The Gate)
Runs every generated transition against declared resource bounds and specified conservation groups. Reports violation counts, affected entities, and pass/fail gate status.

### 2. One-Step Predictive Error
Measures prediction error across one-step transitions $(S_t, A_t \to S_{t+1})$:
- Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Per-resource breakdown to identify neglected minority metrics.
- Mean Absolute Percentage Error (MAPE) where non-zero.
- Aggregate Symlog Error for scale-invariant cross-model comparison.

### 3. Autoregressive Rollout Divergence
Evaluates compounding multi-step drift over multi-step horizons:
- Generates trajectories using the candidate dynamics in place of reference ground truth.
- Measures trajectory-level Mean Absolute Error and Dynamic Time Warping (DTW) distance over $H$ steps.

### 4. Stochastic Calibration (Reusing Uncertainty Tools)
For probabilistic or stochastic models (sampling $K$ transitions per state):
- Continuous Ranked Probability Score (CRPS) computed via energy kernel representation.
- Empirical coverage of $[p_{05}, p_{95}]$ confidence intervals (nominal 90% coverage).

### 5. Interventional Shift Detection
Evaluates the model on an out-of-distribution (interventional) action dataset:
- Measures the interventional generalization gap: $\text{Gap} = MAE_{\text{interventional}} - MAE_{\text{in-distribution}}$.
- Proves whether the harness detects out-of-distribution distribution shift ($\text{Ratio} > 1.15$).

---

## 6. Using the Neural Baseline

```python
from ewm_engine.experimental import (
    TorchNeuralResidualDynamics,
    collect_transition_dataset,
    evaluate_dynamics_model,
)

# 1. Collect transitions from a known simulation world
dataset = collect_transition_dataset(world, scenario, steps_per_rollout=20, n_rollouts=10)
train_ds, test_ds = dataset.split(train_ratio=0.8, seed=42)

# 2. Train baseline (requires 'ml' extra)
model = TorchNeuralResidualDynamics(
    target_resources=["cash", "inventory"],
    action_types=["purchase", "fulfill"],
    epochs=100,
    seed=42,
)
model.fit(train_ds)

# 3. Evaluate with full harness
report = evaluate_dynamics_model(
    model=model,
    dataset=test_ds,
    reference_world=world,
    rollout_scenario=scenario,
)

assert report.invariants.passed, "Model failed invariant gate!"
print(f"One-step Symlog MAE: {report.one_step.symlog_error:.4f}")
if report.rollout:
    print(f"Rollout Divergence MAE: {report.rollout.trajectory_mae:.4f}")
```
