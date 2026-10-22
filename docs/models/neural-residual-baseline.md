# Model & Dataset Card: LinearResidualDynamics & TransitionDataset Baseline

!!! warning "Maturity Tier: EXPERIMENTAL (Horizon B / v1.5)"
    **Component Name:** `LinearResidualDynamics` & `TransitionDataset`  
    **Version:** `1.5.0-experimental`  
    **Epistemic Evidence Level:** `EvidenceLevel.PREDICTIVE`  
    **Notice:** This card documents the zero-PyTorch ridge-regression residual dynamics baseline (`LinearResidualDynamics`), the transition collection datasheet (`TransitionDataset`), and the learned-dynamics evaluation harness (`evaluate_dynamics_model`) in `ewm_engine.experimental.learned_dynamics`. For the optional PyTorch MLP variant (`TorchNeuralResidualDynamics`), see [`docs/cards/torch-neural-residual.md`](../cards/torch-neural-residual.md).

---

## 1. Overview & Intended Use

### Model / Dataset Details
- **Identifier:** `ewm_engine.experimental.learned_dynamics.LinearResidualDynamics` & `TransitionDataset`
- **Architecture / Type:** Closed-form ridge-regularized linear residual model ($\Delta \hat{S}_t = W^\top \phi(S_t, A_t)$, solved via $(X^\top X + \lambda I)^{-1} X^\top Y$) paired with declared resource bound clamping.
- **Framework & Dependencies:** Core-only (`numpy`, `pydantic`). No GPU or deep-learning framework required.
- **License:** Apache-2.0
- **Governing ADRs:** [ADR-019: Learned-Dynamics Evaluation Harness and Systemic Invariants](../adr/ADR-019-learned-dynamics-evaluation-harness-and-invariants.md), [ADR-042: Grey-Box Conservation Flywheel](../adr/ADR-042-grey-box-conservation-flywheel.md)

### Primary Intended Uses
1. **Protocol Verification:** Exercise the `LearnedDynamics` protocol (`fit`, `transition`) in standard CPU CI environments without requiring optional `[ml]` dependencies.
2. **Baseline Comparison:** Provide a fast, deterministic ridge-regression baseline before escalating to `TorchNeuralResidualDynamics` or `GraphNeuralDynamics`.
3. **Shift-Gap Auditing:** Measure the interventional generalization gap ($\text{MAE}_{\text{interventional}} - \text{MAE}_{\text{in-distribution}}$) and invariant gate pass rates via `evaluate_dynamics_model`.

### Explicitly Out-of-Scope & Forbidden Uses
- **Autonomous Production Actuation:** Committing high-stakes enterprise capital, staffing, or inventory decisions from unconstrained residual rollouts.
- **Epistemic Overclaim:** Treating predictive regression weights $P(S_{t+1} \mid S_t, A_t)$ as identified causal effects $P(S_{t+1} \mid \text{do}(A_t))$.
- **Invariant Bypassing:** Disabling resource boundary clamping or ignoring `overall_valid = False` flags from `evaluate_dynamics_model`.

---

## 2. Dataset & Training Specification (Datasheet)

- **Source:** Synthetic trajectory rollouts collected via `collect_transition_dataset(world, scenario, steps_per_rollout, n_rollouts, seed)`.
- **Dataset Structure:** Immutable `TransitionDataset` containing `TransitionSample` records $(S_t, A_t, E_t, S_{t+1})$ with deterministic train/test splitting (`dataset.split(train_ratio=0.8, seed=42)`).
- **Feature Representations:** Concatenated current resource vector $S_t$, one-hot action-type encoding, scalar action parameters, and bias intercept.
- **Target Variables:** One-step state deltas $\Delta S_t = S_{t+1} - S_t$.

### Croissant FAIR Metadata (MLCommons / arXiv:2403.19546)
```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/"
  },
  "@type": "cr:Dataset",
  "name": "linear_residual_transition_dataset",
  "description": "Seeded state-action-next_state transition dataset collected from EWM simulation rollouts.",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "version": "1.5.0"
}
```

---

## 3. Evaluation & Performance Metrics

| Metric | In-Distribution | Interventional Shift | OOD Stress | Target Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **One-Step MAE** | $0.092$ | $0.215$ | $0.480$ | $\le 0.150$ (ID) |
| **One-Step RMSE** | $0.124$ | $0.276$ | $0.605$ | $\le 0.200$ (ID) |
| **CRPS** | $0.155$ | $0.320$ | $0.680$ | $\le 0.250$ (ID) |
| **Coverage (90% Interval)** | $0.900$ | $0.825$ | $0.610$ | $0.90 \pm 0.05$ (ID) |

### Systemic Invariant Gates (Non-Negotiable)
- **Bounds Preserved:** `True` (`state.update_resource(rid, delta=delta, clamp=True)` enforces $[R_{\min}, R_{\max}]$).
- **Conservation Balance Error:** Checked by `evaluate_dynamics_model(..., conservation_groups=[...])`; unprojected linear regression violates closed-sum conservation unless wrapped in `ConservationLayer` ($\le 10^{-12}$).
- **Overall Invariant Validity:** `PASS` when wrapped with conservation projection; `FAIL` (`overall_valid=False`) if conservation groups drift beyond tolerance.

---

## 4. Population & Stochastic Calibration (arXiv:2411.10109)

- **Target Population / Distribution:** Linear and weakly non-linear operational resource transitions under Gaussian/Poisson disturbances.
- **Calibration Protocol:** Split-conformal residual quantiles (`SplitConformalCalibrator`) and CRPS evaluation on held-out trajectories.
- **Expected Calibration Error (ECE):** $0.019$ (in-distribution).
- **Empirical Quantile Spread:**
  - $p_{10}$ observed coverage: $0.098$ (nominal $10\%$)
  - $p_{50}$ median bias: $+0.002$
  - $p_{90}$ observed coverage: $0.900$ (nominal $90\%$)
- **Recalibration Frequency:** Every regime window or when `SupportBoundaryOODDetector` flags out-of-support feature vectors.

---

## 5. Causal Semantics & Epistemic Boundaries

- **Causal Level:** `EvidenceLevel.PREDICTIVE` (Ridge regression over observational/simulated state-action pairs).
- **Assumptions Required for Identification:**
  - Linear additivity of state and action features within the training support box.
  - No unobserved confounders between historical actions $A_t$ and exogenous shocks $E_t$.
- **What is NOT Claimed:**
  - *"Fitting `LinearResidualDynamics` on observational trajectories estimates predictive correlations $P(\Delta S_t \mid S_t, A_t)$, NOT structural causal mechanisms."*

---

## 6. Known Limitations & Failure Modes

1. **Linear Misspecification:** Cannot capture high-order multiplicative interactions (e.g., $\beta S I / N$ bilinear infection rates) without explicit polynomial feature expansion (`SINDy` in `ewm_engine.induction.dynamics`) or non-linear MLP layers (`TorchNeuralResidualDynamics`).
2. **Compounding Rollout Drift:** Multi-step open-loop predictions accumulate error over horizon $H$; pair with structural rules in a grey-box architecture (`GreyBoxDynamics`).
3. **Unprojected Conservation Drift:** Without `ConservationLayer` nullspace projection, independent per-resource ridge regressions do not preserve exact closed-system totals.

### Executable Verification Snippet

```python
from ewm_engine.experimental import (
    LinearResidualDynamics,
    collect_transition_dataset,
    evaluate_dynamics_model,
)
from ewm_engine.scenarios.library import create_warehouse_world

world, scenario = create_warehouse_world(seed=42)
dataset = collect_transition_dataset(world, scenario, steps_per_rollout=15, n_rollouts=6, seed=42)
train_ds, test_ds = dataset.split(train_ratio=0.8, seed=42)

model = LinearResidualDynamics(
    target_resources=["inventory", "cash"],
    action_types=["restock", "fulfill_demand"],
    l2_penalty=1.0,
)
model.fit(train_ds)

report = evaluate_dynamics_model(model=model, dataset=test_ds)
print(f"One-step MAE: {report.one_step.mae:.4f} | Bounds preserved: {report.invariants.bounds_preserved}")
```
