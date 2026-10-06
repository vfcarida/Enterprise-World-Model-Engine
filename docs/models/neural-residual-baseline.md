# Model Card: TorchNeuralResidualDynamics Baseline

!!! danger "Experimental Component — Not for Production Decisions"
    **This model is experimental (`Maturity: Experimental`, Horizon B).**
    It is provided exclusively for research benchmarking and to exercise the `LearnedDynamics` evaluation harness.
    Outputs have `EvidenceLevel = PREDICTIVE`. It must **never** be used as an unverified decision-making engine or promoted to production without strict constraint verification (ADR-019).

---

## 1. Model Details

- **Model Name:** `TorchNeuralResidualDynamics`
- **Architecture:** Multi-Layer Perceptron (MLP) with Residual Delta Prediction and DreamerV3 Symlog Scaling
- **Framework:** PyTorch (lazily loaded via `pip install 'ewm-engine[ml]'`)
- **Version:** 0.1.0-experimental
- **License:** Apache-2.0
- **Governing ADR:** [ADR-019: Learned-Dynamics Evaluation Harness and Systemic Invariants](../adr/ADR-019-learned-dynamics-evaluation-harness-and-invariants.md)
- **Epistemic Classification:** `EvidenceLevel.PREDICTIVE`

---

## 2. Intended Use & Scope

### Primary Uses
1. **Protocol Verification:** Exercise and prove the `LearnedDynamics` protocol (`fit`, `transition`) without coupling the EWM core to machine learning libraries.
2. **Evaluation Harness Benchmarking:** Serve as an empirical test subject for `experimental.dynamics_eval` across one-step error, multi-step rollout divergence, and invariant verification.
3. **Distribution Shift Prototyping:** Provide an empirical baseline to measure the interventional generalization gap ($MAE_{\text{interventional}} - MAE_{\text{in-distribution}}$).

### Explicitly Out-of-Scope & Forbidden Uses
- **Autonomous Decision Execution:** Using unverified neural rollout trajectories to commit high-stakes enterprise capital, staffing, or inventory decisions.
- **Unconstrained Dynamic Rollouts:** Bypassing the systemic invariant gate or removing output clamping.
- **Causal Claims Without Identification:** Conflating predictive association ($P(S_{t+1}|S_t, A_t)$) with empirical counterfactual interventional truth ($P(S_{t+1}|\text{do}(A_t))$).

---

## 3. Training Data & Representation

- **Source:** Synthetic trajectory rollouts generated via `collect_transition_dataset` from canonical organizational simulation worlds (e.g., warehouse fulfillment, SaaS subscription economics).
- **Dataset Structure:** In-memory `TransitionDataset` containing tuples $(S_t, A_t, E_t, S_{t+1})$.
- **Feature Preprocessing:**
  - States and action parameters are normalized using the DreamerV3 **symlog** transformation:
    $$\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)$$
    This compresses heterogeneous enterprise scales (e.g., \$10,000,000 cash reserves vs. 15 warehouse staff) into a balanced numerical manifold without distorting zero-crossing signals.
- **Target Vector:** Residual deltas $\Delta s_t = s_{t+1} - s_t$.

---

## 4. Architecture & Safeguards

```
Current State s_t ──► [ symlog ] ──┐
                                   ├──► [ Linear(d_in, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, d_out) ] ──► Predicted Delta Δs_t
Action Vector a_t ──► [ symlog ] ──┘                                                                                           │
                                                                                                                               ▼
                                                  Clamped Next State: s_{t+1} = clamp(s_t + Δs_t, [min_val, max_val]) ◄───────┘
```

### Safety & Invariant Safeguards
- **Declared Bound Clamping:** Predicted state updates are passed through `state.update_resource(rid, delta=delta, clamp=True)`, ensuring that state non-negativity and maximum capacity limits are strictly enforced.
- **Invariant Gate Evaluation:** The model must be evaluated through `evaluate_dynamics_model()`, which checks for bound preservation, non-negativity, and conservation laws. Models that breach invariants are flagged as `overall_valid = False` regardless of low MSE.

---

## 5. Limitations & Known Failure Modes

1. **Compounding Rollout Drift:**
   While one-step prediction error may be minimal ($MAE < 0.05$), errors compound exponentially during multi-step autoregressive rollouts ($S_0 \to S_1 \to S_2 \dots \to S_H$). Without structural feedback, the simulated trajectory quickly drifts from physical plausibility.
2. **Interventional Hallucination (Correlation $\neq$ Causation):**
   The neural network fits empirical conditional distributions observed in training data. Under out-of-distribution interventions (e.g., actions 5x larger than historical bounds), the model produces confident but physically nonsensical predictions. The evaluation harness detects this through the interventional shift gap.
3. **Structural Law Ignorance:**
   Unlike structural simulation models, neural weights have no inherent knowledge of mass conservation, double-entry bookkeeping, or organizational queue topology. Bound clamping prevents overflow, but internal flow balance is not guaranteed without external verification.
4. **Scoped Reproducibility:**
   CPU execution is deterministic given a fixed seed (`torch.manual_seed(seed)`). GPU or parallel BLAS multi-threading may introduce non-deterministic floating-point discrepancies across platforms.

---

## 6. How to Run Evaluation

```python
from ewm_engine.experimental import (
    TorchNeuralResidualDynamics,
    collect_transition_dataset,
    evaluate_dynamics_model,
)

# 1. Collect synthetic transitions
dataset = collect_transition_dataset(world, scenario, steps_per_rollout=20, n_rollouts=10)
train_ds, test_ds = dataset.split(train_ratio=0.8, seed=42)

# 2. Train baseline
model = TorchNeuralResidualDynamics(
    target_resources=["inventory", "cash"],
    action_types=["restock", "fulfill"],
    epochs=100,
    seed=42,
)
model.fit(train_ds)

# 3. Evaluate with systemic invariant gate
report = evaluate_dynamics_model(
    model=model,
    dataset=test_ds,
    conservation_groups=[("cash", "inventory")],
)

print(f"Overall Valid: {report.overall_valid}")
print(f"Invariants Passed: {report.invariants.passed}")
print(f"One-Step Symlog Error: {report.one_step.symlog_error:.4f}")
```
