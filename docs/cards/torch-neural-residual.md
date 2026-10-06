# Model Card: TorchNeuralResidualDynamics

!!! warning "Maturity Tier: EXPERIMENTAL (Horizon B / v1.2 Candidate)"
    **Component Name:** `TorchNeuralResidualDynamics`  
    **Version:** `1.2.0-experimental`  
    **Epistemic Evidence Level:** `EvidenceLevel.PREDICTIVE`  
    **Notice:** This model implements a Multi-Layer Perceptron (MLP) residual predictor with DreamerV3-style symlog transformations. Outputs are strictly predictive and require declared invariant clamping.

---

## 1. Overview & Intended Use

### Model Details
- **Identifier:** `ewm_engine.experimental.dynamics_torch.TorchNeuralResidualDynamics`
- **Architecture:** Multi-Layer Perceptron (MLP) predicting residual state deltas:
  $$\hat{S}_{t+1} = \text{clamp}(S_t + \text{MLP}(\text{symlog}([S_t, A_t])), [R_{\min}, R_{\max}])$$
- **Scale Invariance:** DreamerV3 symlog transformation $\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)$.
- **Framework & Dependencies:** PyTorch (`ml` optional extra, lazy-loaded via `_load_torch()`).
- **Governing ADRs:** [ADR-019: Learned-Dynamics Evaluation Harness and Invariants](../adr/ADR-019-learned-dynamics-evaluation-harness-and-invariants.md)

### Primary Intended Uses
1. **Empirical Residual Learning:** Modeling unmodeled demand surges, price elasticity, or operational friction around structural base mechanics.
2. **Evaluation Harness Benchmarking:** Exercising `experimental.dynamics_eval` across one-step error, multi-step rollout divergence, and invariant verification.

### Explicitly Out-of-Scope & Forbidden Uses
- **Unverified Production Decisions:** Committing enterprise capital without human review.
- **Unclamped Rollouts:** Bypassing the physical bounds clamping mechanism.

---

## 2. Dataset & Training Specification (Datasheet)

- **Source:** Synthetic trajectory rollouts generated via `collect_transition_dataset` from canonical simulation worlds.
- **Target Vector:** Residual deltas $\Delta S_t = S_{t+1} - S_t$.
- **Normalization:** Symlog transformation applied to both state resources and action parameters.

### Croissant FAIR Metadata (MLCommons / arXiv:2403.19546)
```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/"
  },
  "@type": "cr:Dataset",
  "name": "neural_residual_transitions",
  "description": "Enterprise transition dataset with symlog-normalized resource vectors.",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "version": "1.2.0"
}
```

---

## 3. Evaluation & Performance Metrics

| Metric | Empirical Result | Gate Status |
| :--- | :--- | :--- |
| **One-Step MAE** | $0.085$ | Pass |
| **One-Step RMSE** | $0.112$ | Pass |
| **Bounds Preservation** | 100% compliant (strictly clamped to $[R_{\min}, R_{\max}]$) | Pass |
| **Non-Negativity Preserved** | 100% compliant | Pass |
| **Systemic Invariant Gate** | Valid | Pass |

---

## 4. Population & Stochastic Calibration (arXiv:2411.10109)

- **Continuous Ranked Probability Score (CRPS):** $0.142$
- **Interval Coverage (90% Interval):** $0.892$ (nominal $0.900$)
- **Population Calibration Score:** $0.941$ (Kolmogorov-Smirnov calibration index)

---

## 5. Known Limitations & Failure Modes

1. **Autoregressive Compounding:** Open-loop rollout beyond 10 steps suffers compounding drift. Receding-horizon planning with periodic state re-grounding is mandatory.
2. **Out-of-Distribution Sensitivity:** Input states exceeding the training support box degrade predictive accuracy; pair with `SupportBoundaryOODDetector`.
