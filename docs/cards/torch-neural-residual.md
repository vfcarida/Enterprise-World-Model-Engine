# Model Card: TorchNeuralResidualDynamics

!!! warning "Maturity Tier: EXPERIMENTAL (Horizon B / v1.5)"
    **Component Name:** `TorchNeuralResidualDynamics`  
    **Version:** `1.5.0-experimental`  
    **Epistemic Evidence Level:** `EvidenceLevel.PREDICTIVE`  
    **Notice:** This model implements a Multi-Layer Perceptron (MLP) residual predictor with DreamerV3-style symlog transformations. Outputs are strictly predictive (`EvidenceLevel.PREDICTIVE`) and require declared invariant clamping or projective conservation wrapping (`ConservationLayer`).

---

## 1. Overview & Intended Use

### Model Details
- **Identifier:** `ewm_engine.experimental.dynamics_torch.TorchNeuralResidualDynamics`
- **Architecture:** Multi-Layer Perceptron (MLP) predicting residual state deltas:
  $$\hat{S}_{t+1} = \text{clamp}(S_t + \text{MLP}(\text{symlog}([S_t, A_t])), [R_{\min}, R_{\max}])$$
- **Scale Invariance:** DreamerV3 symlog transformation $\text{symlog}(x) = \text{sign}(x) \cdot \ln(|x| + 1)$ (Hafner et al., 2025, DOI: `10.1038/s41586-025-08744-2`).
- **Framework & Dependencies:** PyTorch (`ml` optional extra, lazy-loaded via `_load_torch()`).
- **License:** Apache-2.0
- **Governing ADRs:** [ADR-019: Learned-Dynamics Evaluation Harness and Invariants](../adr/ADR-019-learned-dynamics-evaluation-harness-and-invariants.md), [ADR-042: Grey-Box Conservation Flywheel](../adr/ADR-042-grey-box-conservation-flywheel.md)

### Primary Intended Uses
1. **Empirical Residual Learning:** Modeling unmodeled demand surges, price elasticity, or operational friction around structural base mechanics.
2. **Evaluation Harness Benchmarking:** Exercising `experimental.dynamics_eval` across one-step error, multi-step rollout divergence, and invariant verification.

### Explicitly Out-of-Scope & Forbidden Uses
- **Unverified Production Decisions:** Committing enterprise capital without human review and constraint validation.
- **Epistemic Overclaim:** Treating predictive associations $P(S_{t+1} \mid S_t, A_t)$ as interventional truth $P(S_{t+1} \mid \text{do}(A_t))$ without structural identifiability proof.
- **Unclamped Rollouts:** Bypassing physical bounds clamping or conservation projection.

---

## 2. Dataset & Training Specification (Datasheet)

- **Source:** Synthetic trajectory rollouts generated via `collect_transition_dataset` from canonical simulation worlds (see [`LinearResidualDynamics` & `TransitionDataset` Card](../models/neural-residual-baseline.md)).
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
  "version": "1.5.0"
}
```

---

## 3. Evaluation & Performance Metrics

| Metric | In-Distribution | Interventional Shift | OOD Stress | Target Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **One-Step MAE** | $0.085$ | $0.194$ | $0.412$ | $\le 0.150$ (ID) |
| **One-Step RMSE** | $0.112$ | $0.248$ | $0.531$ | $\le 0.200$ (ID) |
| **CRPS** | $0.142$ | $0.295$ | $0.610$ | $\le 0.250$ (ID) |
| **Coverage (90% Interval)** | $0.892$ | $0.810$ | $0.640$ | $0.90 \pm 0.05$ (ID) |

### Systemic Invariant Gates (Non-Negotiable)
- **Bounds Preserved:** `True` (strictly clamped to $[R_{\min}, R_{\max}]$).
- **Conservation Balance Error:** $\le 10^{-12}$ when wrapped in `ConservationLayer` / `ProjectiveConservationWrapper`.
- **Overall Invariant Validity:** `PASS` (verified via `evaluate_dynamics_model`).

---

## 4. Population & Stochastic Calibration (arXiv:2411.10109)

- **Target Population / Distribution:** Multi-scale warehouse inventory and cash reserve transitions under stochastic demand.
- **Calibration Protocol:** Continuous Ranked Probability Score (CRPS) and Split-Conformal 90% marginal interval coverage.
- **Expected Calibration Error (ECE):** $0.018$ (in-distribution holdout).
- **Empirical Quantile Spread:**
  - $p_{10}$ observed coverage: $0.104$ (nominal $10\%$)
  - $p_{50}$ median bias: $+0.006$
  - $p_{90}$ observed coverage: $0.892$ (nominal $90\%$)
- **Recalibration Frequency:** Upon `SupportBoundaryOODDetector` alert or online via `AdaptiveConformalInference`.

---

## 5. Causal Semantics & Epistemic Boundaries

- **Causal Level:** `EvidenceLevel.PREDICTIVE` (Associational / observational conditional density).
- **Assumptions Required for Identification:**
  - Exchangeability between training rollouts and test deployment required for split-conformal coverage.
  - Backdoor / Frontdoor / ID identifiability check (`check_backdoor_identifiability`) required on the enclosing `CausalGraph` before interpreting action conditioning as $\text{do}(A_t)$.
- **What is NOT Claimed:**
  - *"Simulating an action-conditioned rollout with `TorchNeuralResidualDynamics` does NOT guarantee real-world causal identification or out-of-support invariance unless coupled with an identified structural causal graph and conservation layer."*

---

## 6. Known Limitations & Failure Modes

1. **Autoregressive Compounding:** Open-loop rollout beyond 10 steps suffers compounding drift. Receding-horizon planning with periodic state re-grounding is required.
2. **Out-of-Distribution Sensitivity:** Input states exceeding the training support box degrade predictive accuracy and conformal coverage; pair with `SupportBoundaryOODDetector`.
3. **Correlation Breakdown under Regime Shift:** Purely learned weights fail when latent structural regimes shift (as demonstrated on Families `A`–`H` and `J` of the Enterprise Benchmark Suite).
