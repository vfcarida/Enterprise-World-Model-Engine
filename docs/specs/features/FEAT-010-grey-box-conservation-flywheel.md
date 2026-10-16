# FEAT-010: Grey-Box Conservation Layer & Neuro-Symbolic Flywheel

- **Status:** Approved / Implemented
- **Horizon:** v1.8.0 "Grey-Box Conservation Layer & Neuro-Symbolic Flywheel"
- **Authors:** Hybrid Dynamics, Causal Systems & Verification Teams
- **Governance:** [frontier-roadmap.md](../../architecture/frontier-roadmap.md) (Grey-Box Dynamics), ADR-042, ACP-011
- **Dependencies:** Core engine (Zero new mandatory dependencies: stdlib, NumPy, Pydantic); Optional extras: `[ml]` (torch), `[symbolic]` (pysr)

---

## 1. Motivation & Context

The Enterprise World Model Engine's core value proposition is the **separation of what is KNOWN from what is LEARNED**:
- **KNOWN:** Physical laws, accounting balance identities, organizational inventory constraints, and deterministic routing.
- **LEARNED:** Empirical human behavior, stochastic disruptions, friction, and market demand drift.

However, existing machine learning simulators either:
1. Attempt to learn transitions from scratch in black-box neural networks, which hallucinate impossible states (e.g., creating capital or inventory out of thin air).
2. Attempt to enforce conservation laws via soft penalty functions or post-hoc state rejection, both of which permit non-zero violation rates under distribution shift.

FEAT-010 operationalizes the state-of-the-art hybrid-dynamics literature (Rackauckas et al., 2020; KKT-hPINN, 2024; Port-Hamiltonian NNs, 2025; SINDy, 2016; LLM-SR, 2025):
- **Universal Differential Equation (UDE) Grey-Box Framing:** Learned models predict **only the residual/delta** over the explicit known ruleset, never the full next-state. When the residual is zero, the model reduces bit-for-bit to the symbolic kernel (S00 Invariant 5).
- **Architectural Conservation Layer:** Exact KKT nullspace projection layers ($P = I - A^T (A A^T)^{-1} A$) guarantee **0.0% constraint violation by construction** for any neural output.
- **Neuro-Symbolic Flywheel:** SINDy STLSQ sparse polynomial regression fits closed-form symbolic expressions to empirical residuals, enabling an audited, human-in-the-loop promotion workflow to move mass from **LEARNED $\to$ KNOWN** (the self-whitening grey-box).

---

## 2. Requirements & Acceptance Criteria

### Sub-track 1: Residual-Only Learned Dynamics
- **AC-066 (UDE Residual Interface & Invariant 5 Bit-Identity):**
  - Implement `GreyBoxDynamics` conforming to `DynamicsModel`.
  - Reframe learned dynamics to estimate only residual deltas $\Delta_\theta(s, a, e)$ over the output of `known_model: DynamicsModel`.
  - Strict Zero-Residual Invariant: When the learned residual is absent or zero, the next state and its cryptographic SHA-256 fingerprint must be bit-for-bit identical to the symbolic kernel (`assert next_state.fingerprint == known_next_state.fingerprint`).
  - Epistemic Status: Transitions with zero residual retain `EvidenceLevel.STRUCTURAL`. Applying empirical residuals shifts the step result to `EvidenceLevel.PREDICTIVE`.
  - Implement `ResidualEstimator` protocol with `ZeroResidualEstimator`, `CallableResidualEstimator`, and `LinearResidualEstimator`.

### Sub-track 2: Architectural Conservation Layer
- **AC-067 (Exact Conservation Projection Layer):**
  - Implement `LinearConservationLaw` capturing linear equality constraints ($A \Delta = \mathbf{b}$), with factory helpers for zero-sum flows ($\sum \Delta_i = 0$) and double-entry balance ($\sum \text{debit} - \sum \text{credit} = 0$).
  - Implement `ConservationProjectionLayer` computing the Moore-Penrose / KKT orthogonal projection onto the nullspace of $A$:
    $$\Delta_{\text{proj}} = \Delta - A^T (A A^T)^{-1} (A \Delta - \mathbf{b})$$
  - Include boundary scaling preserving $A (\alpha \Delta_{\text{proj}}) = \mathbf{0}$ while strictly obeying $[r_{\min}, r_{\max}]$ bounds.
  - Implement `PortHamiltonianConservationLayer` parameterizing flows via $\Delta = (J - R) \nabla H$ with skew-symmetric $J = -J^T$ and positive semi-definite $R \succeq 0$, guaranteeing Lyapunov stability ($\dot{H} \le 0$).
- **AC-068 (Honesty Metric & Structural Zero Violation Report):**
  - Implement `evaluate_conservation_honesty(...)` producing `ConservationHonestyReport`.
  - Report must confirm that the grey-box equipped with the conservation projection layer achieves a `violation_rate == 0.0` (structurally zero), contrasted with non-zero violation rates on unconstrained learned baselines.

### Sub-track 3: Neuro-Symbolic Flywheel & Audited Promotion
- **AC-069 (SINDy Sparse Symbolic Regression Engine):**
  - Implement `SINDyRegressor` using Sequential Thresholded Least Squares (STLSQ) in pure NumPy with candidate polynomial libraries up to degree 2/3.
  - Compute normalized column projections for scale-invariant sparse term selection.
  - Score candidate equations by $R^2$, MSE, Bayesian Information Criterion (BIC), and parsimony score $k$.
- **AC-070 (Advisory Equation Priors & LLM-SR Advisory Role):**
  - Implement `AdvisoryEquationPrior` capturing external hypothesis skeletons (e.g. proposed by LLM-SR program synthesis or human experts).
  - Enforce that advisory priors are non-binding and never execute directly in the simulation core.
- **AC-071 (Audited Promotion Workflow: LEARNED $\to$ KNOWN):**
  - Implement `CandidateRulePromotion` and `ResidualPromotionWorkflow`.
  - Generate declarative World Specification Language (WSL) templates and executable deterministic Python rule functions from discovered equations.
  - Audit Gate: Rule promotion requires explicit human sign-off (`is_approved=True`, reviewer ID, review notes). Unapproved promotion attempts raise `DynamicsError`.
  - Self-Whitening Verification: Promoting an approved candidate into the grey-box's known pipeline collapses the remaining residual for that term to zero, proving empirical mass moved from LEARNED $\to$ KNOWN.
  - Discovered dynamics default strictly to observational evidence level (`EvidenceLevel.PREDICTIVE`).

### Sub-track 4: Concept Documentation & Architectural Governance (Track D / PR 4)
- **AC-072 (Documentation & Governance):**
  - Publish concept document `docs/concepts/grey-box-dynamics.md`.
  - Publish ADR-042 documenting design decisions, KKT projection mechanics, and the self-whitening doctrine.
  - Publish newsfragment `newsfragments/204.feature.md`.

---

## 3. Architecture & Data Flow

```
                      Raw Residual Samples (Y = S_actual - S_known)
                                            │
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │ SINDyRegressor (Pure NumPy STLSQ)         │
                      │ Polynomial candidate basis Theta(X)       │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                      ┌───────────────────────────────────────────┐
                      │ CandidateRulePromotion                    │
                      │ - Parsimony Score & BIC                   │
                      │ - WSL Rule Template                       │
                      │ - Epistemic Level: PREDICTIVE             │
                      └─────────────────────┬─────────────────────┘
                                            │
                                            ▼
                         Formal Human Review Sign-Off?
                                   /          \
                            [No]  /            \  [Yes]
                                 ▼              ▼
                     Promotion Blocked     Promote to DeterministicDynamics
                     (DynamicsError)       Incorporate into CompositeDynamics
                                           Residual on Term Collapses to 0.0
```

---

## 4. Verification & Testing

1. **Bit-for-Bit Identity:** `tests/unit/test_grey_box.py` and `tests/contract/test_grey_box_contract.py` prove zero-residual grey-box state fingerprints match symbolic kernel bit-for-bit across seeds (S00 Invariant 5).
2. **Structural Zero Conservation:** `tests/unit/test_conservation_layer.py` proves 50 adversarial unconstrained deltas project to exact $\sum \Delta_i = 0$ ($10^{-12}$) and reports 0.0% violation rate vs 100% on unconstrained baseline.
3. **Symbolic Regression & Promotion:** `tests/unit/test_symbolic_flywheel.py` proves SINDy recovers exact quadratic terms ($R^2 \ge 0.99$), enforces human approval gates, and demonstrates the self-whitening phenomenon.
