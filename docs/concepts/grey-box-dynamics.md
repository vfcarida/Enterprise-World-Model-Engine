# Grey-Box Dynamics & The Self-Whitening Neuro-Symbolic Flywheel

!!! warning "Maturity Level: Experimental / Research"
    Grey-box dynamics, architectural conservation layers, and symbolic regression flywheels reside in `ewm_engine.experimental`.
    Core engine execution remains pure Python/NumPy, deterministic, and dependency-light.

---

## 1. Lineage & The Grey-Box Thesis

The central thesis of the Enterprise World Model Engine is that **forcing known rules into opaque neural weights produces hallucinated states, whereas pure symbolic models fail to capture unmodeled real-world friction**.

The scientific literature over 2020–2026 resolves this dichotomy through **hybrid known + learned dynamics**:

- **Universal Differential Equations (UDE; Rackauckas et al., arXiv:2001.04385):**
  Instead of learning the entire system transition from scratch, the system is modeled as a known symbolic kernel plus an embedded learnable residual:
  $$\dot{u} = f_{\text{known}}(u, t) + g_\theta(u, t)$$
- **KKT-hPINN (Hard Linear Equality Constraints; arXiv:2402.07251):**
  Enforcing conservation laws via soft penalties in loss functions or post-hoc rejection fails under distribution shifts. KKT nullspace projection layers enforce exact linear conservation *by construction* directly in the architecture.
- **Stable Port-Hamiltonian Neural Networks (arXiv:2502.02480; HNN 1906.01563, LNN 2003.04630):**
  Skew-symmetric inter-node routing combined with positive semi-definite dissipation guarantees Lyapunov stability and prevents energy/inventory explosions.
- **SINDy & PySR (Brunton et al., arXiv:1509.03580; Cranmer, arXiv:2305.01582):**
  Sparse regression extracts human-interpretable, closed-form equations from empirical time-series data.
- **LLM-SR (ICLR 2025 oral, arXiv:2404.18400):**
  Uses equation-as-program hypothesis priors as advisory proposers, leaving numerical fitting and scientific verification to deterministic solvers.

EWM Engine unites these advances into an enterprise-native architecture: **learn only the residual over known rules, enforce conservation architecturally, and symbolically whiten the grey-box over time**.

```
                   ┌────────────────────────────────────────┐
                   │        Current State S_t, Action A_t   │
                   └───────────────────┬────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │   KNOWN Symbolic      │             │   LEARNED Residual    │
        │   Kernel f_known      │             │   Estimator g_theta   │
        └───────────┬───────────┘             └───────────┬───────────┘
                    │                                     │
                    │ Next State (Base)                   │ Unconstrained Delta
                    ▼                                     ▼
             S_{known, t+1}               ┌───────────────────────────────┐
                    │                     │ KKT Nullspace Projection Layer│
                    │                     │ P = I - A^T (A A^T)^{-1} A    │
                    │                     │ (sum Delta = 0, exact bounds) │
                    │                     └───────────────┬───────────────┘
                    │                                     │ Conserved Delta
                    └──────────────────┬──────────────────┘ Delta_{proj}
                                       ▼
                       ┌───────────────────────────────┐
                       │ Final State S_{t+1}           │
                       │ (Violation Rate = 0.0% Strict)│
                       └───────────────┬───────────────┘
                                       │
                                       ▼ (Discrepancy Data)
                       ┌───────────────────────────────┐
                       │ SINDy / PySR Sparse Fit       │
                       │ Closed-Form Residual Discovery│
                       └───────────────┬───────────────┘
                                       │
                                       ▼ (Goodness-of-Fit + Parsimony)
                       ┌───────────────────────────────┐
                       │ Audited Human Promotion Gate  │
                       │ (ADR / Feature Spec Review)   │
                       └───────────────┬───────────────┘
                                       │
                                       ▼ Move Mass: LEARNED -> KNOWN
                       ┌───────────────────────────────┐
                       │ Updated WSL Ruleset (Whitened)│
                       └───────────────────────────────┘
```

---

## 2. Mathematical Foundation

### 2.1 The Discrete UDE Grey-Box Formulation

In discrete-time enterprise world models, transitions decompose into:

$$S_{t+1} = f_{\text{known}}(S_t, A_t, E_t) \oplus \mathcal{P}_{\text{conserved}}\left(\Delta_{\theta}(S_t, A_t, E_t)\right)$$

Where:
- $f_{\text{known}}$ is an explicit `DynamicsModel` (e.g. `DeterministicTransferDynamics` or `CompositeDynamics`).
- $\Delta_\theta$ is a `ResidualEstimator` predicting only unexplained discrepancies (friction, leakage, unmodeled demand drift).
- $\mathcal{P}_{\text{conserved}}$ is the architectural conservation projection layer.

### 2.2 S00 Invariant 5: Zero-Residual Bit-Identity

When the learned residual model predicts zero ($\Delta_\theta \equiv \mathbf{0}$) or is absent:

$$S_{t+1} \equiv f_{\text{known}}(S_t, A_t, E_t)$$

The cryptographic state hash and SHA-256 fingerprint of the resulting world state are **bit-for-bit identical** to the pure symbolic kernel.

---

## 3. Why Architectural Conservation Beats Penalty & Rejection

Enterprise world models govern capital, inventories, and legal obligations. In these domains, constraint violations are unacceptable:

| Approach | Mechanism | Failure Mode |
| :--- | :--- | :--- |
| **Soft Loss Penalty (PINN)** | Adds $\lambda \|A \Delta - b\|^2$ to training loss | Violations remain non-zero during test time; catastrophic drift under distribution shift. |
| **Post-Hoc Rejection** | Discards states that violate constraints | Wastes compute; produces non-differentiable simulation discontinuities; fails if entire batch is invalid. |
| **Architectural Projection (EWM)** | Projects output onto constraint manifold nullspace | **0.0% violation rate by construction**; exact satisfaction even for random or adversarial neural outputs. |

### 3.1 KKT Nullspace Projection

Given linear enterprise conservation laws (such as multi-depot inventory conservation $\sum \Delta r_i = 0$ or double-entry balance $A \Delta r = \mathbf{b}$):

$$\Delta_{\text{conserved}} = \Delta - A^T (A A^T)^{-1} (A \Delta - \mathbf{b}) = P \Delta + A^\dagger \mathbf{b}$$

Where $P = I - A^T (A A^T)^{-1} A$ is the orthogonal projection onto $\ker(A)$.

**Proof of Invariance:**
$$A \Delta_{\text{conserved}} = A (I - A^T (A A^T)^{-1} A)\Delta = (A - A)\Delta = \mathbf{0}$$

For **any** neural vector $\Delta \in \mathbb{R}^d$, $A \Delta_{\text{conserved}} \equiv \mathbf{0}$ within machine precision ($\approx 10^{-15}$).

### 3.2 Port-Hamiltonian Stability

To prevent runaway feedback loops, conservative flows are parameterized via Port-Hamiltonian routing:

$$\Delta = (J - R) \nabla H$$

Where:
- $J = -J^T$ is skew-symmetric (lossless routing of inventory/funds across nodes).
- $R \succeq 0$ is positive semi-definite (frictional dissipation, such as spoilage or fees).
- $\frac{dH}{dt} = (\nabla H)^T (J - R) \nabla H = -(\nabla H)^T R \nabla H \le 0$ guarantees Lyapunov stability.

---

## 4. The Neuro-Symbolic Flywheel: Self-Whitening Dynamics

A grey-box that remains permanently grey is an incomplete solution. EWM Engine implements a **neuro-symbolic flywheel** where the grey-box whitens itself over time:

1. **Empirical Residual Extraction:**
   Observed discrepancies between ground truth transition data and the known ruleset are extracted:
   $$Y_{\text{residual}} = S_{\text{observed}, t+1} - S_{\text{known}, t+1}$$
2. **SINDy Sparse Regression:**
   Sequential Thresholded Least Squares (STLSQ) evaluates a candidate polynomial library $\Theta(X)$ to discover sparse closed-form equations (e.g. $\Delta_{\text{leakage}} = -0.05 \cdot \text{inventory} + 0.001 \cdot \text{orders}$).
3. **Goodness-of-Fit & Parsimony Scoring:**
   Candidate terms are scored by $R^2$, MSE, Bayesian Information Criterion (BIC), and parsimony score $k$ (favoring simpler equations).
4. **Advisory Hypotheses (LLM-SR):**
   External LLM-SR synthesizers can propose equation templates as advisory hypotheses. These remain strictly advisory: never executing in the simulation core without numerical verification.
5. **Audited Human-in-the-Loop Promotion:**
   Promotion into the World Specification Language (WSL) ruleset requires formal human approval linked to an ADR or Feature Spec. Automated core promotion is architecturally prohibited.
6. **Mass Shift (LEARNED $\to$ KNOWN):**
   Once promoted into the known dynamics pipeline, the previously unexplained residual variance collapses to zero.

---

## 5. Epistemic Status & Honesty Guarantees

- **Default Evidence Level:** All dynamically discovered symbolic terms carry `EvidenceLevel.PREDICTIVE` (observational evidence tier). They are never upgraded to `STRUCTURAL` or `COUNTERFACTUAL` without independent physical/accounting validation.
- **Honesty Metrics:** EWM Engine reports the `ConservationHonestyReport`, contrasting the structural-zero violation rate of the grey-box against unconstrained baselines.
