# ADR-019: Learned Dynamics Evaluation Harness, Invariant Verification, and Experimental Neural Baseline

## Status
Accepted (v1.2.0-experimental)

## Context
EWM Engine provides the `LearnedDynamics` protocol and a reference `LinearResidualDynamics` baseline (§39 of the system specification). However, there has been no formal, reproducible mechanism to quantitatively evaluate candidate learned dynamics models before deployment.

Recent machine learning and physics-AI research (Physics-IQ, arXiv:2501.09038; Sora physical law critique, arXiv:2411.02385) has empirically demonstrated a critical finding: **surface predictive accuracy (low MSE/MAE) does not imply systemic understanding or adherence to physical laws**. A neural network or regression model can achieve low one-step numeric error while consistently violating structural invariants—such as generating negative inventory, creating matter out of nothing, exceeding transport link throughput, or destabilizing multi-step rollouts through compounding drift.

Furthermore, EWM Engine enforces a zero-dependency core constraint (ADR-004): deep learning frameworks (PyTorch, TensorFlow, JAX) must never be required by the simulation engine or core library surface.

## Decision

1. **Systemic-Invariant Gate (Mandatory Consistency Gate)**:
   - Any evaluated `DynamicsModel` must pass declared organizational invariants (resource non-negativity, capacity bounds, mass conservation $\sum \Delta r = 0$, and custom business constraints).
   - Invariant violations are reported as first-class failures, completely separate from numeric predictive error.
   - A model with low numeric error but non-zero invariant violations is flagged as `UNFIT` for enterprise world simulation.
2. **Unified Evaluation Harness (`ewm_engine.experimental.dynamics_eval`)**:
   - Implements standardized, reproducible metrics for *any* `DynamicsModel`:
     - **One-Step Error**: Per-resource and aggregate MAE, RMSE, MAPE, and DreamerV3-style Symlog Error ($\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$) to normalize scale differences across heterogeneous metrics.
     - **Multi-Step Rollout Divergence**: Evaluates autoregressive compounding error against a ground-truth reference over horizon $H$.
     - **Stochastic Calibration**: Continuous Ranked Probability Score (CRPS) and quantile coverage for probabilistic dynamics.
     - **Interventional Shift Detection**: Evaluates model performance under out-of-distribution interventional action distributions.
3. **`TransitionDataset` Ergonomics**:
   - Provides utilities (`collect_transition_dataset`, `.split()`, `.to_numpy()`, `.batch()`) to generate training and benchmark sets directly from world simulations without requiring external dependencies.
4. **Experimental Neural Baseline (`ewm_engine.experimental.dynamics_torch`)**:
   - A PyTorch MLP state-space residual model (`TorchNeuralResidualDynamics`) implementing `LearnedDynamics`.
   - Clamps predictions to declared resource `[min_value, max_value]` bounds.
   - Carries an explicit evidence tier of `EvidenceLevel.PREDICTIVE`.
   - Quarantined behind the optional `ml` extra with strictly lazy imports: importing `ewm_engine` or `ewm_engine.experimental` never imports PyTorch into `sys.modules`.
5. **Model Card & Transparency**:
   - Includes a published model card detailing synthetic training setup, intended benchmarking scope, known failure modes (interventional hallucination, compounding rollout drift), and an explicit "Experimental, not for decision-making" banner.

## Consequences

### Positive
- Formalizes a rigorous benchmark protocol to measure predictive accuracy, multi-step stability, and invariant preservation.
- Proves that the `LearnedDynamics` protocol supports both linear OLS and deep neural backends.
- Prevents models with good numeric fit but illegal physical behaviors from passing validation.
- Preserves the zero-dependency core and strict import boundaries.

### Negative / Trade-offs
- PyTorch remains an optional dependency (`pip install "ewm-engine[ml]"`).
- Neural models cannot guarantee exact mass conservation without explicit post-hoc projection or specialized architectural constraints.

## Compliance Verification
- Automated integration tests (`tests/integration/test_learned_neural_dynamics.py`) verify the harness detects interventional distribution shifts.
- Invariant tests verify that models generating invalid states are flagged with `invariant_pass = False`.
- Architecture boundary tests (`tests/architecture/test_import_boundaries.py`) verify zero top-level PyTorch imports in core modules.
