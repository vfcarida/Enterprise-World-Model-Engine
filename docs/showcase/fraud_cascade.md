# Flagship Fraud Cascade — Full 6-Part ADR-053 State-Model Showcase

**Architecture Reference**: [ADR-053](../adr/ADR-053-formal-enterprise-world-model-semantics-and-context.md), [Flagship Scenario Reference](../reference/fraud-cascade-scenario.md)  
**Executable Modules**: [`examples/fraud_cascade/main.py`](../../examples/fraud_cascade/main.py), [`examples/fraud_cascade/adr053_full_cascade.py`](../../examples/fraud_cascade/adr053_full_cascade.py)  

---

## 1. The 6-Part ADR-053 State Tuple $(S_t, O_t, B_t, P_t, K_t, A_t)$

This showcase integrates the flagship **Fraud $\to$ Declines $\to$ Call-Center Cascade** directly with all six formal state components specified in **ADR-053**:

| Component | Symbol | Implementation Class | Role in Fraud Cascade |
| :--- | :---: | :--- | :--- |
| **1. Ontic State** | $S_t$ | `WorldState` (`ontic_state_s_t`) | Ground-truth approved/declined cohorts, call-center queue backlog, net cash, and latent fraud rate |
| **2. Partial Observation** | $O_t$ | `ObservationFilter.observe()` (`observation_o_t`) | Planner view containing visible queues and declined totals while masking `latent_true_fraud_attempts` |
| **3. Belief State** | $B_t$ | `BeliefState` (`belief_state_b_t`) | Planner's point estimate and epistemic uncertainty over unobserved fraud arrival intensity |
| **4. Epistemic State** | $P_t$ | `EpistemicState` (`epistemic_state_p_t`) | Decoupled provenance, `EvidenceLevel`, and `DriftSignal(POLICY_INDUCED_SHIFT)` when $\text{do}(\tau=0.40)$ saturates queues |
| **5. Runtime Context** | $K_t$ | `ContextSnapshot` (`context_k_t`) | Versioned traffic regime (`nominal` $\to$ `holiday_traffic_surge`), staffing capacity, and SLA penalty terms |
| **6. Action / Intervention** | $A_t$ | `Action` (`action_a_t`) | Fraud-blocking score threshold intervention $\text{do}(\tau \in \{0.40, 0.55\})$ |

---

## 2. Running the Showcase

```bash
python examples/fraud_cascade/main.py
```

The showcase verifies that:
- Cohort mass conservation ($\text{Submitted} = \text{Approved} + \text{Declined}$) holds within IEEE-754 double precision (`0.000e+00` drift),
- `latent_true_fraud_attempts` is strictly excluded from $O_t$ while tracked with growing epistemic uncertainty in $B_t$, and
- Balanced EWM planning ($\text{do}(\tau = 0.55)$) eliminates the `$148,932.20` policy-induced cascade regret incurred by myopic fraud thresholding ($\text{do}(\tau = 0.40)$).
