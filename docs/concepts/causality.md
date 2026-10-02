# Causal Epistemology and Scientific Integrity

One of the defining commitments of the **Enterprise World Model Engine (EWM Engine)** is **epistemic honesty**: the refusal to label predictive or observational associations as proven causal relationships.

---

## $P(Y \mid X)$ vs. $P(Y \mid \text{do}(X))$

A foundational error in applied machine learning is equating conditional observational forecasting with interventional counterfactual outcomes:

$$P(Y \mid X) \;\neq\; P(Y \mid \text{do}(X))$$

### The Observational Trap
Observing that an organization experiences low inventory whenever discounts are offered ($P(\text{Stock} \mid \text{Discount})$) does **not** prove that unilaterally changing discount rates will causally reproduce that historical stock trajectory. Unobserved confounding (e.g., seasonal holidays, competitor bankruptcies, or regional marketing campaigns) can drive both variables simultaneously.

### The Interventional Exploration
EWM Engine enables developers to simulate counterfactual branches:
$$\text{Simulate under } \text{do}(\text{Intervention}_A) \quad\text{vs}\quad \text{do}(\text{Intervention}_B)$$
However, **simulating an intervention within a model does not magically guarantee that the simulation is causally identified in reality**. The validity of the simulated future depends entirely on the epistemic basis of the transition models and assumptions.

---

## The Evidence Level Hierarchy

Every transition model, dynamic component, and systemic trace edge in EWM Engine is explicitly tagged with an `EvidenceLevel`:

| Evidence Level | Status | Epistemic Basis |
|---|---|---|
| `STRUCTURAL` | Proven Invariant | Grounded in deterministic physical conservation, legal accounting invariants, or mathematical rules (e.g., $S_{t+1} = S_t - \text{outflow} + \text{inflow}$). |
| `INTERVENTIONAL` | Empirically Identified | Validated through randomized controlled trials (A/B testing) or formal Judea Pearl *do-calculus* identification. |
| `QUASI_CAUSAL` | Econometric Estimate | Estimated via quasi-experimental techniques (e.g., difference-in-differences, instrumental variables, synthetic control). |
| `PREDICTIVE` | Observational Association | Learned through statistical regression, neural networks, or supervised forecasting without causal identification ($P(Y \mid X)$). |
| `ASSUMED` | Heuristic / Working Hypothesis | Operational assumption or behavioral heuristic not yet empirically verified. |

---

## Scientific Humility in Enterprise Modeling

EWM Engine models the interactions of rules, dynamics, and shocks. It provides:
1. **Scenario Dependency Tracking**: Showing the step-by-step propagation of assumptions.
2. **Epistemic Trace Aggregation**: When composing models, the weakest evidence tier determines the overall evidence status of the composite outcome.
3. **No Overclaiming**: The engine produces simulated counterfactual rollouts to evaluate potential decisions, but never claims ungrounded causal omnipotence.
