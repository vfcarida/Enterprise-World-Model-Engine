# Concept Guide: Common Pitfalls — Leakage, Drift, and Overclaiming

> *"A world model can fail in subtle, mathematically devastating ways that pass traditional unit tests. This guide identifies the four most pervasive pitfalls in simulation-based decision engineering and explains how EWM prevents them architecturally."*

---

## 1. Pitfall 1: Information Leakage in Multi-Step Simulators

### The Pathology
In a dynamic organizational simulation, decision policies (whether heuristic rules or reinforcement learning agents) must act based **only on the information available at step $t$**. 

Information leakage occurs when:
* An actor's decision logic reads future exogenous shock arrays before they are sampled in time.
* Summary statistics computed across the entire horizon (e.g., standardizing inventory levels using the global trajectory mean) are leaked into intermediate transition functions.

When leakage occurs, a policy appears brilliant in backtests because it "anticipates" shocks that were impossible to foresee in real time.

### How EWM Prevents It
* **Causal Event Buffers:** Exogenous shocks are sampled on demand at step $t$ using the step's isolated PRNG stream.
* **Strict State Immutability:** Step transitions produce a new, immutable [`WorldState`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/state.py) instance. Actors receive only read-only views of past and present states; they have no access to the future simulation schedule or downstream random draws.

---

## 2. Pitfall 2: Pseudo-Random Number Generator (PRNG) Drift

### The Pathology
Many simulation frameworks rely on global random seeds (e.g., `np.random.seed(42)`). When evaluating counterfactual scenario branches:
* Scenario A (Status Quo) evaluates a normal schedule, consuming 50 random draws.
* Scenario B (Intervention) introduces an additional transfer action that consumes 2 extra random draws.
* **The Disaster:** Downstream random draws for customer orders in Scenario B are now shifted by 2 positions relative to Scenario A. The two scenarios are no longer experiencing the same world! What appears to be an effect of the policy is actually an artifact of **PRNG phase drift**.

```
  Global Seed Failure (Drift):
  Scenario A:  [Shock 1] [Shock 2] [Shock 3] [Shock 4]
  Scenario B:  [Extra Draw 1] [Extra Draw 2] [Shock 1 shifted] [Shock 2 shifted] -> UNCOUPLED!
```

### How EWM Prevents It
* **Hierarchical `SeedSequence` Trees:** EWM utilizes NumPy's 64-bit `SeedSequence` protocol ([ADR-012](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-012-determinism-reproducibility-and-branch-isolation.md)).
* **Independent Substream Spawning:** Each trajectory sample $k$ and each subcomponent (actors, exogenous events, stochastic dynamics) spawns a dedicated, mathematically independent child PRNG stream. Adding or removing actions in one branch cannot shift the random sequence of another branch.

---

## 3. Pitfall 3: Observational Overclaiming (The Lucas Critique)

### The Pathology
The **Lucas Critique** (1976) states that it is naive to predict the effects of a change in economic policy entirely on the basis of relationships observed in historical data.

* If you observe that "promotional discounts are correlated with 15% revenue growth," that relationship was formed under the historical pricing policy.
* If management permanently lowers baseline prices by 20%, competitors adapt, consumers shift reference points, and the historical regression equation collapses.

Supervised machine learning optimizes conditional associations $P(Y \mid X)$. When an organization intervenes, it changes the data-generating process, invalidating historical correlations.

### How EWM Prevents It
* **First-Class Structural Invariants:** Instead of modeling entire systems as monolithic black-box regressions, EWM anchors simulation on **KNOWN structural mechanisms**: physical transport corridors, accounting conservation rules, and legal constraints.
* **Pearl SCM Graph Surgery:** Interventions are evaluated as formal graph surgeries ($do(A=a)$), severing historical policy arrows while preserving immutable structural relations ([ADR-040](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/docs/adr/ADR-040-twin-rollout-counterfactual-scm-kernel.md)).

---

## 4. Pitfall 4: The Flaw of Averages

### The Pathology
The "Flaw of Averages" (Savage, 2000) states that plans based on average assumptions will, on average, fail.

Consider a logistics depot:
* Expected daily inbound freight: 100 units.
* Processing capacity: 100 units/day.
* A spreadsheet using average numbers forecasts 0 backlog.
* In reality, if Day 1 has 150 units and Day 2 has 50 units, the depot overflows on Day 1 (incurring overflow fees and delayed orders) and stands idle on Day 2. The average backlog is strictly positive!

In any non-linear, constrained system:

$$\mathbb{E}[f(X)] \neq f(\mathbb{E}[X])$$

Evaluating a simulation on a single "average run" is mathematically fraudulent.

### How EWM Prevents It
* **Mandatory Ensemble Sampling:** EWM runs full Monte Carlo ensembles (typically 500–1,000 runs) and aggregates results across trajectories.
* **Non-Point Forecast Outputs:** EWM’s [`ScenarioForecastResult`](https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/src/ewm_engine/core/trust.py) computes full percentile distributions ($p05, p50, p95$) and dispersion bounds, making the true distribution visible to leadership.

---

## Summary: Epistemic Safeguard Matrix

| Pitfall | Root Cause | Failure Symptom | EWM Architectural Defense |
| :--- | :--- | :--- | :--- |
| **Information Leakage** | Policies observing future trajectory state | Artificially inflated backtest returns | Causal event buffers + state immutability |
| **PRNG Drift** | Shared global random state across branches | Accidental noise divergence mistaken for treatment effect | Hierarchical NumPy `SeedSequence` trees |
| **Observational Overclaiming** | Regression curves treated as causal policies | Policy collapse under structural regime shifts | Structural Causal Model (SCM) graph surgery |
| **Flaw of Averages** | Modeling expected values instead of full distributions | Unforeseen capacity bottlenecks & covenant breaches | Mandatory Monte Carlo ensembles & $p05/p95$ bounds |
