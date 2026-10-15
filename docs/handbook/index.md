# The World Modeling Handbook: Simulation-Based Decision Science

> *"A practitioner's guide to modeling, simulating, and evaluating complex organizational dynamics. Modeled on the scikit-learn User Guide: organized by the question being answered, grounded in mathematical formulation, verified with executable code, and uncompromising on epistemic honesty."*

---

## 1. Why a Handbook?

Most documentation in the machine learning and simulation ecosystem is organized around **software modules** (e.g., `core`, `simulation`, `evaluation`). While convenient for developers, this fails practitioners who approach their systems with a specific decision question:

* *"Will our intervention work, by how much, and how sure are we?"*
* *"Under which future operating conditions will our policy catastrophically break?"*
* *"Is the observed treatment effect identifiable, or are we fooling ourselves with confounded data?"*
* *"When must the engine refuse to rank a policy because the claim is mathematically non-identifiable?"*

The **World Modeling Handbook** is designed to teach the discipline of **auditable, structural decision simulation**. It serves simultaneously as a practitioner textbook, a mathematical reference, and an API guide.

---

## 2. The Method Quartet

Every algorithmic method and workflow in the Handbook adheres to a standardized **four-part structure**:

```
      ┌────────────────────────────────────────────────────────┐
      │                   The Method Quartet                   │
      └────────────────────────────────────────────────────────┘
                                   │
      ┌────────────────────────────┼────────────────────────────┐
      ▼                            ▼                            ▼
1. FORMULATION              2. PRACTICAL USAGE          3. IMPLEMENTATION
   Mathematical theory &       Minimal runnable code       Computational mechanics,
   formal equations.           & API patterns.             vectorization, scaling.
                                   │
                                   ▼
                            4. COMMON PITFALLS
                               Where practitioners fail &
                               epistemic failure modes.
```

1. **Mathematical / Theoretical Formulation:** Grounding the method in peer-reviewed decision science, causal inference, and dynamical systems theory.
2. **Practical Usage & API Pattern:** Minimal, idiomatically correct Python examples using `ewm-engine`.
3. **Implementation & Computational Notes:** Insights into numerical stability, seed isolation, state immutability, and runtime complexity.
4. **Common Pitfalls & Failure Modes:** Warning practitioners against real-world traps, data leakage, and ungrounded extrapolation.

---

## 3. Table of Contents

### Part I: Question-Driven Decision Science
* **[1. Interventions & Uncertainty](interventions-and-uncertainty.md)**  
  *Question:* *"Will this intervention help, by how much, and how sure are we?"*  
  *Covers:* Pearl's $do(A=a)$, potential outcomes, non-parametric BCa bootstrap confidence intervals, finite-sample conformal prediction bands, and anytime-valid E-values.

* **[2. Scenario Discovery & Vulnerability Isolation](scenario-discovery-and-vulnerability.md)**  
  *Question:* *"Under which futures does my decision break?"*  
  *Covers:* Patient Rule Induction Method (PRIM / CART), hyper-box peeling $\alpha$, pasting $\beta$, coverage vs. density trade-offs, and multi-dimensional vulnerability mapping.

* **[3. Causal Identifiability & Structural Graphs](causal-identifiability-and-graphs.md)**  
  *Question:* *"Is this effect even identifiable?"*  
  *Covers:* The Backdoor Criterion, unobserved confounding, d-separation, positivity/overlap diagnostics, and graph surgery.

* **[4. Simulation Calibration & SBI](calibration-and-sbi.md)**  
  *Question:* *"How do we calibrate simulation parameters without overfitting to synthetic worlds?"*  
  *Covers:* Simulation-Based Inference (SBI), Neural Posterior Estimation, TARP coverage tests, and Simulation-Based Calibration (SBC) rank statistics.

---

### Part II: The Epistemic Honesty Track
* **[5. When to Refuse a Claim: The Epistemic Honesty Guide](when-to-refuse-a-claim.md)**  
  *The core of EWM's identity:* How to read an E-value under optional stopping; why uncertainty quantification is mathematically mandatory; when `do()` must be refused due to unobserved confounding; the critical distinction between a model-counterfactual and a physical counterfactual; and out-of-support epistemic abstention ([RWM-U](https://arxiv.org/abs/2504.16680)).

---

### Part III: Conceptual Foundations (No 1:1 Class)
* **[6. Choosing a Robustness Metric](choosing-robustness-metric.md)**  
  *Decision taxonomy:* Minimax Regret vs. Satisficing / Domain Criterion vs. Info-Gap Robustness vs. Active Inference Expected Free Energy. When to use which criterion under deep uncertainty.

* **[7. Common Pitfalls: Leakage, Drift, and Overclaiming](pitfalls-leakage-drift-overclaiming.md)**  
  *Practitioner traps:* Future information leakage in multi-step actors; pseudo-random number generator (PRNG) drift across branches; confusing statistical correlation with structural causality; and the flaw of averages.
