# Model & Scenario Card Template

> **Standard:** Based on **Model Cards for Model Reporting** (Mitchell et al., [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)), **Datasheets for Datasets** (Gebru et al., [arXiv:1803.09010](https://arxiv.org/abs/1803.09010)), **FAIR Data with Croissant** (MLCommons, [arXiv:2403.19546](https://arxiv.org/abs/2403.19546)), and **Population Calibration Reporting** ([arXiv:2411.10109](https://arxiv.org/abs/2411.10109)).

---

## Maturity Banner

!!! warning "Maturity Tier: [STABLE | BETA | EXPERIMENTAL | RESEARCH]"
    **Component Name:** `<ComponentName>`  
    **Version:** `<version>`  
    **Epistemic Evidence Level:** `[STRUCTURAL | INTERVENTIONAL | QUASI_CAUSAL | PREDICTIVE | ASSUMED]`  
    **Notice:** `<Summary of experimental boundaries, stability guarantees, or production readiness caveats.>`

---

## 1. Overview & Intended Use

### Model / Scenario Details
- **Identifier:** `<unique_component_id>`
- **Architecture / Type:** `<e.g., Relational GNN, Multi-Layer Perceptron, Receding-Horizon Controller, SCM>`
- **Framework & Dependencies:** `<e.g., PyTorch 2.x, OR-Tools 9.x, SciPy 1.x, or Core-Only>`
- **License:** Apache-2.0
- **Governing ADR:** `[ADR-XXX: Title](../adr/ADR-XXX.md)`
- **Contact / Maintainer:** `<team/author>`

### Primary Intended Uses
1. `<Intended application 1>`
2. `<Intended application 2>`

### Explicitly Out-of-Scope & Forbidden Uses
- **Prohibited Deployment:** `<e.g., Unmonitored automated execution in live production systems without human-in-the-loop review.>`
- **Epistemic Overclaim:** `<e.g., Treating predictive associations P(Y|X) as interventional truth P(Y|do(X)) without backdoor identifiability proof.>`
- **Invariant Bypassing:** `<e.g., Disabling hard constraint gates or boundary clamping.>`

---

## 2. Dataset & Training Specification (Datasheet)

### Data Acquisition & Generation
- **Source:** `<e.g., Historical enterprise ERP logs, synthetic trajectory generator, or scientific benchmark>`
- **Collection Mechanism:** `<e.g., collect_transition_dataset() from canonical simulation world>`
- **Feature Representations:** `<e.g., Symlog-scaled continuous resource levels, one-hot entity types, multi-relational adjacency>`
- **Target Variables:** `<e.g., One-step state delta \Delta S_t = S_{t+1} - S_t>`

### Croissant FAIR Metadata (MLCommons / arXiv:2403.19546)
Where a simulation rollout emits a trace or transition dataset, it is described via Croissant-compliant JSON-LD metadata for reproducibility:

```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/"
  },
  "@type": "cr:Dataset",
  "name": "<dataset_name>",
  "description": "<dataset_description>",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "version": "1.0.0",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "name": "transitions.parquet",
      "contentUrl": "data/transitions.parquet",
      "encodingFormat": "application/vnd.apache.parquet",
      "sha256": "<canonical_sha256_hash>"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "name": "state_transitions",
      "field": [
        {"@type": "cr:Field", "name": "step", "dataType": "sc:Integer"},
        {"@type": "cr:Field", "name": "state_vector", "dataType": "sc:Float"},
        {"@type": "cr:Field", "name": "action_type", "dataType": "sc:Text"},
        {"@type": "cr:Field", "name": "next_state_vector", "dataType": "sc:Float"}
      ]
    }
  ]
}
```

---

## 3. Evaluation & Performance Metrics

### Predictive Metrics
| Metric | In-Distribution | Interventional Shift | OOD Stress | Target Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **MAE** | `<value>` | `<value>` | `<value>` | `<threshold>` |
| **RMSE** | `<value>` | `<value>` | `<value>` | `<threshold>` |
| **CRPS** | `<value>` | `<value>` | `<value>` | `<threshold>` |
| **Coverage (90% Interval)** | `<value>` | `<value>` | `<value>` | `0.90 +/- 0.05` |

### Systemic Invariant Gates (Non-Negotiable)
- **Bounds Preserved:** `<True/False>` (Hard non-negativity and capacity constraints).
- **Conservation Balance Error:** `<value>` (Mass/financial balance delta $\sum \Delta R = 0$).
- **Overall Invariant Validity:** `<PASS / FAIL>` (Any violation invalidates the model).

---

## 4. Population & Stochastic Calibration (arXiv:2411.10109)

> **Calibration Provenance Principle:** Any dynamic component or policy actor representing aggregate population or stochastic group behavior must document its **calibration score** as a first-class provenance artifact.

- **Target Population / Distribution:** `<e.g., Regional consumer demand, hospital arrivals, logistics driver fleet>`
- **Calibration Protocol:** `<e.g., Continuous Ranked Probability Score (CRPS), Quantile Coverage, Kolmogorov-Smirnov distance>`
- **Expected Calibration Error (ECE):** `<value>`
- **Empirical Quantile Spread:**
  - $p_{10}$ observed coverage: `<value>` (nominal 10%)
  - $p_{50}$ median bias: `<value>`
  - $p_{90}$ observed coverage: `<value>` (nominal 90%)
- **Recalibration Frequency:** `<e.g., Every N steps or upon OOD regime-shift alert>`

---

## 5. Causal Semantics & Epistemic Boundaries

- **Causal Level:** `<e.g., CausalLevel.INTERVENTIONAL (via Backdoor Criterion) | CausalLevel.ASSOCIATION (Predictive)>`
- **Assumptions Required for Identification:**
  - `<e.g., Positivity overlap index > 0.80>`
  - `<e.g., Unconfoundedness conditional on {covariates}>`
  - `<e.g., Rosenbaum hidden confounding sensitivity Gamma_crit > 1.8>`
- **What is NOT Claimed:**
  - *“Simulating an intervention under this model does NOT guarantee real-world causal identification if unmodeled confounders exist in the operational environment.”*

---

## 6. Known Limitations & Failure Modes

1. **Long-Horizon Autoregressive Compounding:** `<Description of error drift beyond H steps>`
2. **Support Boundary Extrapolation:** `<Description of behavior when state leaves empirical training support>`
3. **Correlation Breakdown:** `<Scenarios where historical covariance assumptions collapse>`
