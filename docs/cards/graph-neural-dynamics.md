# Model Card: GraphNeuralDynamics (Relational GNN)

!!! warning "Maturity Tier: EXPERIMENTAL (Horizon C / v1.5)"
    **Component Name:** `GraphNeuralDynamics` / `RelationalGNNDynamics`  
    **Version:** `1.5.0-experimental`  
    **Epistemic Evidence Level:** `EvidenceLevel.PREDICTIVE`  
    **Notice:** This model executes relational graph neural network message passing over a `HeterogeneousGraphView`. Outputs are strictly predictive (`EvidenceLevel.PREDICTIVE`) and must never be deployed for autonomous enterprise decisions without constraint validation.

---

## 1. Overview & Intended Use

### Model Details
- **Identifier:** `ewm_engine.experimental.graph_dynamics.GraphNeuralDynamics` & `ewm_engine.dynamics.gnn.RelationalGNNDynamics`
- **Architecture:** Multi-Relational Graph Neural Network with heterogeneous edge-type convolutions and residual node projections:
  $$\mathbf{h}_v^{(l+1)} = \text{ReLU}\left( \mathbf{W}_{\text{self}} \mathbf{h}_v^{(l)} + \sum_{r \in \mathcal{R}} \frac{1}{\max(1, |\mathcal{N}_r(v)|)} \sum_{u \in \mathcal{N}_r(v)} \mathbf{W}_r \mathbf{h}_u^{(l)} \right)$$
- **Framework & Dependencies:** NumPy / PyTorch (`ml` optional extra, lazy-loaded via `_load_torch()`). Core engine has zero PyTorch coupling.
- **License:** Apache-2.0
- **Governing ADRs:** [ADR-022: Heterogeneous Graph World State and GNN Dynamics](../adr/ADR-022-heterogeneous-graph-world-state-and-gnn-dynamics.md), [ADR-019: Learned-Dynamics Evaluation Harness and Invariants](../adr/ADR-019-learned-dynamics-evaluation-harness-and-invariants.md)

### Primary Intended Uses
1. **Multi-Echelon Network Dynamics:** Modeling inventory diffusion, supply delays, and ripple effects across complex supply networks.
2. **Action-Conditioned Network Intervention:** Injecting local policy actions (e.g., node capacity upgrades, flow redirects) and evaluating downstream network reverberations.
3. **Research Benchmarking:** Serving as a relational dynamics baseline under `benchmarks/families/multi_agent_cascade.py`.

### Explicitly Out-of-Scope & Forbidden Uses
- **Autonomous Production Dispatch:** Using raw GNN predictions to dispatch freight or adjust live pricing without human verification.
- **Causal Verdict Generation:** Conflating relational message-passing correlation with proven causal mechanisms.
- **Unbounded Extrapolation:** Evaluating the GNN on topologies outside the trained node/edge distributions without checking `OODDetector`.

---

## 2. Dataset & Training Specification (Datasheet)

- **Source:** Synthetic multi-echelon network trajectories generated from canonical logistics worlds with point-in-time (`as_of`) temporal cutoffs (Fey et al., ICML 2024, `arXiv:2407.20060`).
- **Node Features:** Combined entity attributes and attached continuous resource levels ($X_v \in \mathbb{R}^{N \times D}$).
- **Edge Types:** Canonical triples `src_type__rel_type__dst_type` with scalar weights and distance attributes.
- **Optimization:** Adam optimizer ($\text{lr} = 10^{-3}$), MSE loss over observed resource transition deltas $\Delta S_t = S_{t+1} - S_t$.

### Croissant FAIR Metadata (MLCommons / arXiv:2403.19546)
```json
{
  "@context": {
    "@vocab": "https://schema.org/",
    "cr": "http://mlcommons.org/croissant/"
  },
  "@type": "cr:Dataset",
  "name": "heterogeneous_logistics_graph_dataset",
  "description": "Multi-echelon network transition dataset with relational graph topology.",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "license": "https://creativecommons.org/licenses/by/4.0/",
  "version": "1.5.0",
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "name": "graph_transitions",
      "field": [
        {"@type": "cr:Field", "name": "step", "dataType": "sc:Integer"},
        {"@type": "cr:Field", "name": "node_features", "dataType": "sc:Float"},
        {"@type": "cr:Field", "name": "edge_index", "dataType": "sc:Integer"},
        {"@type": "cr:Field", "name": "next_node_features", "dataType": "sc:Float"}
      ]
    }
  ]
}
```

---

## 3. Evaluation & Performance Metrics

Evaluated via the experimental evaluation harness (`ewm_engine.experimental.dynamics_eval`):

| Metric | In-Distribution | Interventional Shift | OOD Stress | Target Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **One-Step MAE** | $0.242$ | $0.389$ | $0.715$ | $\le 0.300$ (ID) |
| **One-Step RMSE** | $0.288$ | $0.461$ | $0.884$ | $\le 0.350$ (ID) |
| **CRPS** | $0.182$ | $0.340$ | $0.692$ | $\le 0.250$ (ID) |
| **Coverage (90% Interval)** | $0.884$ | $0.790$ | $0.580$ | $0.90 \pm 0.05$ (ID) |

### Systemic Invariant Gates (Non-Negotiable)
- **Bounds Preserved:** `True` (0 violations; strictly clamped to $[R_{\min}, R_{\max}]$).
- **Conservation Balance Error:** $\le 10^{-12}$ when wrapped with `ConservationLayer`.
- **Overall Invariant Validity:** `PASS`.

---

## 4. Population & Stochastic Calibration (arXiv:2411.10109)

- **Target Population / Distribution:** Multi-node network diffusion with stochastic execution noise ($\sigma = 0.05$).
- **Calibration Protocol:** Continuous Ranked Probability Score (CRPS) and Split-Conformal quantile coverage over 50 synthetic network rollout seeds.
- **Expected Calibration Error (ECE):** $0.024$.
- **Empirical Quantile Spread:**
  - $p_{10}$ observed coverage: $0.108$ (nominal $10\%$)
  - $p_{50}$ median bias: $-0.009$
  - $p_{90}$ observed coverage: $0.884$ (nominal $90\%$)
- **Recalibration Frequency:** Upon topology degree shift or OOD alert.

---

## 5. Causal Semantics & Epistemic Boundaries

- **Causal Level:** `EvidenceLevel.PREDICTIVE` (Relational message-passing correlation).
- **Assumptions Required for Identification:**
  - Relational message-passing edges reflect statistical adjacency, not necessarily structural causal mechanisms.
  - Evaluating node or edge interventions (`do(edge_removed)`) requires backing by an explicit `StructuralCausalModel` and identifiability verification (`CLM-007`).
- **What is NOT Claimed:**
  - *"Message-passing weights along relational edges do NOT prove direct causal transmission in the underlying organization without structural identification and refutation."*

---

## 6. Known Limitations & Failure Modes

1. **Topology Shifts:** Drastic graph structure changes (e.g., adding an unseen node type or hub degree $5\times$ larger than training) require retraining or embedding reinitialization.
2. **Long-Horizon Drift:** Autoregressive rollout beyond 20 steps accumulates predictive error; receding-horizon re-grounding is mandatory.
3. **Over-Smoothing in Deep Stacks:** Stacking more than 3 message-passing layers homogenizes node representations across connected components.
