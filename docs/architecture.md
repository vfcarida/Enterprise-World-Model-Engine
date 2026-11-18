# Realigned Core & Domain Architecture: OCEL 2.0, CDA Dynamics, and DoWhy-GCM Causality

The **Enterprise World Model Engine (EWM Engine)** strictly separates its domain-neutral **Core Engine** (`ewm_engine.core`) from domain-specific **Application Packs** (`ewm_engine.domains.finance` and `ewm_engine.domains.software_engineering`). Historical documentation prior to the E1–E10 realignment is archived under [`docs/legacy/`](legacy/architecture_legacy.md).

---

## 1. Core Engine vs. Domain Packs Delineation

| Architectural Layer | Module Path | Responsibility & Mathematical Contract |
| :--- | :--- | :--- |
| **Core State (OCEL 2.0)** | `ewm_engine.core.ocel` | Relational SQLite Object-Centric Event Log (`StateEncoder`, $N:M$ `event_object` E2O and `object_object` O2O mappings) with PM4Py JSON/XML/CSV/XES interoperability. |
| **Core Dynamics (CDA)** | `ewm_engine.core.dynamics` | Continual Discovery Agent (`RetrieveAndReasonEngine`) modeling $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t, c)$ with explicit `RuleAttribution` to retrieved `BusinessRule` configs $c$. |
| **Core Causality (DoWhy-GCM)** | `ewm_engine.core.causal` | Graphical Structural Causal Models (`EngineCausalInterface`) enforcing strict DAG acyclicity and modular graph surgery $do(V_s = v_s)$. |
| **Finance Domain Pack (E6)** | `ewm_engine.domains.finance` | Arbitrary-precision `Decimal` double-entry ledgers (`ImmutableLedger`, `ImmutableTransaction`, `Entry`) enforcing $dr - cr = 0$ and $h_a = h_o$. |
| **Software Engineering Pack (E7)** | `ewm_engine.domains.software_engineering` | Sparse eIRWR + DynaCausal microservice incident cascade simulator (`DynamicRCASimulator`) with anomaly-conditioned resilience $R_i = R_{\text{base}} \exp(-\beta \hat{s}_i)$ and backward retry edges ($\rho$). |

---

## 2. Hybrid Transition Router Pipeline (`Pre-rules -> ML -> Post-rules`)

```mermaid
flowchart LR
    accTitle: Hybrid Transition Router Three-Stage Pipeline
    accDescr: Flowchart showing Stage 1 Pre-Rules action validation and do() graph surgery, Stage 2 learned ML residual inference with NaN/Inf safeguard, and Stage 3 Post-Rules constraint and bounds verification.

    ST["Immutable State S_t + Action A_t"] --> PRE["Stage 1: Pre-Rules\n(Action Validation & do(X_t=x) Graph Surgery)"]
    PRE -->|"Invalid Action"| REJ_PRE["Reject Action\n(Preserve Invariants)"]
    PRE -->|"Accepted Action + Severed Inbound Edges"| STRUCT["Structural Baseline\nS_{t+1}^{struct} = f_struct(PA_Y(t), A_t, U_t)"]
    STRUCT -->|"No ML Configured"| DET["Deterministic Outcome\n(is_deterministic = True)"]
    STRUCT -->|"ML Predictor Active"| ML["Stage 2: Learned ML Inference\n(Predict Residual Mean & Std)"]
    ML --> NAN_GATE{"Operational Safeguard:\nNaN / Inf Check"}
    NAN_GATE -->|"NaN / Inf Detected"| ROLLBACK_NAN["Catch Before Post-Rules\n(Rollback to S_{t+1}^{struct})"]
    NAN_GATE -->|"Finite Proposal"| POST["Stage 3: Post-Rules\n(Resource Bounds, Constraints & Hallucination Gate)"]
    POST -->|"Out-of-Bounds or Invalid Action"| ROLLBACK_POST["100% Reject & Rollback\nto S_{t+1}^{struct}"]
    POST -->|"All Post-Rules Satisfied"| CAL["Calibrated Distribution Outcome\n(CalibratedStateDistribution at t+1)"]
```

---

## 2. Pearlian `do(X_t = x)` Causal Surgery & Confounding Removal

For a structural causal system $Y_{t+1} = f(X_t, Z_t, U_t)$ where an observational confounder $Z_t$ drives both treatment $X_t$ and outcome $Y_{t+1}$:

```mermaid
flowchart TD
    accTitle: Pearlian do() Causal Graph Surgery
    accDescr: Comparison of observational graph G with confounding edges versus mutilated interventional graph G_do(X_t=x) where inbound edges to X_t are severed while causal edge X_t to Y_{t+1} is preserved.

    subgraph Observational["Observational Graph G"]
        Z1["Confounder Z_t"] -->|"Confounding Edge"| X1["Treatment X_t"]
        UX1["Exogenous Noise U_X"] -->|"Inbound Edge"| X1
        Z1 --> Y1["Outcome Y_{t+1}"]
        X1 -->|"Causal Edge"| Y1
        W1["Independent Node W_t"]
    end

    subgraph Interventional["Mutilated Graph G_{do(X_t = x)} via engine.branch().do(action)"]
        Z2["Confounder Z_t"] -.->|"SEVERED (Z_t -> X_t)"| X2["Forced X_t := x"]
        UX2["Exogenous Noise U_X"] -.->|"SEVERED (U_X -> X_t)"| X2
        Z2 --> Y2["Outcome Y_{t+1}"]
        X2 -->|"Causal Edge Preserved"| Y2
        W2["Independent Node W_t\n(Negative Control: No Path to Y_{t+1})"]
    end
```

### Key Guarantees (`EWM-002`)
1. **Deterministic vs. Calibrated Outcome (`TransitionEngine.step(state, action)`)**:
   - Returns a deterministic `TransitionStepOutcome` (`is_deterministic=True`, `distribution=None`) when only structural rules apply.
   - Returns a calibrated distribution (`is_deterministic=False`, `distribution=CalibratedStateDistribution`) when an ML module is active.
2. **Pre-Rule Structural Gate**: Rejects invalid enterprise actions before any state transition or learned inference executes.
3. **Operational `NaN`/`Inf` Safeguard**: Intercepts numerical `NaN` or `Inf` emitted by learned modules *before* structural `Post-rules` evaluation (`NumericalInstabilityError`).
4. **Post-Rule 100% Rollback**: Rejects 100% of out-of-bounds ML predictions and hallucinated invalid actions, rolling back to the valid deterministic structural state $S_{t+1}^{\text{struct}}$.
5. **Context-Managed `Intervention` & `engine.branch().do(action)`**: Forces specific state nodes, overrides standard and learned dynamics, and severs all inbound edges $(u \to X_t)$ in the causal graph (Peters et al., 2017; Pearl, 2009).
