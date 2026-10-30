# Cost-Benefit Protocol for World Modeling Adoption

**Version**: 1.0.0  
**Date**: 2026-10-30  
**Author**: Vinicius Caridá  
**Scope**: Total Cost of Ownership (TCO) and Expected Value of Information (EVOI) evaluation protocol for `ewm-engine` adoption decisions  

---

## 1. Introduction

- **Purpose**: Quantify Total Cost of Ownership (TCO), Return on Investment (ROI), payback period, and break-even thresholds for adopting an explicit Enterprise World Model (EWM) compared against simpler analytical baselines (Excel/spreadsheet models, deterministic rules-only heuristics, and observational ML-only predictors).
- **Core Economic Question**:
  $$\text{"When does the expected decision-regret reduction of modeling multi-layer system dynamics outweigh the incremental engineering and maintenance TCO?"}$$
- **When to Use**:
  1. Pre-pilot architecture and budget justification.
  2. Post-simulation evaluation comparing Rung 1–4 baselines against Rung 5 (`Full EWM`) on the [Decision-First Comparative Evaluation Ladder](../results/decision-first-comparative-evaluation.md).
  3. Periodic governance reviews deciding whether to retain, simplify, or retire a world model in production.

---

## 2. Cost Categories

### 2.1 Modeling Effort (Person-Hours)

Initial specification, calibration, and verification of the world model:

| Sub-Activity | Typical Effort Range (Hours) | Primary Role | Rate Assumption ($/hr) |
| :--- | :---: | :--- | :---: |
| **Scenario Design & Causal Boundary Definition** | `24 – 80` | Senior Data Scientist + Domain Expert | `$100 – $150` |
| **Typed State & Entity Schema Definition (`WorldState`)** | `8 – 32` | Senior Data Scientist / ML Engineer | `$100` |
| **Phase-Aware Constraint Specification (`PRE_ACTION`, `POST_TRANSITION`)** | `16 – 40` | Senior Data Scientist + Domain Expert | `$100 – $150` |
| **Structural & Residual Dynamics Calibration (`f_known + P_C(g_theta)`)** | `24 – 64` | Senior Data Scientist | `$100` |
| **Property, Metamorphic & Conformance Testing** | `16 – 40` | Senior Data Scientist / QA Engineer | `$100` |
| **Total Initial Modeling Effort** | **`88 – 256` hours** | Blended Rate | **`$100 – $125/hr`** |

**Hourly Rate Assumptions**:
- `$100/hour`: Senior Data Scientist / Simulation Engineer (loaded base benchmark; sensitivity range `$75–$150/hour`).
- `$150/hour`: Principal Domain Architect / Subject-Matter Expert.

### 2.2 Data Preparation (Person-Hours + $)

Point-in-time (`BitemporalLedger.as_of(t)`) data extraction, cleaning, and replay wiring:

- **Source Extraction**: `8 – 40` hours (extracting event logs, queue telemetry, and ledger snapshots).
- **Cleaning & Bitemporal Alignment**: `8 – 32` hours (eliminating lookahead leakage and aligning state intervals).
- **Pipeline & Connector Setup**: `16 – 40` hours (automated ingestion and schema validation).
- **Total Data Preparation**: **`32 – 112` hours** (`$3,200 – $11,200` at `$100/hour`).

### 2.3 Compute Resources ($)

Because the `ewm-engine` deterministic core executes on CPU without mandatory GPU or LLM inference dependencies, compute costs are explicitly separated into:

- **Deterministic CPU Core Execution**:
  $$\text{Compute Cost}_{\text{CPU}} = N_{\text{runs}} \times T_{\text{run (hours)}} \times C_{\text{CPU-hour}}$$
  At standard cloud CPU pricing (`$0.05/CPU-hour`) and typical rollout durations (`5–120` seconds per multi-seed evaluation), annual CPU compute is typically `< $5/year`.
- **Optional GPU / LLM Boundary Adapters** (only if `[ml]`, `[jax]`, `[cuda]`, or `[mcp]` boundary agents are enabled):
  $$\text{Compute Cost}_{\text{Boundary}} = N_{\text{queries}} \times C_{\text{token/GPU-hour}}$$
- **Infrastructure & Artifact Storage**:
  $$\text{Storage Cost} = \text{Volume}_{\text{GB}} \times C_{\text{GB-month}} \times 12$$

### 2.4 Maintenance (Person-Hours/Year)

Ongoing operational hygiene required to prevent model staleness under regime drift:

- **Policy & Rule Updates**: `16 – 40` hours/year (updating structural rules when business policies change).
- **Quarterly Drift Recalibration & TARP / Conformal Audits**: `16 – 48` hours/year (`4–12` hours/quarter).
- **Total Annual Maintenance**: **`32 – 88` hours/year** (`$3,200 – $8,800/year`).

---

## 3. Benefit Categories

### 3.1 Decision Improvement ($)

Quantified via pre-deployment counterfactual regret reduction across candidate interventions $\mathcal{A}$:

$$\Delta \text{Regret}_{\text{episode}} = \text{Regret}(\pi_{\text{baseline}}) - \text{Regret}(\pi_{\text{EWM}})$$

$$\text{Annual Decision Benefit} = N_{\text{decisions/year}} \times \Delta \text{Regret}_{\text{episode}}$$

Component breakdown:
- **Direct Policy Regret Reduction ($)**: Avoided net margin erosion, operational overload cost, and false-positive revenue decline.
- **SLA Violation Avoidance ($)**: $\Delta \mathbb{P}(\text{SLA Breach}) \times \text{Contractual / Churn Penalty}$.
- **Construct-Validity Protection ($)**: Avoided losses from optimizing narrow single-team proxy objectives (e.g., fraud-only loss minimization ignoring downstream call-center collapse).

### 3.2 Process Efficiency (Hours Saved)

- **Automated Counterfactual What-If Analysis**: Replaces manual cross-departmental spreadsheet reconciliation (`10 – 40` hours saved per quarterly policy review).
- **Automated Audit & Trace Generation**: `SystemicTrace` and `LeakageCertificate` artifacts eliminate manual compliance documentation overhead.

### 3.3 Risk Reduction ($)

- **Pre-Action Hard Constraint Gate (`PRE_ACTION`)**: Blocks invalid or conservation-violating interventions prior to deployment.
- **Out-of-Validity Abstention (`"DO_NOT_USE_ME_HERE"`)**: Prevents automated policy execution during unmodeled structural regime shifts (`ValidityLimitAbstentionGate`).

---

## 4. ROI Calculation

### Core Financial Metrics

1. **Total First-Year Cost ($\text{TCO}_1$)**:
   $$\text{TCO}_1 = C_{\text{modeling}} + C_{\text{data}} + C_{\text{compute}} + C_{\text{maintenance}}$$

2. **Net Annual Benefit ($B_{\text{net}}$)**:
   $$B_{\text{net}} = B_{\text{decision}} + B_{\text{efficiency}} + B_{\text{risk}} - \text{TCO}_1$$

3. **First-Year Return on Investment ($\text{ROI}$)**:
   $$\text{ROI} = \frac{\text{Total Annual Benefit} - \text{TCO}_1}{\text{TCO}_1} \times 100\%$$

4. **Payback Period (Years / Days)**:
   $$\text{Payback Period (years)} = \frac{\text{TCO}_1}{\text{Total Annual Benefit}}, \qquad \text{Payback (days)} = \text{Payback (years)} \times 365$$

5. **Break-Even Decision Stake ($\Delta \text{Regret}_{\min}$)**:
   $$\Delta \text{Regret}_{\min} = \frac{\text{TCO}_1}{N_{\text{decisions/year}}}$$
   If the expected per-decision regret reduction is below $\Delta \text{Regret}_{\min}$, an EWM is **not** economically justified and a simpler baseline should be used.

---

## 5. Comparison Baselines

Every EWM TCO evaluation must compare against three mature, lower-specification alternatives:

| Baseline Family | Typical Initial Effort | First-Year TCO | Counterfactual & Cascade Capability | Maintenance Profile | When Baseline Wins |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **5.1 Excel / Spreadsheet Model** | `24 – 40` hrs (`$2.4K – $4.0K`) | `$4,000 – $6,000` | Static single-layer formulas; no closed-loop queue or feedback dynamics | Brittle manual formula edits (`16+` hrs/yr) | Low decision stakes (`< $10K/yr`), single-layer accounting |
| **5.2 Rules-Only Heuristic (`Rung 1` / `arch_rules_only`)** | `16 – 32` hrs (`$1.6K – $3.2K`) | `$3,200 – $5,600` | Encodes known rules without empirical friction or uncertainty calibration | Low (`8–16` hrs/yr), ad-hoc threshold tuning | Stationary single-layer regimes (`Negative Result N4`) |
| **5.3 ML-Only Prediction (`Rung 2` / `Rung 3` GBT)** | `60 – 100` hrs (`$6.0K – $10.0K`) | `$9,000 – $15,000` | High 1-step observational accuracy (`Negative Result N1`), but $P(Y \mid X) \ne P(Y \mid \text{do}(X))$ under interventions | High (`32–64` hrs/yr), retraining under covariate shift | Purely observational forecasting without structural interventions |
| **5.4 Full EWM (`Rung 5`)** | `120 – 180` hrs (`$12.0K – $18.0K`) | `$18,000 – $25,000` | Coupled multi-layer dynamics, KKT conservation projection, bitemporal context $K_t$, abstention gates | Moderate (`32–48` hrs/yr), modular rule + residual updates | Multi-layer cascades, capacity/conservation constraints, policy interventions |

---

## 6. Sensitivity Analysis

To prevent point-estimate overconfidence, every TCO evaluation must compute a multi-parameter sensitivity grid across at least $\pm 20\%$ perturbations:

1. **Hourly Engineering Rate**: Vary across `$75/hour` (`-25%`), `$100/hour` (base), and `$150/hour` (`+50%`).
2. **Modeling & Data Effort Hours**: Vary across `-20%` (experienced team with existing schemas), `Base`, and `+20%` to `+100%` (first-time adoption with legacy data).
3. **Decision Improvement Magnitude ($\Delta \text{Regret}$)**: Vary across `-20%`, `Base`, and `+20%`, plus compute the exact break-even degradation percentage where $\text{ROI} = 0\%$.
4. **Decision Frequency ($N_{\text{decisions/year}}$)**: Evaluate `1/year` (annual strategic review), `4/year` (quarterly policy recalibration), and `12/year` (monthly operational tuning).

---

## 7. Limitations

1. **Synthetic & In-Model Estimand Caveat ($P_{\text{model}}$ vs. $P_{\text{real}}$)**:
   - Simulated regret reductions are computed under the declared world model ($P_{\text{model}}$). Realized production ROI depends on structural fidelity, unobserved confounders, and operational execution compliance.
2. **Experience & Learning-Curve Variance**:
   - Baseline hour estimates assume an engineer familiar with Python, Pydantic, and causal/simulation concepts; novice teams may require $2\times\text{–}3\times$ initial ramp-up effort.
3. **Organizational & Change-Management Costs**:
   - Cross-team alignment (e.g., aligning Fraud Operations and Call-Center Operations on a shared enterprise objective) incurs organizational friction not captured in engineering person-hours.
4. **Multi-Scenario Amortization**:
   - Initial schema and bitemporal connector investments amortize across subsequent scenarios (`Scenario 2..N` typically require `40–60%` less effort than `Scenario 1`).

---

## 8. Application Template

Use the following markdown table (or the companion CSV spreadsheet [`docs/methodology/tco_template.csv`](tco_template.csv)) to evaluate a candidate scenario:

```markdown
| Category | Item | Hours | Rate ($/hr) | Annual Cost / Benefit ($) | Notes / Source |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Cost: Modeling** | Scenario design & causal graph | [ ] | $100 | $[ ] | Domain + DS |
| **Cost: Modeling** | State schema & constraints | [ ] | $100 | $[ ] | `WorldState`, `ConstraintRegistry` |
| **Cost: Modeling** | Dynamics calibration & tests | [ ] | $100 | $[ ] | `f_known + g_theta`, pytest |
| **Cost: Data Prep** | Extraction, cleaning & pipeline | [ ] | $100 | $[ ] | `BitemporalLedger.as_of(t)` |
| **Cost: Compute** | Development + production runs | - | - | $[ ] | CPU-hours × $/hr |
| **Cost: Maintenance** | Quarterly recalibration | [ ] | $100 | $[ ] | 4 quarters/year |
| **TOTAL COST (TCO_1)** | **First-Year Total Cost** | **[ ]** | - | **$[ ]** | Sum of cost rows |
| **Benefit: Decision** | Regret reduction × frequency | - | - | $[ ] | `N_decisions × (Regret_base - Regret_EWM)` |
| **Benefit: Efficiency** | Analyst hours saved | [ ] | $100 | $[ ] | Automated what-if rollouts |
| **TOTAL BENEFIT** | **First-Year Total Benefit** | - | - | **$[ ]** | Under P_model |
| **NET ROI & PAYBACK** | **ROI (%) / Payback (days)** | - | - | **[ ]% / [ ] days** | `(Benefit - Cost) / Cost` |
```

---

## Appendix A: Comparison to Published Enterprise Simulation Benchmarks

| Benchmark Source | Typical Initial Project Cost | Typical 1–3 Year ROI | Primary Cost Drivers | Alignment with EWM TCO Protocol |
| :--- | :---: | :---: | :--- | :--- |
| **Law & Kelton (2014), *Simulation Modeling and Analysis* (5th ed.)** | `$25,000 – $150,000` (`150–800` person-hours) | `250% – 1,500%` | Conceptual modeling (`30%`), data collection (`30%`), verification/validation (`40%`) | Matches EWM's `136h` modeling + `40h` data prep split (`$20.8K` first-year TCO) for a bounded 4-layer operational cascade |
| **Gartner (2024), *Digital Twin & Enterprise Simulation TCO*** | `$50,000 – $500,000` | `300% – 1,000%` over 3 years | Custom data integration, 3D visualization, cloud compute | `ewm-engine` eliminates 3D rendering and mandatory GPU costs, reducing initial TCO for tabular/queueing socio-technical systems to `~$20K–$35K` |
| **McKinsey (2023), *Value of Advanced Decision Automation*** | `$35,000 – $250,000` per workflow | `400% – 2,000%` in high-frequency operational queues | Cross-silo objective misalignment and post-deployment drift | Directly addressed by EWM's construct-validity checks (`Cautionary Case A`) and runtime context discovery ($K_t$) |
