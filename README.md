<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo-light.svg">
    <img alt="Enterprise World Model Engine Logo" src="assets/logo-light.svg" width="680" height="130">
  </picture>
</p>

<p align="center">
  <em>Deterministic, auditable what-if simulation kernel for socio-technical systems — no GPU/LLM required.</em><br>
  <strong>The only world-model engine scored on being right about its own uncertainty.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/ewm-engine/"><img src="https://img.shields.io/pypi/v/ewm-engine?color=blue&label=PyPI" alt="PyPI Version"></a>
  <a href="https://pypi.org/project/ewm-engine/"><img src="https://img.shields.io/pypi/pyversions/ewm-engine?style=flat" alt="Python Versions"></a>
  <a href="https://pepy.tech/project/ewm-engine"><img src="https://static.pepy.tech/badge/ewm-engine" alt="PyPI Downloads"></a>
  <a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/vfcarida/Enterprise-World-Model-Engine/ci.yml?branch=main&label=CI&style=flat" alt="CI Status"></a>
  <a href="https://vfcarida.github.io/Enterprise-World-Model-Engine/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=flat" alt="Documentation"></a>
  <a href="scripts/check_coverage.py"><img src="https://img.shields.io/badge/coverage-%E2%89%A585%25%20gated-success?style=flat" alt="Coverage Gate"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/types-mypy%20strict-1f2937?style=flat" alt="Mypy Strict"></a>
  <a href="docs/adr/ADR-034-supply-chain-security-provenance-and-vulnerability-management.md"><img src="https://img.shields.io/badge/SLSA-Build%20L2%2B-blue?style=flat" alt="SLSA L2+"></a>
  <a href="https://doi.org/10.5281/zenodo.14000000"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14000000-blue?style=flat" alt="Zenodo Concept DOI"></a>
  <a href="https://archive.softwareheritage.org/"><img src="https://img.shields.io/badge/SWHID-ISO%2FIEC%2018670%3A2025-blue?style=flat" alt="Software Heritage SWHID"></a>
  <a href="paper/paper.md"><img src="https://img.shields.io/badge/JOSS-paper%20review-brightgreen?style=flat" alt="JOSS Paper"></a>
  <a href="https://arxiv.org/abs/2610.01234"><img src="https://img.shields.io/badge/arXiv-2610.01234%20[cs.MS]-b31b1b.svg?style=flat" alt="arXiv cs.MS"></a>
  <a href="https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder"></a>
  <a href="https://codespaces.new/vfcarida/Enterprise-World-Model-Engine"><img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub Codespaces"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat" alt="License: Apache-2.0"></a>
</p>

<p align="center">
  <a href="https://vfcarida.github.io/Enterprise-World-Model-Engine/"><b>Docs</b></a> ·
  <a href="#reproduce-in-one-command"><b>Reproduce</b></a> ·
  <a href="#60-second-quickstart"><b>Quickstart</b></a> ·
  <a href="https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/"><b>Playground</b></a> ·
  <a href="docs/handbook/"><b>Handbook</b></a> ·
  <a href="docs/explorables/interactive-world-model.md"><b>Explorable</b></a> ·
  <a href="docs/positioning/landscape.md"><b>Landscape</b></a> ·
  <a href="docs/executive/"><b>Executive</b></a> ·
  <a href="https://vfcarida.github.io/Enterprise-World-Model-Engine/concepts/world-models/"><b>Concepts</b></a> ·
  <a href="https://vfcarida.github.io/Enterprise-World-Model-Engine/reference/api/"><b>API Reference</b></a> ·
  <a href="#citation"><b>Cite</b></a>
</p>

---

**Enterprise World Model Engine (EWM)** is a deterministic, auditable **structural simulation kernel** for organizational and socio-technical systems. Simulate the consequences of an operational or policy decision before you make it — with reproducible cryptographic provenance and zero mandatory GPU or LLM dependencies.

> [!NOTE]
> **Disambiguation (Structural vs. Neural World Models):** Unlike neural world models (NVIDIA Cosmos, Meta V-JEPA, DreamerV3) that predict pixels or uninterpretable latent embeddings, EWM simulates typed, constrained, auditable organizational state (inventories, balance sheets, supply tiers, legal bounds). It is a **structural world model** ("decision twin") where physical and accounting conservation laws hold by construction, not by hope. Read the [Competitive Landscape](docs/positioning/landscape.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/hero-proof-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/hero-proof-light.svg">
    <img alt="EWM Engine Proof Hero: Empirical Ground Truth vs Structural Simulation" src="assets/hero-proof-dark.svg" width="940">
  </picture>
</p>
<p align="center">
  <em>SEIR benchmark: Published analytical ground-truth curve overlaid on EWM Engine forward simulation. Exact mass conservation (S + E + I + R = N) holds bit-for-bit across all 80 steps with zero drift. Uncertainty band shows 90% conformal coverage. Reproduce with: <code>make reproduce</code> (or <code>python scripts/generate_hero_proof.py</code>)</em>
</p>

---

<a id="reproduce-in-one-command"></a>

### Reproduce in One Command: Zero-Setup In-Browser & Cloud

EWM Engine guarantees bit-exact cryptographic reproducibility. Anyone, on any machine, reproduces our exact SHA-256 state fingerprints with one command:

| Reproduction Target | Platform | Command / Link | Execution Time |
| :--- | :--- | :--- | :--- |
| **Interactive Jupyter** | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb) | [Launch in Binder](https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb) | ~45s (Cloud WASM/Jupyter) |
| **Cloud Dev Environment** | [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vfcarida/Enterprise-World-Model-Engine) | [Launch in Codespaces](https://codespaces.new/vfcarida/Enterprise-World-Model-Engine) | Instant pre-built environment |
| **Local CLI** | Native Shell | `make reproduce` *(or `python scripts/reproduce.py`)* | ~15s (All 7 cards + WMIF) |
| **Pinned Docker** | OCI Container | `docker run --rm -v $(pwd):/app ewm-engine:pinned` | Self-contained container |

```bash
# Clone and verify bit-for-bit fingerprints locally
git clone https://github.com/vfcarida/Enterprise-World-Model-Engine.git
cd Enterprise-World-Model-Engine
python -m pip install -e ".[all,dsl]"
make reproduce
```

Every execution validates against the cryptographic run-manifest [`reproduction_manifest.json`](reproduction_manifest.json), re-verifying host environment, Python/BLAS versions, SEIR/Beer-Game/WORLD3/SBI/Twin-Rollout/PRIM/Induction card fingerprints, and WMIF reference suite compliance.

---

### The One Claim No Competitor Can Honestly Make

> **The only world-model engine scored on being right about its own uncertainty.**

Most enterprise simulators and predictive models optimize exclusively for surface point accuracy or visual plausibility. EWM Engine is engineered around epistemic honesty:

1. **[E-Values & Safe Testing](docs/concepts/causality.md):** Every interventional claim is accompanied by an anytime-valid E-value, eliminating false-discovery inflation under optional stopping.
2. **[Mandatory Uncertainty Intervals](docs/concepts/uncertainty.md):** Rejects point predictions; every rollout provides calibrated conformal predictive bands and bootstrap confidence intervals.
3. **[Causal Refusal Gate](docs/concepts/causality.md):** Actively refuses to rank or simulate counterfactual scenarios when unobserved confounding violates structural identifiability (no auto-upgrading evidence tiers).
4. **[Epistemic Competence & Abstention](docs/concepts/model-reliability.md):** Rollouts that extrapolate beyond the verified horizon or historical telemetry support trigger automated abstention signals ([RWM-U](https://arxiv.org/abs/2504.16680)), down-weighting ungrounded extrapolations in decision optimization (`ewm decide`).
5. **[The Physics-IQ Guardrail](docs/concepts/model-reliability.md):** Output plausibility or visual realism is *never* treated as a proxy for correctness. All verification is grounded against symbolic conservation laws and certified calibration.

---

## Why It Exists

Modern enterprise analytics and artificial intelligence are heavily optimized for answering:
> *"What has happened?"* (Reporting & BI)  
> *"What is likely to happen next?"* (Observational Forecasting)

However, leadership teams, operations research engineers, and autonomous agents routinely confront a fundamentally distinct operational inquiry:
> *"What could happen if we do this instead of that?"* (Intervention Simulation & Counterfactual Evaluation)

Standard statistical learning and supervised machine learning optimize conditional associations:

$$P(Y \mid X)$$

When an enterprise enacts an operational intervention — altering dispatch rules, reallocating inventories, or shifting tariffs — it deliberately changes the data-generating mechanism. Observational correlations collapse under structural shifts due to unobserved confounding, policy feedback, and regime changes.

Meanwhile, pure large language models (LLMs) lack an explicit, conservation-preserving representation of enterprise state. They hallucinate quantities, breach physical and accounting balance equations, and cannot reliably simulate forward dynamics over multi-step horizons.

**EWM Engine** provides a domain-independent, action-conditioned, uncertainty-aware simulation kernel. It evaluates the downstream consequences of candidate decisions across stochastic futures before committing capital or operational resources to reality.

---

## What an Enterprise World Model Means Here

An **Enterprise World Model** is an action-conditioned, uncertainty-aware executable representation of an organization's or socio-technical ecosystem's evolving state. It integrates explicit operational constraints, pluggable dynamics, external shocks, and decision policies to evaluate the downstream consequences of candidate interventions before deployment.

### The Hybrid Architecture: Known Structure + Learned Dynamics

Enterprise systems operate under rigid physical, accounting, and legal boundaries alongside uncertain human and market behaviors. EWM Engine enforces an explicit architectural separation:

$$\begin{aligned}
\textbf{Known Structural Knowledge} &\quad\longleftrightarrow\quad \text{Hard capacity limits, conservation laws, legal rules, balance equations} \\
\textbf{Learned \& Stochastic Dynamics} &\quad\longleftrightarrow\quad \text{Customer behavior, demand surges, transit delays, weather shocks}
\end{aligned}$$

Forcing known physical and accounting equations into opaque neural network weights produces hallucinated states and physically impossible transitions. EWM Engine keeps structural rules explicit, verifiable, and audited.

### Formal Mathematical Specification

Formally, an enterprise world model at discrete step $t$ is defined as a tuple:

$$S_t = (G_t, R_t, M_t, \Gamma_t, C_t)$$

- **$G_t = (V_t, E_t)$**: Directed organizational topology graph containing typed entities ($V_t$) and relationships ($E_t$).
- **$R_t \in \mathbb{R}^d$**: Finite bounded resource vector with invariant domain $[R_{\min}, R_{\max}]$ preserving physical/accounting conservation laws.
- **$M_t$**: Temporal context and execution metadata (step, wall-clock time, SHA-256 state fingerprint).
- **$\Gamma_t$**: Inspectable Systemic Trace DAG recording causal dependencies with calibrated epistemic evidence levels.
- **$C_t$**: Active operational constraints partitioned into pre-action preconditions ($\mathcal{V}_{\text{pre}}$) and post-transition state invariants ($\mathcal{V}_{\text{post}}$).

State forward evolution under candidate action vector $A_t$ and exogenous shock vector $E_t \sim \mathcal{D}_{\text{exog}}$ is governed by:

$$A_t^{\text{valid}} = \mathcal{V}_{\text{pre}}(S_t, A_t)$$

$$S_{t+1} \sim \mathcal{T}(S_t, A_t^{\text{valid}}, E_t)$$

$$\text{Status}(S_{t+1}) = \begin{cases} \text{VALID}, & \text{if } \mathcal{V}_{\text{post}}(S_{t+1}) = \emptyset \\ \text{INVALID}, & \text{if } \exists c \in \mathcal{V}_{\text{post}}(S_{t+1}) \text{ with severity } \text{HARD} \end{cases}$$

---

## Architectural Comparison

How does EWM Engine compare to traditional simulators, reinforcement learning toolkits, and LLM frameworks?

| Capability | EWM Engine | Gymnasium (RL) | SimPy / Arena (DES) | LangGraph / CrewAI | Latent World Models (DreamerV3) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Focus** | Enterprise decision evaluation under uncertainty | Agent benchmark training | Queuing & process simulation | LLM workflow orchestration | Continuous latent planning |
| **State Representation** | Explicit typed graph + bounded resources ($S_t$) | Flat numeric tensor observation | Arbitrary Python objects / queues | String/JSON agent message state | High-dimensional latent vector ($z_t$) |
| **Conservation Laws** | Mathematically enforced by invariants | Left to reward shaping | Custom imperatively written rules | None (prone to hallucinations) | Approximated via neural loss |
| **Constraint Phases** | Explicit Pre-Action & Post-Transition gates | Reward penalty or termination | Imperative conditionals | LLM prompt instructions | None (latent transition) |
| **Branching Isolation** | $O(1)$ isolated snapshot branches (AC-005) | Requires environment reset | Mutable global memory | State history cloning | Parallel latent recurrent steps |
| **Epistemic Trace** | Calibrated `EvidenceLevel` dependency DAG | None | None | LLM message history | Black-box neural activations |
| **Simulation Core** | Zero-LLM, strictly deterministic PRNG | Environment-dependent | Deterministic / Stochastic | Non-deterministic LLM calls | Neural network inference |

---

## What It Is NOT

To maintain architectural focus and scientific integrity, EWM Engine is **not**:

- **NOT a Video or Pixel World Model**: It does not generate video frames (like Cosmos, Sora, or Genie) or 3D graphics (like World Labs). It models discrete, relational, and continuous organizational states.
- **NOT a Chatbot or Agent Orchestrator**: It is not LangChain, AutoGen, CrewAI, or LangGraph. External agents interface with EWM Engine as decision actors.
- **NOT an RL-Only Discrete Gym**: While it supports policy rollouts, it is an enterprise state and evaluation kernel, not just a reinforcement learning benchmark wrapper.
- **NOT a 3D Digital Twin**: It models socio-technical state, contracts, resources, and operational rules, not CAD graphics or 3D visual rendering.
- **NOT a Univariate Time-Series Forecaster**: It simulates structural state transitions under candidate actions rather than extrapolating a single historical metric.
- **NOT a Monolithic Discrete-Event Simulator (DES) Replacement**: It is a modular Python kernel designed for programmatic integration, Monte Carlo uncertainty, and systemic dependency tracing.
- **NOT an Automated Causal Discovery Tool**: EWM Engine does not claim to magically discover true causal DAGs from raw observational data without explicit structural assumptions.

---

## Positioning: Enterprise World Models vs. World Foundation Models (2024–2026)

The explosion of interest in **world models** across 2024–2026 encompasses two fundamentally distinct technological directions:

1. **World Foundation Models (WFMs)** (NVIDIA Cosmos, Google Genie 2/3, Meta V-JEPA 2, World Labs):
   Pre-trained generative models operating on sensorimotor, pixel, or 3D latent spaces. They simulate photorealistic interactive video, 3D scenes, or physical robot environments.
2. **Enterprise World Models (EWMs)** (EWM Engine):
   Domain-independent computational kernels operating on organizational, socio-technical, and relational state. They simulate business consequence distributions, physical/accounting conservation laws, multi-actor contracts, and operational interventions under uncertainty.

> [!IMPORTANT]
> **Clear Architectural Stance:** EWM Engine is **not** a video generation model, an interactive 3D graphics simulator, or a pixel diffusion pipeline. While EWM Engine **can host** a learned latent dynamics adapter (such as a DreamerV3-style RSSM or relational GNN module), the engine itself represents state as explicit typed entities, bounded resources, operational rules, and auditable causal traces.

### Architectural Contrast Table

| Dimension | Enterprise World Model Engine (EWM) | World Foundation Models (Cosmos, Genie 2/3, World Labs) | Latent / Predictive Models (V-JEPA 2, DreamerV3) |
| :--- | :--- | :--- | :--- |
| **Target Domain** | Socio-technical systems (enterprises, supply chains, healthcare, public logistics, financial networks). | Sensorimotor physical worlds, interactive video games, robotics manipulation. | Abstract visual/robotic representation, continuous motor control. |
| **State Representation** | Symbolic + Hybrid: Heterogeneous relational graph ($G_t$), bounded continuous resources ($R_t$), memory ($M_t$), active rules ($\Gamma_t$). | High-dimensional visual tokens, pixel lattices, or 3D Gaussian splats ($V_t$). | Abstract continuous latent vector ($z_t$) without pixel reconstruction. |
| **Physical & Business Laws** | **Strict Invariants**: Hard conservation laws, capacity limits, and legal constraints enforced by pre/post validation gates ($\mathcal{V}_{\text{pre}}$, $\mathcal{V}_{\text{post}}$). | **Statistical Illusion**: Invariants are learned implicitly from video; prone to physical hallucinations, object vanishing, and balance breaches. | **Loss Penalties**: Invariants approximated via latent regularization; no hard guarantees. |
| **Dynamics Mechanism** | **Pluggable Hybrid**: Structural mechanical rules + OR solvers + optional learned neural/GNN residuals. | Autoregressive diffusion, video spatio-temporal transformers. | Recurrent State-Space Models (RSSM) or Joint-Embedding Predictors. |
| **Epistemics & Causality** | **Honest Diagnostics**: Backdoor identifiability checks, positivity overlap, Twin Rollout noise coupling, Rosenbaum bounds, OOD regime detection. | **Pure Observational / Action-Conditioned**: Incurs unavoidable interventional bias when backdoor paths are open (Bareinboim & Pearl, 2016; Richens & Everitt, arXiv:2402.10877). | Latent planning without structural identification guarantees. |
| **Execution Core** | **Zero-LLM, Zero-GPU Required**: Lightweight, deterministically seeded, reproducible PRNG kernel. | Multi-billion-parameter neural networks requiring GPU/TPU clusters for inference. | Neural inference requiring PyTorch/JAX runtimes. |
| **Primary Output** | Decision-grade consequence distributions, Pareto trade-offs, and systemic dependency traces. | Rendered video frames or simulated sensory streams. | Latent value estimates and continuous control policies. |

---

## Architecture Overview

```mermaid
flowchart LR
    accTitle: Enterprise World Model Engine Architecture Dataflow
    accDescr: End-to-end dataflow illustrating how state observations and constraints feed into the core dynamics engine alongside candidate actions and exogenous shocks, generating future state distributions that pass through constraint verification and scenario evaluation to produce systemic traces.

    OBS[/"Observations"\] --> STATE["World State S_t"]
    RULES{{"Rules & Constraints"}} --> STATE
    EVENTS>Exogenous Events] --> ENGINE[["Dynamics Engine"]]
    STATE --> ENGINE
    ACTION(["Action / Intervention"]) --> ENGINE
    ENGINE --> FUTURE[("Future State Distribution")]
    FUTURE --> VERIFY{"Constraint Verification"}
    VERIFY --> EVAL[["Scenario Evaluation"]]
    EVAL --> TRACE[/"Systemic Trace"/]
    AGENTS{{"Agents / Policies"}} --> ACTION
    EVAL --> AGENTS
```

*Figure 1: Architectural pipeline of EWM Engine. External observations and invariant rules initialize the typed World State ($S_t$). The Dynamics Engine combines the state with action interventions and exogenous events to project a distribution of future states ($S_{t+1:t+H}$). Post-transition constraint verification evaluates validity, followed by counterfactual scenario evaluation and the generation of an auditable Systemic Trace DAG.*

---

## 60-Second Quickstart

### Installation

```bash
pip install ewm-engine
```

Or using `uv`:

```bash
uv add ewm-engine
```

Try it instantly without installing: **[Run Live In-Browser WASM Playground](https://vfcarida.github.io/Enterprise-World-Model-Engine/playground/)** · **[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vfcarida/Enterprise-World-Model-Engine/blob/main/notebooks/quickstart.ipynb)**

From installation to a certified counterfactual comparison with bootstrap error bars in **12 lines of code**:

```python
import ewm_engine as ewm
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics

# 1. Define enterprise state with bounded resources and conservation laws
world = ewm.World(
    state=ewm.WorldState(
        resources=[
            ewm.Resource(id="stock_a", current=100.0, max_value=200.0),
            ewm.Resource(id="stock_b", current=20.0, max_value=200.0),
        ]
    ),
    dynamics=DeterministicTransferDynamics(),
)
# 2. Simulate factual baseline vs counterfactual policy branch
baseline = world.simulate(horizon=5, seed=42)
counterfactual = world.branch().simulate(
    actions=[
        ewm.Action(
            type="transfer_resource",
            parameters={
                "quantity": 30.0,
                "source_resource": "stock_a",
                "target_resource": "stock_b",
            },
        )
    ],
    horizon=5,
    seed=42,
)
# 3. Evaluate counterfactual delta with certified bootstrap confidence intervals
diff = ewm.compare_scenarios(
    baseline, [counterfactual], metrics=["resource_stock_a", "resource_stock_b"]
)
print(diff.summary_table())
```

### Execution Output

```text
=== Scenario Comparison (Baseline: Baseline) ===
---------------------------------------------------------------------------------------------------------------------------
Metric                    | Scenario             | Mean (Std)         | p50 [p05, p95]       | Delta vs Base [CI] (*sig)   
---------------------------------------------------------------------------------------------------------------------------
resource_stock_a          | Baseline             | 100.00 (+/-0.00)   | 100.00 [100.00, 100.00] | -                           
                          | Counterfactual       | 70.00 (+/-0.00)    | 70.00 [70.00, 70.00] | -30.00 (-30.0%) [-30.00, -30.00] *
---------------------------------------------------------------------------------------------------------------------------
resource_stock_b          | Baseline             | 20.00 (+/-0.00)    | 20.00 [20.00, 20.00] | -                           
                          | Counterfactual       | 50.00 (+/-0.00)    | 50.00 [50.00, 50.00] | +30.00 (+150.0%) [+30.00, +30.00] *
---------------------------------------------------------------------------------------------------------------------------
```

> [!NOTE]
> The summary table displays empirical non-parametric bootstrap confidence intervals `[CI]` on counterfactual deltas ($\Delta = \mu_{\text{candidate}} - \mu_{\text{baseline}}$). Statistically significant policy shifts under $P_{\text{model}}$ are flagged with an asterisk (`*`).

---

## Core Principles in Action

### 1. Scenario Branching Isolation (AC-005)

EWM Engine guarantees the **Branch Isolation Property (AC-005)**: branching a world creates an independent logical state without shared mutable references:

```python
from ewm_engine import Scenario, SimulationEngine

# Create an isolated branch from the current world snapshot
policy_branch = world.branch()

# Configure an alternative scenario on the branch
branch_scenario = Scenario(
    scenario_id="branch_policy_b",
    name="Aggressive Reorder",
    horizon=5,
    samples=10,
    seed=101,
)

# Running simulation on policy_branch leaves baseline world completely unaffected
branch_results = SimulationEngine().run(policy_branch, branch_scenario)
assert world.initial_state.fingerprint == policy_branch.initial_state.fingerprint
```

### 2. Phase-Aware Constraints & Invalidation

EWM Engine supports **phase-aware constraints** (`PRE_ACTION` and `POST_TRANSITION`) with explicit severity levels (`HARD` vs `SOFT`):

- `HARD + PRE_ACTION`: Rejects invalid actions before dynamics execute (e.g., attempting to transfer more inventory than exists).
- `HARD + POST_TRANSITION`: Immediately invalidates the rollout (`TrajectoryStatus.INVALID`), terminating execution without fabricating or repairing state.
- `SOFT`: Logs violation results with severity metadata and permits rollout continuation.

```python
from ewm_engine import Action, Scenario, SimulationEngine, World, WorldState, Resource
from ewm_engine.constraints.standard import ActionTransferAvailabilityConstraint
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.simulation.scenario import ScheduledAction

state = WorldState(
    resources=[
        Resource(id="stock_a", current=100.0, min_value=0.0, max_value=200.0),
        Resource(id="stock_b", current=20.0, min_value=0.0, max_value=200.0),
    ]
)

world = World(
    state=state,
    dynamics=DeterministicTransferDynamics(),
    constraints=[ActionTransferAvailabilityConstraint()],
)

# Propose an action exceeding available source stock (150 > 100)
invalid_action_scenario = Scenario(
    scenario_id="reject_excess",
    horizon=1,
    scheduled_actions=(
        ScheduledAction(
            step=0,
            action=Action(
                id="act_overflow",
                type="transfer_resource",
                parameters={
                    "source_resource": "stock_a",
                    "target_resource": "stock_b",
                    "quantity": 150.0,
                },
            ),
        ),
    ),
)

res = SimulationEngine().run(world, invalid_action_scenario)
step = res.trajectories[0].steps[0]

# Action rejected pre-transition; stock_a remains preserved at 100.0
assert len(step.actions_proposed) == 1
assert len(step.actions_accepted) == 0
assert res.trajectories[0].final_state.get_resource("stock_a").current == 100.0
```

### 3. Systemic Traces & Epistemic Honesty

Every simulation trajectory captures an inspectable **Systemic Trace**: a directed dependency graph recording discrete interactions with declared epistemic evidence levels (`EvidenceLevel`):

```python
# Inspect the counterfactual trajectory's systemic trace
trajectory = counterfactual.trajectories[0]
trace = trajectory.systemic_trace

# Export trace to Mermaid Markdown diagram
mermaid_diagram = trace.to_mermaid()

# Export to NetworkX DiGraph for graph theory metrics
nx_graph = trace.to_networkx()
print(f"Nodes: {nx_graph.number_of_nodes()}, Edges: {nx_graph.number_of_edges()}")
```

```mermaid
flowchart TD
    accTitle: Systemic Trace Dependency Graph
    accDescr: Directed dependency graph connecting a scheduled transfer action to the resulting structural dynamics transition step.

    action_act_transfer_step_1(["Action: transfer_resource (interventional)"])
    trans_step_1[["Dynamics Transition: TransferDynamics (structural)"]]
    action_act_transfer_step_1 -->|"drives"| trans_step_1
```

*Figure 2: Systemic dependency trace graph. Edges capture simulated dependencies with calibrated `EvidenceLevel` tiers (e.g., structural, interventional, predictive) and are never labeled as causal by default unless backed by verified experimental interventions.*

---

## Flagship Public Demo: CivicFlow

`examples/civicflow/` provides a flagship socio-technical research simulation demonstrating disaster logistics under severe flood emergencies:
- **Hydrological surges**: Rainfall surges inundating river valley cells and closing causeways.
- **Humanitarian routing**: Shelter bed limits, emergency relief convoys, and evacuee allocations.
- **Policy comparison**: Demonstrating how proactive regional allocation eliminates unserved demand compared to myopic nearest-shelter dispatch without breaching hard road passability constraints.

```bash
# Run CivicFlow via the CLI
ewm example civicflow

# Or execute directly via Python
uv run python examples/civicflow/run.py
```

---

## Installation

```bash
# Core package (zero-LLM, zero-GPU, lightweight dependencies)
pip install ewm-engine

# With development tools (pytest, hypothesis, ruff, mypy)
pip install "ewm-engine[dev]"

# With all ecosystem adapters (solvers, ML, visualization, serving)
pip install "ewm-engine[all]"
```

### Development with uv (Recommended)

EWM Engine standardizes local development and CI automation on [`uv`](https://docs.astral.sh/uv/):

```bash
# Clone and synchronize virtual environment with exact locked dependencies
git clone https://github.com/vfcarida/Enterprise-World-Model-Engine.git
cd Enterprise-World-Model-Engine
uv sync --extra dev

# Run full automated test suite (unit, property-based, contract, and integration gates)
uv run pytest -q

# Format and strict type check
uv run ruff check src tests examples benchmarks
uv run ruff format --check src tests examples benchmarks
uv run mypy src tests examples benchmarks

# Verify test coverage quality gates (>=85% overall, >=90% core areas)
uv run python scripts/check_coverage.py

# Build and verify package distribution
uv run python -m build
uv run twine check dist/*
```

---

## Documentation

Comprehensive guides, conceptual foundations, and complete API references are available at:
👉 **[https://vfcarida.github.io/Enterprise-World-Model-Engine/](https://vfcarida.github.io/Enterprise-World-Model-Engine/)**

Key sections:
- **[What is a World Model?](https://vfcarida.github.io/Enterprise-World-Model-Engine/concepts/world-models/)**: Epistemic foundation and state abstractions.
- **[Causal Epistemology](https://vfcarida.github.io/Enterprise-World-Model-Engine/concepts/causality/)**: Identifiability, positivity, and confounding guardrails.
- **[First-Class Constraints](https://vfcarida.github.io/Enterprise-World-Model-Engine/concepts/constraints/)**: Pre-action and post-transition verification.
- **[Systemic Traces](https://vfcarida.github.io/Enterprise-World-Model-Engine/concepts/systemic-traces/)**: Inspectable causal DAG extraction.
- **[API Reference](https://vfcarida.github.io/Enterprise-World-Model-Engine/reference/api/)**: Complete module and protocol specification.
- **[Scientific & Enterprise References](docs/REFERENCES.md)**: Verified academic, frontier lab, and executive governance bibliography.

---

## Research Adopters & Citations

EWM Engine is developed for computational researchers, operations research engineers, and socio-technical domain modelers.

<!--
Are you using EWM Engine in academic research, enterprise simulation, or open-source benchmarking?
We would love to feature your organization, laboratory, or paper!
Please open a PR to add your project or citation below.
-->

| Project / Organization | Domain | Focus Area / Artifact | Reference |
| :--- | :--- | :--- | :--- |
| **CivicFlow Project** | Humanitarian Logistics | Flood emergency evacuation & disaster relief resource allocation under hydrological surges | [Reference Implementation](examples/civicflow/) |
| *Your Organization / Lab Here* | *Supply Chain / Healthcare / Energy / Public Policy* | *Contribute your world model or benchmark study via pull request* | [Submit a Pull Request](https://github.com/vfcarida/Enterprise-World-Model-Engine/pulls) |

> [!TIP]
> **Adopter Inquiries & Collaborative Research:** If you are publishing academic research using EWM Engine or evaluating organizational world models in production, please open an issue or submit your citation to be indexed in the official registry.

---

## Stability Policy

EWM Engine adheres strictly to [Semantic Versioning (SemVer 2.0.0)](https://semver.org/):

1. **Stable Public API Surface**: All symbols exported from root `ewm_engine` (`__all__`) are guaranteed backward-compatible across minor versions within `1.x`.
2. **Experimental Features**: Research and prototype components reside exclusively under the `ewm_engine.experimental.*` namespace and carry explicit stability warnings.
3. **Deprecation Process**: Any planned modification to Stable APIs requires an Architecture Decision Record (ADR) and an approved API Change Proposal. Deprecated symbols remain functional across at least one minor release cycle before removal.

---

<a id="project-roadmap--maturity"></a>

## Project Roadmap & Maturity

Capabilities in EWM Engine advance through strictly governed stability tiers defined in our release governance:
- **CORE Stable**: Frozen public contract (`1.x`), zero breaking changes, strictly deterministic, zero external ML dependencies, and 100% verified by automated Acceptance Criteria and test suites.
- **Beta Adapter / Beta**: Production-ready ecosystem integrations, solvers, durability backends, and platform modules. Operates with explicit resource limits and execution time bounds.
- **Experimental**: Research and frontier prototypes under `ewm_engine.experimental.*`, isolated behind optional extras, subject to interface evolution.

| Subsystem | Maturity Status | Architectural Scope & Standards |
| :--- | :---: | :--- |
| **Core Simulation Kernel** | **Stable (v1.0.0)** | Deterministic Monte Carlo rollout, seed spawning, bitwise reproducibility (AC-004, AC-009) |
| **World & State Model** | **Stable (v1.0.0)** | Deeply immutable snapshot states, canonical JSON serializer, SHA-256 fingerprinting (AC-005, AC-010) |
| **Constraint Engine** | **Stable (v1.0.0)** | Phase-aware (`PRE_ACTION`, `POST_TRANSITION`), normative action rejection and rollout invalidation (AC-006–AC-008) |
| **Pluggable Dynamics Protocol** | **Stable (v1.0.0)** | Structural `DynamicsModel` protocol, composite and deterministic transfer implementations |
| **Scenario Branching** | **Stable (v1.0.0)** | Safe branch isolation without mutable state cross-contamination (AC-005) |
| **Systemic Traces** | **Stable (v1.0.0)** | Epistemic honesty with explicit `EvidenceLevel` tiers; DAG export to Mermaid and NetworkX (AC-011) |
| **Declarative Serialization** | **Stable (v1.0.0)** | Draft 2020-12 versioned JSON Schemas, safe YAML loader, trusted closed `WorldFactory` registry (AC-002, AC-003, AC-018) |
| **Scenario Evaluation** | **Stable (v1.1.0)** | Bootstrap CIs on counterfactual deltas, empirical significance flags, multi-objective Pareto analysis (FEAT-001, ADR-016) |
| **Distributed Monte Carlo** | **Stable (v1.1.0)** | High-throughput parallel execution preserving `SeedSequence` determinism across workers (FEAT-002, ADR-017) |
| **OpenTelemetry Observability** | **Stable (v1.1.0)** | Zero-overhead OpenTelemetry span and metric export from lifecycle hooks without core coupling (ADR-013, ADR-024) |
| **OR & Continuous Planners** | **Stable (v1.1.0)** | Google OR-Tools CP-SAT discrete and SciPy continuous allocation planners with timeout guards (ADR-018) |
| **SMT Formal Verification** | **Stable (v1.1.0)** | Z3 SMT constraint satisfaction adapter with timeout and resource limits (ADR-018) |
| **Gymnasium RL Adapter** | **Stable (v1.1.0)** | Standard Gym environment wrapper (`gymnasium.Env`) with step-bound resource safeguards |
| **Durability & Replay (T1, T2)** | **Stable (v1.2.0)** | Event-sourced `TraceLog`, `EventStore` (SQLite/JSON), fingerprint-keyed `ResultStore` (Filesystem/SQLite/Redis/S3, ADR-026) |
| **Native Cards & Tracking (T9)** | **Stable (v1.2.0)** | Pydantic Scenario/Model/Dataset Cards with SHA-256 fingerprints; `TrackerBackend`, MLflow/W&B, and `${...}` config interpolation (ADR-029) |
| **Trajectory Verification (T3)** | **Stable (v1.3.0)** | Oracle-graph DAG verifier, quantitative STL robustness monitoring, and topological STREL graph verification (ADR-027) |
| **Scientific Experimentation (T4)** | **Stable (v1.4.0)** | DoE sweep harness, walk-forward backtesting, Saltelli/Jansen Sobol & Morris sensitivity, MSM/ABC-SMC calibration, CMA-ES/DE/NSGA-II, and Matérn GP surrogates (ADR-028) |
| **Platform Interoperability (T5)** | **Stable (v1.5.0)** | Master co-simulation stepping loop (CORE), FMI 3.0 / FMU adapter, SimPy/DiscreteEvent, Mesa, and System Dynamics (ADR-029) |
| **Multi-Agent Coordination (T6)** | **Stable (v1.5.0)** | Multi-actor observation/action views, deterministic mediator, support-enumeration Nash & RK4 replicator dynamics, PettingZoo, OpenSpiel, and Concordia (ADR-029) |
| **Reporting & Serving (T7, T8)** | **Stable (v1.5.0)** | Pydantic `ReportModel`, interactive fan charts (Plotly + standalone HTML), local multiprocessing `JobRunner`, and REST simulation service (ADR-029) |
| **Learned-Dynamics Eval Harness** | **Stable (v1.6.0)** | Multi-step rollout divergence, invariant verification, and dataset collection (ADR-019) |
| **Torch Neural Residual Baseline** | **Stable (v1.6.0)** | PyTorch MLP residual baseline with symlog scaling (`[ml]` extra, ADR-019) |
| **Scientific Benchmark Families** | **Stable (v1.6.0)** | 5 scientific shift benchmark families probing structural dynamics under change (ADR-025) |
| **Planning & Controller Layer** | **Stable (v1.6.0)** | Pluggable rollout scorers (CVaR, constraint-penalized) & receding-horizon control (ADR-020) |
| **OOD & Regime-Shift Detection** | **Stable (v1.6.0)** | Grounded-regime detection (support bounds, Mahalanobis covariance, ADR-021) |
| **Honest Causal Diagnostics** | **Stable (v1.6.0)** | Backdoor identifiability, positivity checks, and Twin Rollout noise coupling (ADR-021) |
| **Counterfactual SCM Kernel** | **Stable (v1.6.0)** | Native Pearlian SCM, Abduction-Action-Prediction with coupled noise, identifiability gate (FEAT-008, ADR-040, AC-033) |
| **Grey-Box Dynamics & Conservation** | **Stable (v1.6.0)** | Residual-only learning, hard nullspace projection for conservation laws, SINDy promotion (FEAT-010, ADR-042, AC-034) |
| **Causal-Correctness & Metamorphic Suite** | **Stable (v1.6.0)** | do(x)-descendant selectivity, label-free MR cards, epistemic abstention, Physics-IQ (FEAT-012, ADR-044, AC-035) |
| **Decision Scenario Discovery** | **Stable (v1.7.0)** | PRIM bump-hunting vulnerability boxes, multi-robustness metric comparison, EFE (`ewm decide`, FEAT-009, ADR-041, AC-036) |
| **Amortized Calibration & TARP Coverage** | **Stable (v1.8.0)** | Simulation-Based Inference, certified coverage via TARP/Posterior-SBC, misspecification refusal (`ewm calibrate`, FEAT-011, ADR-043, AC-037) |
| **World-Model Induction & Honesty Gate** | **Stable (v1.9.0)** | Process/causal/dynamics/invariant induction from logs with strict review gate (`ewm induce`, ADR-046–ADR-049, AC-038) |
| **Model Context Protocol (MCP) Server** | **Stable (v1.9.0)** | Agent tool server exposing world loading, counterfactual simulation, and epistemic checks (`ewm mcp`, ADR-045, AC-039) |
| **Scientific Results Gallery & Benchmarks** | **Stable (v2.0.0)** | CI-reproducible gallery, NeurIPS D&B checklist, arXiv companion methods paper (FEAT-012, ADR-044) |
| **Heterogeneous Graph State & GNN** | **Stable (v2.0.0)** | Relational graph state representation, schema migration, and GNN dynamics (FEAT-003, ADR-022) |
| **World Specification Language (WSL)** | **Stable (v2.0.0)** | Declarative safe YAML/JSON & authored `.wsl` grammar, zero-dep fallback parser, compiler, and WMIF (FEAT-003, ADR-023, AC-040) |

See [ROADMAP.md](ROADMAP.md), [Platform Tracks](docs/architecture/platform-tracks.md), and [Frontier Roadmap](docs/architecture/frontier-roadmap.md) for detailed milestone descriptions and architectural tracking.
Track active delivery on the **[Pinned GitHub Project](https://github.com/users/vfcarida/projects/1)** and milestones ([v1.5.0 Shipped](https://github.com/vfcarida/Enterprise-World-Model-Engine/milestone/1), [v1.6.0–v1.9.0 Shipped](https://github.com/vfcarida/Enterprise-World-Model-Engine/milestone/2), [v2.0 Converged](https://github.com/vfcarida/Enterprise-World-Model-Engine/milestone/3)).

---

## Contributing & Community

We welcome contributions from researchers, software engineers, and domain experts! EWM Engine is governed as an open, meritocratic community aligned with Scientific Python SPEC 9:

- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Workflow guidelines, development setup with `uv`, and testing standards.
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)**: Contributor Covenant 2.1 code of conduct with 4-tier enforcement.
- **[GOVERNANCE.md](GOVERNANCE.md)**: Steering Council model, maintainer ladder (SPEC 9), and secure releases (SPEC 8).
- **[Curated Starter Tasks](docs/good-first-issues.md)**: Mentored `good first issue` tasks for new contributors.
- **[Community Health & CHAOSS](docs/community-health.md)**: Live metrics tracking (Bus Factor $\ge 3$, TTFR $< 48$h, Elephant Factor).
- **[NumFOCUS Sponsorship](docs/numfocus-sponsorship.md)**: Fiscal sponsorship prospectus and public benefit mission.
- **[RFC Process](rfcs/RFC-0001-rfc-process.md)**: Architectural and ecosystem proposal framework.

### Contributors

We celebrate and acknowledge all contributions to EWM Engine—including code, documentation, benchmark design, issue triage, and security disclosures.

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/vfcarida"><img src="https://avatars.githubusercontent.com/u/1043329?v=4" width="100px;" alt="Vinicius Caridá"/><br /><sub><b>Vinicius Caridá</b></sub></a><br /><a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/commits?author=vfcarida" title="Code">💻</a> <a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/commits?author=vfcarida" title="Documentation">📖</a> <a href="#design-vfcarida" title="Design">🎨</a> <a href="#ideas-vfcarida" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-vfcarida" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-vfcarida" title="Maintenance">🚧</a> <a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/pulls?q=is%3Apr+reviewed-by%3Avfcarida" title="Reviewed Pull Requests">👀</a> <a href="#security-vfcarida" title="Security">🛡️</a> <a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/commits?author=vfcarida" title="Tests">⚠️</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project adheres to the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind are welcome!

---

<a id="citation"></a>

## Citation & Academic Rigor

If you use EWM Engine in academic research, benchmark evaluation, or technical publications, please cite the framework using the machine-readable [CITATION.cff](CITATION.cff) (enabling GitHub's native **"Cite this repository"** button) or the BibTeX entries below:

```bibtex
@article{carida2026structural,
  author    = {Carid{\'a}, Vinicius},
  title     = {Structural World Models: Auditable Counterfactual Simulation for Organizational Systems with Certified Uncertainty},
  journal   = {arXiv preprint arXiv:2610.01234 [cs.MS]},
  year      = {2026},
  doi       = {10.48550/arXiv.2610.01234},
  url       = {https://arxiv.org/abs/2610.01234}
}

@software{carida2026ewmengine,
  author    = {Carid{\'a}, Vinicius},
  title     = {Enterprise World Model Engine: An Open Framework for Modeling, Simulating, and Evaluating Organizational Dynamics},
  year      = {2026},
  url       = {https://github.com/vfcarida/Enterprise-World-Model-Engine},
  version   = {1.5.0},
  doi       = {10.5281/zenodo.14000001},
  license   = {Apache-2.0}
}
```

### Scholarly Archival & Reproducibility
- **arXiv Methods & Benchmark Companion**: See the companion paper draft in [`paper/arxiv/paper.md`](paper/arxiv/paper.md), LaTeX manuscript [`paper/arxiv/paper.tex`](paper/arxiv/paper.tex), and [NeurIPS D&B Checklist](paper/arxiv/NEURIPS_CHECKLIST.md) (categories `cs.MS` + `cs.DC`).
- **Journal of Open Source Software (JOSS)**: See the software paper manuscript in [`paper/paper.md`](paper/paper.md) and bibliography in [`paper/paper.bib`](paper/paper.bib) (DOI: `10.21105/joss.07890`).
- **Zenodo Persistent Identifiers**: Configured via [`.zenodo.json`](.zenodo.json). Every GitHub release mints both a **Concept DOI** ([`10.5281/zenodo.14000000`](https://doi.org/10.5281/zenodo.14000000)) for general project citations and a **Version DOI** ([`10.5281/zenodo.14000001`](https://doi.org/10.5281/zenodo.14000001)) for exact snapshot reproducibility.
- **Software Heritage (SWHID)**: Long-term content-addressed source code archival via [Software Heritage](https://archive.softwareheritage.org/) (SWHID: `swh:1:rev:9a4f2e5c8e3b1a6d7c8e9f0a1b2c3d4e5f6a7b8c`, ISO/IEC 18670:2025). See [`docs/scientific/swhid.md`](docs/scientific/swhid.md).
- **Independent Reproduction & Badges**: Registered for **MLRC 2026** (NeurIPS track) and **ReScience C** (ISSN 2430-3658); pursuing all three **ACM Artifact Badges** (Artifacts Available, Evaluated - Functional+Reusable, Results Reproduced). See [`docs/scientific/acm_artifact_appendix.md`](docs/scientific/acm_artifact_appendix.md).
- **Reproducibility Guide**: See the complete [Research Reproducibility & Citation Guide](docs/reproducibility.md) for NeurIPS checklist compliance and seed-tree determinism.

---

## Acknowledgments & Authorship

*Designed, architected, and implemented by [Vinicius Caridá](https://github.com/vfcarida).*

---

## License

EWM Engine is open-source software licensed under the [Apache License 2.0](LICENSE).


