# Enterprise World Model Engine (EWM Engine)

**Simulate consequences before acting.**

An open-source framework for building executable, action-conditioned models of complex organizational and socio-technical systems.

<p align="center">
  <a href="https://pypi.org/project/ewm-engine/"><img src="https://img.shields.io/pypi/v/ewm-engine?color=blue&label=PyPI" alt="PyPI Version"></a>
  <a href="https://doi.org/10.5281/zenodo.14000000"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.14000000-blue?style=flat" alt="Zenodo Concept DOI"></a>
  <a href="https://archive.softwareheritage.org/"><img src="https://img.shields.io/badge/SWHID-ISO%2FIEC%2018670%3A2025-blue?style=flat" alt="Software Heritage SWHID"></a>
  <a href="https://arxiv.org/abs/2610.01234"><img src="https://img.shields.io/badge/arXiv-2610.01234%20[cs.MS]-b31b1b.svg?style=flat" alt="arXiv cs.MS"></a>
  <a href="https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb"><img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder"></a>
  <a href="https://codespaces.new/vfcarida/Enterprise-World-Model-Engine"><img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub Codespaces"></a>
  <a href="https://github.com/vfcarida/Enterprise-World-Model-Engine/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue?style=flat" alt="License: Apache-2.0"></a>
</p>

!!! tip "Reproduce All Published Results in One Command"
    EWM Engine's results gallery and SHA-256 state fingerprints are 100% reproducible:
    - **In-Browser Jupyter**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb) [Launch Binder](https://mybinder.org/v2/gh/vfcarida/Enterprise-World-Model-Engine/HEAD?filepath=notebooks%2Freproduce.ipynb)
    - **Cloud Codespaces**: [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/vfcarida/Enterprise-World-Model-Engine) [Open Codespaces](https://codespaces.new/vfcarida/Enterprise-World-Model-Engine)
    - **Local Terminal**: `make reproduce` or `python scripts/reproduce.py`
    - **Manifest**: Every run verifies against [`reproduction_manifest.json`](results/index.md#run-manifest-provenance).

---

## The Core Thesis

Traditional machine learning optimizes specialized predictive mappings:
$$\text{input} \longrightarrow \text{prediction}$$

Examples include:
- `text` $\to$ `intent`
- `customer history` $\to$ `churn probability`
- `sensor stream` $\to$ `failure probability`
- `demand time-series` $\to$ `demand forecast`

Foundation models unified representations across tasks, but **observational prediction is not an executable model of how a system evolves under intervention**.

An **Enterprise World Model** formalizes an executable representation of organizational dynamics:

$$\begin{aligned}
&\text{Current World State } (S_t) \\
+\; &\text{Intervention / Proposed Action } (A_t) \\
+\; &\text{Exogenous Environmental Shocks } (E_t) \\
+\; &\text{Systemic Dynamics } (\mathcal{T}) \\
\hline
\longrightarrow\; &\textbf{Distribution Over Future States } (S_{t+1:t+H})
\end{aligned}$$

```mermaid
flowchart LR
    accTitle: Enterprise World Model Engine Dataflow Pipeline
    accDescr: End-to-end dataflow showing state observations and rules initializing the world state, feeding through the dynamics engine with actions and exogenous events to project future state distributions.
    OBS[/"Observations"/] --> STATE["World State S_t"]
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

---

## The Four Fundamental Questions

EWM Engine strictly differentiates four distinct inquiries:

| Question | Mathematical Form | System Goal |
|---|---|---|
| **Prediction** | $P(Y \mid X)$ | What is likely to happen next based on historical correlations? |
| **Simulation** | $P(S_{1:T} \mid S_0, \theta)$ | How could this system evolve forward in time? |
| **Intervention** | $P(Y \mid \text{do}(X))$ | What could happen if we deliberately alter policy or action $X$? |
| **Decision Evaluation** | $\mathbb{E}[U(S) \mid \text{do}(X)] \text{ vs } \mathbb{E}[U(S) \mid \text{do}(Y)]$ | How do the systemic consequences of alternative actions compare? |

EWM Engine is primarily architected around **Simulation, Intervention, and Decision Evaluation**. It never conflates observational predictive associations with interventional causal identification.

---

## Architectural North Star

> *"EWM Engine is not another agent framework. It is the environment model agents can use to ask 'what happens if I do this?' before they act."*

> *"Prediction estimates what may happen. A world model lets us explore what may happen if the system is changed."*

> *"Learn what is uncertain. Encode what is known. Simulate the interaction between both."*

---

## What EWM Engine Is NOT

To preserve intellectual honesty and architectural focus, EWM Engine is **not**:
- Another chatbot, RAG, or agent orchestration framework (not LangChain, AutoGen, CrewAI, or LangGraph).
- An RL-only discrete environment gym (though it supports policy rollouts).
- A digital-twin 3D visualization platform.
- A univariate statistical time-series forecasting library.
- A replacement for low-level discrete-event simulators (DES).
- A JEPA neural network implementation alone.
- A financial high-frequency trading simulator.

It is a **general-purpose integration architecture for executable socio-technical dynamics**.
