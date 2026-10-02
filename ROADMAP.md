# Enterprise World Model Engine Roadmap

This document outlines the phased development roadmap for the **Enterprise World Model Engine (EWM Engine)**.

## Maturity Framework

To preserve scientific and engineering integrity, capabilities in EWM Engine are classified under explicit maturity levels:

- **Stable**: Fully tested, validated, deterministic, and covered by backward compatibility guarantees.
- **Alpha**: Core interfaces defined and working; subject to API refinement based on community usage.
- **Experimental**: Active proof-of-concept; interfaces may change significantly without deprecation cycles.
- **Research / Exploration**: Formal hypotheses under theoretical or empirical investigation.
- **Planned Adapter**: Planned integration with external ecosystems.

---

## Current Status (v0.1.0 - Foundation)

| Subsystem | Maturity Level | Notes |
|---|---|---|
| Core Simulation Engine | **Alpha** | Deterministic Monte Carlo rollout, scenario seeding, trajectory tracking |
| World & State Model | **Alpha** | Immutable snapshot states, entities, resources, relationships |
| Constraint Engine | **Alpha** | Pre-action and post-state verification with detailed violation provenance |
| Pluggable Dynamics | **Alpha** | Deterministic, stochastic, and composite dynamics interfaces |
| Scenario Branching | **Alpha** | Counterfactual branching from identical initial states |
| Systemic Traces | **Alpha** | Directed dependency tracking with epistemic level tagging |
| Scenario Comparison | **Alpha** | Quantile distributions, violation rates, delta summaries |
| Reference Examples | **Alpha** | Minimal Supply World & CivicFlow Disaster Response |
| Learned Dynamics Protocol | **Experimental** | Protocol definitions and minimal neural baseline integration |
| Causal Identification | **Research** | Explicit epistemics separating prediction from interventional effects |
| Agent Orchestrators | **Planned Adapter** | External adapters for LangGraph, AutoGen, CrewAI, and RLlib |

---

## Near-Term Roadmap (v0.2.0 - v0.4.0)

### 1. Extended Constraints & Solvers
- **SMT / SAT Solver Adapters**: Optional integration with Z3/CVC5 for formal constraint verification and unreachable state detection.
- **Operations Research Integrations**: Optional adapters for Google OR-Tools and mixed-integer linear programming (MILP).

### 2. Receding Horizon & Online Simulation (MPC)
- Continuous state re-grounding:
  $$\text{Observe} \to \text{Simulate Short Horizon} \to \text{Select Action} \to \text{Re-ground} \to \text{Repeat}$$
- Support for state estimation filtering under noisy or partially observed organizational telemetry.

### 3. Declarative World Specification
- Validated YAML/JSON schemas for declarative world definition, parsed strictly via typed models without code execution.

---

## Long-Term Research Agenda (v0.5.0+)

1. **Latent Dynamics & World Representations**:
   - Recurrent State-Space Models (RSSM / Dreamer-like latent rollouts).
   - Joint Embedding Predictive Architectures (JEPA) for multi-scale organizational dynamics.
   - Graph Neural Networks (GNNs) for heterogeneous organizational relational graphs.
2. **Causal Epistemics & Discovery**:
   - Causal discovery over observational enterprise logs.
   - Off-policy counterfactual evaluation with overlap and unconfoundedness diagnostics.
3. **Out-of-Distribution (OOD) & Regime Shifts**:
   - Automated detection when simulation trajectories enter ungrounded or epistemically unsupported state regimes.
4. **Scalable Distributed Rollouts**:
   - Ray / multi-node parallel Monte Carlo simulation for high-throughput scenario analysis.
