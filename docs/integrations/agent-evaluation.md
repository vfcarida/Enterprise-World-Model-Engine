# Agent Evaluation & The "Engine != Agent" Architecture

> [!NOTE]
> **Maturity**: Beta. Demonstrates verifiable, state-based evaluation of external agents and Operations Research auditability. Based on the ARE / Gaia2 benchmark methodology ([arXiv:2509.17158](https://arxiv.org/abs/2509.17158)).

---

## The "Engine != Agent" Thesis

In enterprise AI and decision systems, agent orchestrators (such as LangGraph, AutoGen, CrewAI, or RL policies) are frequently conflated with the environments they act upon. This architectural error leads to hallucinated action outcomes, unverified constraint satisfaction, and uncalibrated performance claims.

EWM Engine enforces a strict boundary:

```
+-------------------------------------------------------------+
|                     External Agent                          |
| (LangGraph / AutoGen / Heuristics / OR-Tools / SciPy / LLM) |
+-------------------------------------------------------------+
                              |
                     Candidate Actions (Propose)
                              v
+-------------------------------------------------------------+
|                      EWM Engine                             |
|  - Pre-Action Constraints (Physical corridor limits, funds) |
|  - Conservation Dynamics (Material balance, patient flow)   |
|  - Post-Transition Constraints (State validity invariants)  |
|  - Systemic Provenance Trace (Exact causal DAG)             |
+-------------------------------------------------------------+
                              |
                   Evolved WorldState S_{t+1}
                              v
+-------------------------------------------------------------+
|           Verifiable State Oracle (ARE / Gaia2)             |
|  - Invariant Verification (0 violations, valid stocks)      |
|  - Objective Performance (Unserved demand, service level)   |
|  - Systemic Equity (Regional disparity / Gini)              |
+-------------------------------------------------------------+
```

1. **Agents Propose, Engines Validate**: An agent cannot mutate state directly or bypass physics. If an agent proposes a transfer that exceeds physical corridor limits, the engine's constraint pipeline **rejects the action** and records a structured violation.
2. **Deterministic State-Based Scoring**: Evaluation scores are derived from verifiable, structured world state transitions ($S_0 \to S_1 \dots \to S_T$). In the ARE benchmark (arXiv:2509.17158), oracle-graph environment verifiers reached **0.98 agreement** compared to only **0.72 agreement** for subjective LLM-as-a-judge scorers.
3. **No Subjective Overclaim**: If an agent outputs free-form textual rationale, that text is judged solely as text. Systemic outcomes are scored exclusively on the verified state trajectory.

---

## Runnable Example: Hospital Supply Chain Evaluation

The complete executable example is provided in [`examples/agent_evaluation/`](file:///examples/agent_evaluation/):

```python
from examples.agent_evaluation.run import run_agent_evaluation

# Run end-to-end evaluation across 4 scenarios
results = run_agent_evaluation()

# 1. Systemic comparison table
print(results["comparison"].summary_table())

# 2. ARE Oracle-Graph Verifiable State Scores
for agent, scores in results["oracle_scores"].items():
    print(
        f"{agent}: Invariants={scores['invariants_preserved']} "
        f"ServiceLevel={scores['service_level'] * 100:.1f}% "
        f"CompositeScore={scores['composite_verifiable_score']:.3f}"
    )

# 3. Operations Research Auditability (Infeasibility Certificate)
print("Unsat Core:", results["unsat_core"])
```

### Typical Results Summary

| Agent | Invariants | Violations | Service Level | Disparity | Composite Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Status Quo (No Transfers)** | PASS | 0 | 15.0% | 20.0 | 0.109 |
| **Greedy Myopic Agent** | **FAIL** | 3 | 15.0% | 20.0 | **0.000** |
| **Proportional Heuristic** | PASS | 0 | 75.0% | 20.0 | 0.469 |
| **CP-SAT Optimization Agent**| PASS | 0 | **81.7%** | 45.0 | **0.499** |

---

## Operations Research Auditability: Unsat Core Extraction

When a crisis shock creates an impossible allocation problem (e.g. medical demand exceeds depot supplies and corridor throughput), [`CPSATAllocationPlanner`](or-tools.md) does not return a random guess or an ambiguous error.

Instead, the CP-SAT solver extracts the minimal **unsatisfiable core (infeasibility certificate)**:

```python
# Surfaced Unsat Core Assumptions:
# - supply_capacity_hub_stock_100
# - demand_target_hospital_north_250
# - demand_target_hospital_central_200
```

This mathematical certificate provides an unforgeable, auditable proof of why the requested allocation was physically infeasible, providing complete transparency for high-stakes enterprise governance.
