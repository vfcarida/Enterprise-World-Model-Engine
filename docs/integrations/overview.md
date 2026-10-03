# Ecosystem Integrations & Adapters

EWM Engine follows a **hexagonal / ports-and-adapters architecture**:

1. **Zero-Dependency Core**: The core simulation kernel has zero mandatory dependencies on external machine learning frameworks, LLM APIs, GPU runtimes, or mathematical solvers.
2. **Adapter Boundaries**: External capabilities interface through clean protocol boundaries:
   - **Reinforcement Learning**: [`EnterpriseGymEnv`](gymnasium.md) adapts an EWM `World` into standard Gymnasium (`gymnasium.Env`) environments for policy training.
   - **Operations Research**: [`ORToolsAllocationAdapter`](or-tools.md) uses Google OR-Tools to solve linear programming and resource flow allocation over world states.
   - **Formal SMT Solvers**: [`Z3ConstraintAdapter`](z3-smt.md) enables formal symbolic constraint satisfaction and reachability proofs using Z3.
   - **Agent Orchestrators**: [`CallableActorAdapter`](../api/reference.md) allows arbitrary agent functions (LangGraph, AutoGen, CrewAI) to act as decision actors in simulations.

---

## Installation of Optional Extras

Ecosystem adapters are packaged as optional dependencies:

```bash
# Reinforcement learning and PyTorch
pip install "ewm-engine[ml]"

# Mathematical and SMT solvers (Z3, OR-Tools)
pip install "ewm-engine[solvers]"

# Network graph visualization
pip install "ewm-engine[graphs]"

# All optional dependencies
pip install "ewm-engine[all]"
```
