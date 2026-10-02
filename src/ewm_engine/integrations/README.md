# Ecosystem Integrations & Adapters

This package defines architectural extension points and adapters for integrating external technologies with the **Enterprise World Model Engine (EWM Engine)**.

## Architectural Boundary

To preserve minimal core dependencies, scientific neutrality, and deterministic reproducibility:
1. **Core Zero-Dependency**: The core simulation engine never depends on LLM APIs, cloud services, commercial solvers, or GPU frameworks.
2. **Adapter Pattern**: External tools interface via protocol boundaries:
   - **Agent Orchestrators** (LangGraph, AutoGen, CrewAI, RLlib) connect via the `Actor` protocol:
     ```python
     class ExternalAgentAdapter:
         def act(self, state: WorldState, context: ActorContext) -> Sequence[Action]: ...
     ```
   - **Formal SMT / SAT Solvers** (Z3, CVC5) connect via the `Constraint` protocol:
     ```python
     class Z3FormalConstraint:
         def evaluate(self, state: WorldState, action: Action | None = None) -> ConstraintResult: ...
     ```
   - **Operations Research Optimizers** (Google OR-Tools, SciPy Linear Programming) connect as policy optimizers or interventional planners:
     ```python
     class ORToolsDispatcher:
         def optimize(self, state: WorldState) -> Sequence[Action]: ...
     ```

## Planned Adapters
- `ewm_engine.integrations.langgraph`: Stateful actor adapter translating LangGraph agent thoughts into typed `Action` proposals.
- `ewm_engine.integrations.autogen`: Multi-agent conversation adapter allowing AutoGen group chats to query the world model counterfactually before finalizing operational decisions.
- `ewm_engine.integrations.z3`: SMT symbolic verification of state transitions and invariant reachability.
- `ewm_engine.integrations.ortools`: Mixed-integer linear programming (MILP) flow optimizer for supply chain and logistics dispatch.
