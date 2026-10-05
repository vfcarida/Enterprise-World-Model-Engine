# ADR-018: Solver and Planner Protocols, Bound Enforcement, and the Engine-Agent Separation

## Status
Accepted

## Date
2026-10-05

## Context
EWM Engine provides ecosystem integrations in `ewm_engine.integrations` bridging external agent frameworks (LangGraph, AutoGen), formal verification engines (Z3 SMT), mathematical programming solvers (Google OR-Tools, SciPy), and reinforcement learning environments (Gymnasium).

In `v1.0.0`, these adapters existed as thin **Alpha** components with three critical architectural shortcomings:
1. **Unbounded Execution Risk**: Solvers and planners could theoretically execute without time limits. On NP-hard combinatorial or SMT instances, an unbounded solver can hang the host simulation indefinitely.
2. **Missing Protocol Formalism**: While the engine possessed clean `Actor` and `Constraint` protocols, there were no standardized protocols for mathematical constraint verification or operations-research decision planners.
3. **Engine-Agent Conflation in Evaluation**: AI agent benchmarks often rely on LLM judges to evaluate agent outputs, introducing high subjectivity and hallucination (Gaia2/ARE studies demonstrated oracle-graph state verifiers achieve 0.98 agreement vs. 0.72 for LLM judges; arXiv:2509.17158). The engine must establish a clear boundary: **the engine provides a deterministic, verifiable world; agents are external policies evaluated against state-based systemic outcomes**.

---

## Decision

### 1. Formalize `ConstraintSolver` and `ActionPlanner` Protocols
We formalize two public protocols in `ewm_engine.integrations.protocols`:

```python
@runtime_checkable
class ConstraintSolver(Protocol):
    def check(
        self,
        *,
        state: WorldState,
        actions: Sequence[Action] = (),
        time_limit_seconds: float | None = None,
    ) -> SolverResult: ...


@runtime_checkable
class ActionPlanner(Protocol):
    def propose(
        self,
        *,
        state: WorldState,
        objective: str | dict[str, Any] | None = None,
        time_limit_seconds: float | None = None,
    ) -> Sequence[Action]: ...
```

- **`SolverResult`**: Returns a standardized, immutable payload containing `status` (`OPTIMAL`, `FEASIBLE`, `INFEASIBLE`, `UNKNOWN`), `satisfied: bool`, `solve_time_seconds: float`, `timed_out: bool`, and optional `unsat_core: tuple[str, ...]` for formal auditability.
- **Architectural Boundary**: Planners and solvers **never** mutate state directly or silently repair invalid states. Planners synthesize proposed actions; solvers verify feasibility. The engine's core constraint registry remains the sole authority that accepts or rejects actions.

### 2. Mandatory Time and Resource Limits
Every solver and planner adapter MUST enforce a finite time and resource limit:
- **Z3 SMT (`Z3ConstraintAdapter`)**: Enforces `timeout_ms` on the underlying `z3.Solver` instance. If the timeout expires, the adapter returns `SolverStatus.UNKNOWN` with `timed_out=True` rather than hanging.
- **Google OR-Tools (`ORToolsAllocationAdapter`, `CPSATAllocationPlanner`)**: Enforces `set_time_limit` (Linear Solver) or `max_time_in_seconds` (CP-SAT).
- **SciPy (`SciPyAllocationPlanner`)**: Enforces `time_limit` via optimizer options and bounds iteration counts.

### 3. OR Planners: CP-SAT & SciPy Continuous
We graduate Operations Research integration by introducing two production-grade planners:
- **`CPSATAllocationPlanner`**: Discrete, constraint-programming allocation planner supporting exact integer capacities, multiple source depots, priority weights, and unsat core extraction.
- **`SciPyAllocationPlanner`**: Continuous linear programming planner for high-throughput resource redistribution.

Both planners implement `ActionPlanner` and `Actor` (`act(state, context) -> Sequence[Action]`).

### 4. Verifiable, State-Based Agent Evaluation ("Engine ≠ Agent")
To demonstrate rigorous agent evaluation without LLM judges:
- We ship an agent evaluation example (`examples/agent_evaluation/`) comparing external agents across identical stochastic counterfactual scenarios.
- Scoring is strictly computed from verifiable world state trajectories (e.g. cumulative unmet demand, resource equity/Gini index, hard constraint violation rates).
- When solvers reject proposed allocations, they surface the formal unsat core / infeasibility certificate, providing complete auditability.

### 5. Component Graduation: Alpha → Beta
We formally graduate `integrations` adapters from **Alpha** to **Beta**:
- `CallableActorAdapter`: Beta
- `Z3ConstraintAdapter`: Beta
- `ORToolsAllocationAdapter`: Beta
- `CPSATAllocationPlanner`: Beta
- `SciPyAllocationPlanner`: Beta
- `EnterpriseGymEnv`: Beta

All adapter documentation reflects this maturity status with runnable, tested code examples.

---

## Consequences

### Positive
- Production safety: No solver or planner can deadlock simulations due to enforced timeouts.
- Clean architectural contracts: Planners propose, constraints validate, engine executes.
- Epistemic rigor: Agent evaluation relies on verifiable state histories rather than subjective LLM evaluation.
- Full backwards compatibility with existing `v1.0.0` APIs.

### Negative / Trade-offs
- Solvers reaching tight time limits will return `UNKNOWN` / empty proposals rather than continuing search, requiring caller awareness.
