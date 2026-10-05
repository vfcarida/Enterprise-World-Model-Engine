# Ecosystem Integrations & Adapters

> [!NOTE]
> **Maturity**: Beta. All ecosystem adapters in `ewm_engine.integrations` adhere to the v1.1.0 Beta contract. They have dedicated integration test coverage, enforced finite resource limits, and provide clean isolation from the zero-dependency simulation core.

EWM Engine follows a **hexagonal / ports-and-adapters architecture**:

1. **Zero-Dependency Core**: The core simulation kernel has zero mandatory dependencies on external machine learning frameworks, LLM APIs, GPU runtimes, or mathematical solvers.
2. **Adapter Boundaries**: External capabilities interface through clean, public protocol boundaries:
   - **Operations Research**: [`ORToolsAllocationAdapter` and `CPSATAllocationPlanner`](or-tools.md) (Beta) synthesize optimal discrete/continuous flows and surface unsat cores.
   - **Continuous Optimization**: [`SciPyAllocationPlanner`](scipy.md) (Beta) solves continuous linear programs using the HiGHS solver backend.
   - **Formal SMT Solvers**: [`Z3ConstraintAdapter`](z3-smt.md) (Beta) enables formal symbolic constraint satisfaction, bounded reachability, and unsat core proofs.
   - **Reinforcement Learning**: [`EnterpriseGymEnv`](gymnasium.md) (Beta) adapts an EWM `World` into standard Gymnasium (`gymnasium.Env`) environments for policy training.
   - **Agent Orchestrators**: [`CallableActorAdapter`](agent-evaluation.md) (Beta) allows arbitrary agent functions (LangGraph, AutoGen, CrewAI, heuristics) to act as decision actors.
   - **Observability**: [`OpenTelemetryHook`](../guides/observability-otel.md) (Beta) exposes lifecycle hooks and simulation metrics to OpenTelemetry collectors.

---

## Formal Protocols: `ConstraintSolver` and `ActionPlanner`

All mathematical solvers and optimization planners implement one of two formalized public protocols:

### 1. `ConstraintSolver` Protocol

```python
from collections.abc import Sequence
from typing import Protocol, runtime_checkable
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState
from ewm_engine.integrations.protocols import SolverResult


@runtime_checkable
class ConstraintSolver(Protocol):
    def check(
        self,
        *,
        state: WorldState,
        actions: Sequence[Action] = (),
        time_limit_seconds: float | None = None,
    ) -> SolverResult:
        """Verify state and action invariants, returning structured SolverResult."""
        ...
```

### 2. `ActionPlanner` Protocol

```python
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable
from ewm_engine.core.actions import Action
from ewm_engine.core.state import WorldState


@runtime_checkable
class ActionPlanner(Protocol):
    def propose(
        self,
        *,
        state: WorldState,
        objective: str | dict[str, Any] | None = None,
        time_limit_seconds: float | None = None,
    ) -> Sequence[Action]:
        """Propose candidate actions to optimize the specified objective."""
        ...
```

### Mandatory Finite Resource Limits

> [!IMPORTANT]
> **No Unbounded Solvers**: Every solver and planner adapter in EWM Engine accepts and enforces a finite time limit (default: 5.0 seconds). A solver that runs unbounded is treated as an architectural bug. When a limit is reached, the adapter returns a structured `SolverResult` with `timed_out=True` and `status=SolverStatus.UNKNOWN` rather than hanging or crashing.

---

## Installation of Optional Extras

Ecosystem adapters are packaged as modular extras:

```bash
# Operations Research planners (Google OR-Tools CP-SAT and SciPy HiGHS)
pip install "ewm-engine[or]"

# Formal SMT verification (Z3 Theorem Prover) and OR solvers
pip install "ewm-engine[solvers]"

# Reinforcement learning (Gymnasium and PyTorch)
pip install "ewm-engine[rl]"

# OpenTelemetry instrumentation
pip install "ewm-engine[otel]"

# Distributed Monte Carlo (Ray)
pip install "ewm-engine[distributed]"

# All ecosystem dependencies
pip install "ewm-engine[all]"
```
