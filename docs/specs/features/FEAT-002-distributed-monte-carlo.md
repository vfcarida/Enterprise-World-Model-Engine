# Feature Specification: FEAT-002 — Distributed Monte Carlo with Determinism Preserved

**Feature ID:** `FEAT-002`  
**Milestone:** `v1.1.0` (Prompt `P03`)  
**Status:** Approved  
**Related ADR:** [`docs/adr/ADR-017-distributed-monte-carlo-execution-and-determinism.md`](../../adr/ADR-017-distributed-monte-carlo-execution-and-determinism.md)  
**Governing Specification:** [`docs/specs/spec-driven-development.md`](../spec-driven-development.md)  

---

## 1. Problem Statement

Monte Carlo rollout generation in EWM Engine is an embarrassingly parallel computation. In `v1.0.0`, all sample trajectories executed serially in-process on a single core. While sufficient for small models and test suites, enterprise-scale evaluations ($N \ge 1{,}000$ samples, $H \ge 365$ steps) require substantial wall-clock time.

Naive parallelization across multiple cores or cluster nodes frequently compromises scientific reproducibility:
1. Asynchronous completion order causes trajectories to be gathered non-deterministically.
2. Shared pseudo-random number generator (RNG) state produces race conditions and nondeterministic branching.
3. Mutable shared state leads to memory corruption or cross-worker side-effects.

To enable scale-out while preserving the engine's core promise—**same seed $\implies$ same logical trajectory**—we must provide a pluggable execution layer that guarantees bitwise determinism between serial and distributed backends.

---

## 2. Requirements & Acceptance Criteria

### AC-P03-1: Determinism Equivalence Invariant (Critical)
- Parallel execution using `MultiprocessingExecutor` and `RayExecutor` must yield the **identical logical result** to `SerialExecutor` for any fixed world, scenario, and seed:
  - Identical trajectory statuses (`COMPLETED`, `INVALID`).
  - Identical trajectory seeds and sample IDs.
  - Identical initial, intermediate, and final state fingerprints (`WorldState.fingerprint`, `StepRecord.state_hash`).
  - Identical step metrics and constraint violation lists.
  - Identical systemic trace nodes and causal edges.
  - Identical aggregate counts in `RunMetrics` (completed rollouts, invalid rollouts, total steps, dynamics transitions, constraint evaluations, hard/soft violations).

### AC-P03-2: Deterministic RNG Spawning & Reassembly
- Child seeds must be pre-spawned via `np.random.SeedSequence(scenario.seed).spawn(scenario.samples)` on the orchestrator.
- Each worker executing rollout $i$ must receive and consume the exact child seed for index $i$.
- Workers' completed trajectories must be reassembled in strict ascending `sample_id` order regardless of the order in which workers finish.

### AC-P03-3: Pluggable Executor Abstraction
- Provide a `RolloutExecutor` protocol in `ewm_engine.simulation.executors` defining:
  `execute(world: World, scenario: Scenario, child_seeds: Sequence[SeedSequence], ...) -> list[RolloutResult]`
- Implement three concrete backends:
  1. `SerialExecutor`: Sequential in-process execution (default reference semantics).
  2. `MultiprocessingExecutor`: Multi-core process pool using stdlib `concurrent.futures.ProcessPoolExecutor`.
  3. `RayExecutor`: Distributed actor/task pool using `ray>=2.9.0`.
- `SimulationEngine.__init__` and `SimulationEngine.run(..., executor=...)` accept an executor instance or backend string (`"serial"`, `"multiprocessing"`, `"ray"`).
- Default behavior remains `"serial"` with zero behavior change for existing callers.

### AC-P03-4: Dependency Hygiene & Zero Import-Time Pollution
- Neither `multiprocessing` worker pools nor `ray` may be initialized at module import time.
- `ray` must be an optional dependency packaged under `[project.optional-dependencies] distributed = ["ray>=2.9.0"]` and included in `all`.
- Absence of `ray` must not raise `ImportError` when importing `ewm_engine` or `ewm_engine.simulation`.

### AC-P03-5: Safe Serialization & Immutability
- Worker payloads must consist exclusively of the project's own frozen Pydantic models (`World`, `Scenario`, `SeedSequence`, `RolloutResult`).
- Untrusted arbitrary code execution or unpicklable state is strictly prohibited.

---

## 3. Data Contracts & Architecture

```mermaid
flowchart TD
    Scenario[Scenario: seed, samples, horizon] --> Orchestrator[SimulationEngine.run]
    Orchestrator --> SeedSeq[SeedSequence.spawn(samples)]
    Orchestrator --> Dispatcher{Selected Executor}

    Dispatcher -->|serial| SExec[SerialExecutor]
    Dispatcher -->|multiprocessing| MExec[MultiprocessingExecutor: ProcessPool]
    Dispatcher -->|ray| RExec[RayExecutor: Plasma ObjectStore + Ray Tasks]

    SExec --> W1[Worker 0: child_seed_0]
    SExec --> W2[Worker 1: child_seed_1]
    MExec --> W1
    MExec --> W2
    RExec --> W1
    RExec --> W2

    W1 --> R1[RolloutResult: sample_id=0]
    W2 --> R2[RolloutResult: sample_id=1]

    R1 --> Reassemble[Deterministic Reassembly: sort by sample_id]
    R2 --> Reassemble
    Reassemble --> Result[SimulationResult: Bitwise Identical to Serial]
```

### Protocol & Models

```python
@dataclass(frozen=True)
class RolloutResult:
    sample_id: int
    trajectory: Trajectory
    constraint_evaluations: int
    hard_violations: int
    soft_violations: int
    dynamics_transitions: int


class RolloutExecutor(Protocol):
    def execute(
        self,
        world: World,
        scenario: Scenario,
        child_seeds: Sequence[np.random.SeedSequence],
        *,
        on_constraint_eval: Callable[[ConstraintResult, ConstraintPhase, int, int], None]
        | None = None,
        on_step_completed: Callable[[int, int, str, StepRecord, bool], None] | None = None,
    ) -> list[RolloutResult]: ...
```
