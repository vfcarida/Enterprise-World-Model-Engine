# ADR-017: Distributed Monte Carlo Execution and Determinism Preservation

## Status
Accepted

## Date
2026-10-05

## Context
Monte Carlo simulation of complex enterprise world models is embarrassingly parallel across sample trajectories. For large-scale scenario evaluation ($N \ge 1{,}000$ samples) or long horizons ($H \ge 365$ steps), sequential execution on a single core throttles decision-making throughput.

However, the core integrity of the Enterprise World Model Engine relies upon **Reproducibility Under Counterfactual Interventions (AC-004)**:
> *Same logical initial state + scenario specification + component versions + master seed produces the exact same built-in logical trajectory across platforms and runs.*

Naive parallelization typically breaks determinism through:
1. **Completion-Order Nondeterminism**: Workers finishing asynchronously cause trajectories to be assembled in arbitrary order, producing fluctuating scenario quantiles and non-reproducible downstream traces.
2. **Shared RNG Interference**: Concurrent threads or processes drawing from a shared random state induce race conditions that destroy trajectory alignment.
3. **Shared Mutable State**: Concurrent mutation of entity attributes or resource registers creates race conditions and cross-rollout state pollution.

We require a distributed execution architecture that scales across multi-core CPUs and Ray clusters while guaranteeing that parallel execution yields the **identical logical result** to serial execution.

---

## Decision

### 1. Pluggable Executor Architecture (`RolloutExecutor`)
We define a clean protocol `RolloutExecutor` in `ewm_engine.simulation.executors` with three concrete backends:
1. **`SerialExecutor` (Default)**: In-process sequential reference executor. Preserves existing behavior with zero overhead and immediate fine-grained hook dispatch.
2. **`MultiprocessingExecutor`**: Multi-core process pool using Python standard library `concurrent.futures.ProcessPoolExecutor`. Requires no external dependencies.
3. **`RayExecutor`**: Distributed cluster executor backed by Ray (`ray>=2.9.0`), packaged under the optional extra `distributed`.

`SimulationEngine` accepts an optional `executor: RolloutExecutor | str | None = None` parameter both in `__init__` and in `run(...)` / `simulate(...)`. When omitted, it defaults to `"serial"`.

### 2. Determinism Preservation & Seed Sequence Spawning
To guarantee identical random streams regardless of concurrency:
- The orchestrator pre-spawns independent, non-overlapping child seeds via NumPy's PCG64 seed sequence:
  $$\text{child\_seeds} = \text{SeedSequence}(\text{scenario.seed}).\text{spawn}(\text{scenario.samples})$$
- Each worker executing rollout $i \in [0, N-1]$ receives the immutable child seed $\text{child\_seeds}[i]$ and initializes its own isolated generator $\text{default\_rng}(\text{child\_seed})$.
- Regardless of worker completion order, results are reassembled and sorted by `sample_id` in strict ascending order before populating `SimulationResult.trajectories`.

### 3. Determinism Scope & Explicit Epistemic Boundary
- **In-Scope (Bitwise Invariant)**: All built-in dynamics (deterministic, stochastic, composite), constraint evaluations, exogenous event sampling, actor decisions, and state fingerprints are strictly bitwise identical across `SerialExecutor`, `MultiprocessingExecutor`, and `RayExecutor`:
  $$\text{result}_{\text{distributed}}.\text{logical} \equiv \text{result}_{\text{serial}}.\text{logical}$$
- **Out-of-Scope (Explicit Caveat)**: External neural/GPU/CUDA adapters that perform non-deterministic floating-point reductions across heterogeneous hardware architectures are explicitly outside the bitwise determinism guarantee.

### 4. Zero Shared Mutable State
- `World` and `Scenario` are immutable Pydantic models (`frozen=True`).
- In `MultiprocessingExecutor`, process boundaries ensure complete memory isolation with copy-on-write / serialization semantics.
- In `RayExecutor`, `world` and `scenario` are placed into Ray's Plasma shared-memory object store once via `ray.put(...)` as read-only references, achieving zero-copy deserialization across worker processes on the same node and safe network transmission across cluster nodes.

### 5. Failure Semantics
- **Domain Invalidation**: Fatal hard constraint violations do not raise exceptions; they terminate the rollout deterministically with `TrajectoryStatus.INVALID` and preserve all partial steps in the trajectory.
- **System / Worker Faults**: If a worker encounters an unhandled runtime error (e.g., node eviction, memory exhaustion, or code bug), the executor fails fast and propagates the exception to the orchestrator with full stack trace, preventing silent partial or corrupted simulation results.

### 6. Observability and Lifecycle Hooks in Distributed Mode
- In `SerialExecutor`, fine-grained per-step events (`StepCompleted`, `ConstraintEvaluated`) are emitted synchronously as steps execute.
- In `MultiprocessingExecutor` and `RayExecutor`, workers record step records directly into the trajectory and aggregate rollout statistics (`constraint_evaluations`, `hard_violations`, `soft_violations`, `dynamics_transitions`). The host orchestrator emits `RolloutCompleted` in deterministic `sample_id` order upon reassembly, followed by `SimulationFinished`. This prevents high-frequency IPC serialization bottlenecks on observational events while preserving complete observability.

### 7. Why Default Remains Serial
- For small sample counts ($N \le 50$, $H \le 30$), process spawning and IPC overhead dominate execution time.
- The serial engine has zero system dependencies and runs out-of-the-box in restricted, lightweight, or embedded environments.
- Retaining serial execution as default ensures 100% backward compatibility for existing callers.

### 8. Strict Dependency Isolation
- Core engine modules never import `ray` or `multiprocessing` at module load time.
- `ray` is imported lazily inside `RayExecutor.execute()`. If `ray` is not installed, attempting to execute with `RayExecutor` raises a descriptive `ImportError` directing the user to `pip install ewm-engine[distributed]`.

---

## Consequences

### Positive
- Near-linear throughput speedup across CPU cores and multi-node Ray clusters for compute-intensive Monte Carlo rollouts.
- Zero sacrifice of the core determinism guarantee: `distributed_result.logical == serial_result.logical`.
- No new required dependencies added to core `ewm-engine`.
- 100% backward compatibility with existing tests and client code.

### Negative / Trade-offs
- Process serialization overhead for tiny workloads ($N < 10$), which is why `serial` remains the default.
- Per-step lifecycle hooks are dispatched per-rollout in distributed mode rather than per-step to avoid IPC thrashing.
