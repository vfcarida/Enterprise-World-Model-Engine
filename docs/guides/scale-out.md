# Scaling Out Monte Carlo Simulations

This guide covers parallel and distributed execution of Monte Carlo rollouts in the Enterprise World Model Engine.

---

## 1. Overview & Determinism Guarantee

Monte Carlo rollout generation is embarrassingly parallel: each sample trajectory evolves independently from its initial state under the scenario's horizon.

However, scientific reproducibility requires that:
$$\text{result}_{\text{distributed}}.\text{logical} \equiv \text{result}_{\text{serial}}.\text{logical}$$

The EWM Engine guarantees **bitwise deterministic equivalence** across all execution backends:
- Per-rollout child seeds are generated on the orchestrator via `np.random.SeedSequence(scenario.seed).spawn(scenario.samples)`.
- Each worker receives its assigned immutable child seed and operates in complete isolation.
- Rollouts are reassembled in strict ascending `sample_id` order regardless of asynchronous completion order.
- Trajectory statuses, state fingerprints, metrics, and systemic traces are identical to the serial reference execution.

> [!NOTE]
> **Determinism Scope**: Bitwise determinism holds for all built-in dynamics, constraints, and standard NumPy distributions. External neural/GPU adapters (e.g., PyTorch CUDA atomic additions) are outside this bitwise guarantee due to hardware-level floating-point non-determinism.

---

## 2. Pluggable Executor Backends

The engine provides three execution backends through the `RolloutExecutor` protocol:

| Executor Backend | Module | Dependencies | Best Used For |
| :--- | :--- | :--- | :--- |
| `SerialExecutor` | stdlib | None (built-in) | Unit tests, small scenarios ($N \le 50$), debugging, real-time per-step hooks |
| `MultiprocessingExecutor` | stdlib | `concurrent.futures` (stdlib) | Multi-core workstation / laptop execution ($N \ge 100$) |
| `RayExecutor` | extra | `ray>=2.9.0` (`[distributed]`) | Large-scale multi-core or multi-node Ray clusters ($N \ge 1{,}000$) |

---

## 3. Quickstart & Usage

### Method 1: String Shorthand in `run(...)`

The simplest way to run in parallel is passing `executor="multiprocessing"` or `executor="ray"` to `SimulationEngine.run()`:

```python
from ewm_engine.core import World, WorldState
from ewm_engine.simulation import Scenario, SimulationEngine

world = World(initial_state=WorldState())
scenario = Scenario(scenario_id="capacity_stress", horizon=100, samples=500, seed=42)

engine = SimulationEngine()

# Run across all available local CPU cores
result = engine.run(world, scenario, executor="multiprocessing")

print(
    f"Executed {len(result.trajectories)} rollouts in {result.run_metrics.simulation_duration_seconds:.2f}s"
)
```

### Method 2: Configured Executor Instances

For fine-grained control over concurrency or cluster connection parameters, pass a configured executor instance:

```python
from ewm_engine.simulation import MultiprocessingExecutor, SimulationEngine

# Restrict to 4 worker processes
executor = MultiprocessingExecutor(max_workers=4, chunksize=2)
engine = SimulationEngine(executor=executor)

result = engine.run(world, scenario)
```

---

## 4. Scaling with Ray (`RayExecutor`)

### Installation

Install EWM Engine with the `distributed` extra:

```bash
pip install "ewm-engine[distributed]"
```

### Local Multi-Core Ray Execution

```python
from ewm_engine.simulation import RayExecutor, SimulationEngine

# Initialize RayExecutor with local CPU limit
ray_executor = RayExecutor(num_cpus=8)

engine = SimulationEngine(executor=ray_executor)
result = engine.run(world, scenario)
```

### Connecting to an Existing Ray Cluster

In production or HPC cloud environments (Kubernetes, AWS, GCP, Slurm), connect to the active Ray cluster by setting `address="auto"` or passing the Ray client URI:

```python
from ewm_engine.simulation import RayExecutor, SimulationEngine

# Connect to the remote Ray head node
cluster_executor = RayExecutor(address="ray://ray-head.cluster.local:10001")

engine = SimulationEngine(executor=cluster_executor)
result = engine.run(world, scenario)
```

### Shared Memory Efficiency (Plasma Object Store)

When dispatching rollouts via `RayExecutor`:
1. The orchestrator serializes `World` and `Scenario` **once** and deposits them into Ray's Plasma shared-memory object store via `ray.put()`.
2. Worker tasks on the same physical node access the world and scenario via zero-copy shared memory without redundant IPC serialization overhead.
3. Only the lightweight `sample_id` and `child_seed` are transmitted per task.

---

## 5. Observability and Lifecycle Hooks

In distributed mode (`MultiprocessingExecutor` and `RayExecutor`):
- `SimulationStarted` is emitted on the host orchestrator before workers launch.
- Workers evaluate steps and constraints locally, recording metrics directly into the trajectory payload.
- Upon rollout completion and reassembly, the host orchestrator emits `RolloutCompleted` in deterministic `sample_id` order (0, 1, 2, ...).
- `SimulationFinished` is emitted on the orchestrator with the aggregated `SimulationResult` and `RunMetrics`.

This architecture prevents high-frequency IPC bottlenecks on observational events while preserving complete observability.

---

## 6. Performance Heuristics

| Sample Size ($N$) | Horizon ($H$) | Recommended Executor | Rationale |
| :--- | :--- | :--- | :--- |
| $N \le 20$ | Any | `SerialExecutor` | Process creation / IPC serialization exceeds computation time |
| $20 < N \le 1{,}000$ | $H \ge 30$ | `MultiprocessingExecutor` | Multi-core parallelism amortizes process pool overhead cleanly |
| $N > 1{,}000$ | $H \ge 50$ | `RayExecutor` | Distributed scheduling across node pools maximizes throughput |
