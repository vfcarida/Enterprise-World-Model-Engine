# API Change Proposal (ACP-003): Distributed Monte Carlo Execution & Determinism Preservation

- **Target Release:** v1.1.0
- **Author:** Systems Engineer for Reproducible Scientific Computing
- **Status:** Implemented
- **Impact Level:** Non-breaking additive enhancement (Pluggable execution model)
- **Related Spec:** `docs/specs/features/FEAT-002-distributed-monte-carlo.md`, `docs/specs/spec-driven-development.md`
- **Related ADR:** `docs/adr/ADR-017-distributed-monte-carlo-execution-and-determinism.md`
- **Acceptance Criteria:** AC-025 (Distributed Determinism), AC-004 (Reproducibility), AC-024 (Contract Compatibility)

---

## 1. Summary

This proposal enhances `ewm_engine.simulation` to support multi-core parallel and distributed execution of Monte Carlo rollouts without weakening the engine's core reproducibility guarantee:
> *Same master seed + scenario specification + initial state $\implies$ identical logical trajectory and state hashes, regardless of serial or parallel execution mode.*

The execution model is abstracted behind a pluggable `RolloutExecutor` protocol with three concrete backends:
1. `SerialExecutor` (default, in-process sequential).
2. `MultiprocessingExecutor` (standard library `ProcessPoolExecutor` with deterministic worker seed allocation).
3. `RayExecutor` (distributed cluster execution via `ray`, packaged under the `distributed` extra).

---

## 2. Motivation & Use Case

In enterprise simulation experiments, scenario evaluations frequently require $N \ge 1{,}000$ sample rollouts across $H \ge 90$ discrete steps. Serial execution on a single core throttles experimentation cycles.

Parallelizing rollouts across cores or cluster nodes is embarrassingly parallel, but naive approaches introduce completion-order nondeterminism or race conditions across shared RNG states. This proposal establishes:
- Deterministic sub-seed derivation via `SeedSequence(scenario.seed).spawn(samples)`.
- Worker isolation with zero shared mutable state.
- Strict ordinal reassembly of trajectories (sorted by `sample_id` ascending), guaranteeing identical summary statistics and quantiles.

---

## 3. Proposed API Surface Diff

```python
# Before (v1.0.0)
class SimulationEngine:
    def __init__(self, hooks: HookRegistry | None = None) -> None: ...

    def run(
        self,
        world: World,
        scenario: Scenario,
        actors: Sequence[Actor] | None = None,
    ) -> SimulationResult: ...


# After (v1.1.0 - Additive optional executor parameters with backwards-compatible defaults)
class SimulationEngine:
    def __init__(
        self,
        hooks: HookRegistry | None = None,
        executor: RolloutExecutor | str | None = None,
    ) -> None: ...

    def run(
        self,
        world: World,
        scenario: Scenario,
        actors: Sequence[Actor] | None = None,
        executor: RolloutExecutor | str | None = None,
    ) -> SimulationResult: ...
```

### Affected Symbols in `ewm_engine.__all__`
- [x] **No change to root `ewm_engine.__all__`**: Public surface remains strictly the 22 canonical Stable symbols + `__version__`.
- New executor protocol and implementations are exported cleanly from `ewm_engine.simulation.executors`:
  - `RolloutExecutor` (Protocol)
  - `SerialExecutor`
  - `MultiprocessingExecutor`
  - `RayExecutor`
  - `get_executor`

---

## 4. Backwards Compatibility & SemVer Analysis

1. **Parameter Backwards Compatibility**: The `executor` parameter in both `SimulationEngine.__init__` and `SimulationEngine.run` is keyword-optional with default `None` (which resolves to `"serial"`). Existing code executes identically without modification.
2. **Deterministic Result Equivalence**: For any built-in scenario and dynamics model, `run(world, scenario, executor="multiprocessing")` produces trajectory results that match `run(world, scenario, executor="serial")` with 100% bitwise equality on state hashes, resource values, and constraint evaluations.
3. **Dependency Isolation**: The core engine depends solely on Python's standard library `concurrent.futures`. Distributed Ray execution requires the optional `distributed` extra (`pip install 'ewm-engine[distributed]'`).

---

## 5. Verification & Acceptance Gate

- `tests/property/test_distributed_determinism.py` verifies bitwise trajectory equality between serial and multiprocessing execution across stochastic scenarios.
- `tests/contract/test_api_compatibility.py` verifies that `SimulationEngine.run` has no new required parameters (AC-024).
