# ADR-032: Deterministic Performance Gates, Long-Term History, and Numerical Quality Assurance

## Status
Accepted

## Context
A scientific simulation framework without regression gates suffers from an acute credibility gap. In the Enterprise World Model Engine, simulation rollouts combine discrete state transitions, constraint boundary evaluations, stochastic dynamics, and provenance trace collection.

Prior to Round 3 (R04), three structural limitations existed:
1. **Absence of Deterministic Performance Gates**: The repository only had a coarse throughput sanity guard (`tests/benchmark/test_throughput_benchmark.py`). Attempting to gate PRs on raw wall-clock time (`time.perf_counter`) on shared cloud CI runners (e.g. GitHub Actions `ubuntu-latest`) introduces $15\text{--}30\%$ jitter due to hyperthread contention, CPU thermal throttling, and VM scheduling noise.
2. **Lack of Long-Term System-of-Record**: There was no commit-range performance history, no automated change-point detection, and no bisection mechanism to identify which historical commit introduced a gradual performance regression.
3. **Memory and Numerical Rigor Gaps**: There were no memory limits or leak gates, despite memory footprint being deterministic. Furthermore, numerical test assertions lacked uniform tolerance standards, leaving the suite susceptible to the classic Goldberg floating-point zero-hazard trap (`assert_allclose(actual, desired, atol=0)` against an expected zero).

## Decision
We establish a four-part performance and numerical quality assurance architecture:

### 1. Deterministic Per-PR Performance Gates (CodSpeed Instrumentation)
- We integrate **CodSpeed** (`pytest-codspeed`, `CodSpeedHQ/action`) in **instrumentation mode** on a curated performance-critical subset:
  1. Rollout Step (`execute_step`)
  2. Monte Carlo Spawn (`SeedSequence.spawn` + state cloning)
  3. Canonical Fingerprint (`state.fingerprint` SHA-256)
  4. Constraint Evaluation (`world.constraints.validate_actions`)
  5. Serialization Round-Trip (`state.to_dict()` $\to$ `WorldState.from_dict()`)
  6. Systemic Trace Build (`SystemicTrace` DAG construction)
- **Mechanism**: CodSpeed measures exact CPU instructions, branch misses, and cache interactions rather than wall-clock time, eliminating CI runner variance and providing a safe, noise-free gating signal.
- **Graduation Policy**: Advisory for $\approx 1$ release cycle while establishing baselines, then promoted to a blocking PR gate ($>5\%$ instruction regression halts merge).
- **Advisory Wall-Clock Sanity**: Paired with `pytest-benchmark` (`--benchmark-only`) on the same subset for local and nightly profiling.

### 2. Long-Term Performance History (Airspeed Velocity - ASV)
- We integrate **asv (airspeed velocity)** (`asv.conf.json`, `benchmarks/benchmarks_asv.py`, `scripts/run_asv.py`) running periodically on a dedicated/tuned runner.
- Serves as the system-of-record for commit-range throughput trends, automated change-point detection, and bisection (`asv find`).
- Publishes interactive static HTML dashboards to `.asv/html/`. Stays advisory (non-blocking) due to commit-range execution overhead.

### 3. Deterministic Memory Gates (pytest-memray & tracemalloc)
- Memory usage is deterministic: state size, object allocations, and DAG scaling follow predictable bounds. Memory regressions are therefore safe **hard gates**.
- Gated via `@pytest.mark.limit_memory("N MB")` and `@pytest.mark.limit_leaks`:
  - Single-step rollout: $\le 25\ \text{MB}$
  - 50-step trajectory: $\le 50\ \text{MB}$
  - 10 Monte Carlo replicas: $\le 60\ \text{MB}$
  - 50 state branches: $\le 15\ \text{MB}$ (verifies immutable structural sharing)
  - 500-node systemic trace: $\le 20\ \text{MB}$
  - Simulation loop leak gate: $\le 250\ \text{KB}$ net growth across 40 iterations
- Enforced at native C allocator level via `pytest-memray` under Linux CI and Python standard library `tracemalloc` cross-platform.

### 4. Numerical Reproducibility Discipline & Differential Tests
- **Goldberg Zero-Hazard Elimination**: In `numpy.testing.assert_allclose`, when `desired == 0.0`, relative tolerance has zero effect ($\text{rtol} \times 0.0 = 0.0$). We mandate an **explicit `atol`** whenever an expected value can be zero.
- **Domain Tolerance Standardization**:
  - `physical_quantities`: `rtol=1e-6`, `atol=1e-7`
  - `probabilities`: `rtol=1e-5`, `atol=1e-7`
  - `financial_currency`: `rtol=1e-4`, `atol=1e-4`
  - `constraint_margins`: `rtol=1e-5`, `atol=1e-6`
  - `zero_bounded`: `rtol=1e-5`, `atol=1e-6`
- **Differential Testing**: Added differential tests comparing optimized/component implementations against analytical reference oracles (dynamics vs analytical transfer, constraint registry vs direct boundary checks, serial vs multiprocessing rollout equivalence).
- **RNG Policy**: Strictly mandate `SeedSequence.spawn` for parallel Monte Carlo streams; prohibit the `seed + i` anti-pattern (correlated hyperplanes).
- **BLAS Threading**: Pin single-threaded BLAS (`OMP_NUM_THREADS=1` etc.) for critical numeric tests to tame floating-point non-associativity reduction variance. Document that bit-exact cross-platform invariance is mathematically unachievable (Goldberg 1991).

## Consequences

### Positive
- **Credible Performance Engineering**: Prevents insidious CPU-instruction and memory regressions from entering the codebase without causing CI flakiness.
- **Historical Visibility**: ASV dashboard provides transparent long-term throughput trends and automated regression bisection.
- **Zero Memory Leaks**: Hard gates ensure simulation engines running multi-hour Monte Carlo campaigns do not exhaust heap memory.
- **Robust Numeric Tests**: Eliminates flaky floating-point test failures caused by zero-comparison tolerances, multi-threaded BLAS reductions, or correlated RNG streams.

### Negative / Trade-offs
- Benchmarks must maintain strict hygiene and avoid external side effects.
- Running ASV across broad commit ranges is computationally expensive, necessitating its placement in a periodic/scheduled workflow rather than per-PR CI.
