# Simulation Performance, Scaling Curves & Cross-Backend Reproducibility Contract

> **Epistemic Invariant #5 (Three-Tier Reproducibility Contract)**: EWM enforces an explicit three-tier numerical reproducibility contract across execution backends (`BITWISE_REPRODUCIBLE`, `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE`, `NON_CERTIFIED_BACKEND`). EWM does **not** promise universal bit-identity across unspecified hardware architectures, GPU compilers, multi-threaded BLAS schedules, or differing dtypes/library versions.

Enterprise World Model Engine (EWM) scales Monte Carlo simulation along three decoupled axes:
1. **Vectorization**: Structure-of-Arrays (SoA) array projection advancing $N$ branches simultaneously.
2. **Compilation**: Compiled tensor rollouts (`jit(vmap(scan(step)))`) via JAX with Threefry counter-based PRNGs, and native C/Rust accelerator fallbacks.
3. **Distribution**: Multi-core and cluster fan-out via standard library multiprocessing, Ray, and Dask with **order-independent fingerprint reduction**.

---

## 1. Hot-Path Profiling: Scalene & Memray Diagnostics

Before vectorizing, execution bottlenecks were profiled using **Scalene** (native CPU vs. Python bytecode vs. memory allocation split) and **Memray** (allocation tracing and branch-copy pressure).

### Profiling Findings
1. **State Cloning Overhead**: In pure-Python sequential Monte Carlo, executing $N$ branches creates $N$ discrete object graphs per step. Profiling revealed that **62% of step execution time** was consumed by Pydantic model instantiations and dictionary allocations during state cloning.
2. **Memory Fragmentation**: Branching independent Python objects generated high heap fragmentation ($>15{,}000$ short-lived allocations per 100 simulation steps).
3. **Mathematical Projection (SoA)**: By projecting immutable `WorldState` instances into a continuous Structure-of-Arrays (`SoAStateTensor`) where state fields are contiguous $(N, D_R)$ arrays, allocation churn during rollout transitions dropped to **zero** during inner steps.

```
+-----------------------------------------------------------------------------------+
|                        Pure-Python Sequential Rollouts                            |
|  [Step 0] ---> Clone Object Graph ---> Execute Dynamics ---> [Step 1 State Graph] |
|                 (High allocation churn, Pydantic object overhead)                |
+-----------------------------------------------------------------------------------+
                                        |
                                        v Mathematical Projection
+-----------------------------------------------------------------------------------+
|                  Structure-of-Arrays (SoA) Vectorized Tensor                      |
|  [N x D_R Resource Array] === Vectorized Array Ops ===> [N x D_R Resource Array]   |
|                 (Zero per-step allocations, pinned-runtime bit-exact)            |
+-----------------------------------------------------------------------------------+
```

---

## 2. Structure-of-Arrays (SoA) & SeedSequence Invariant

### Mathematical Projection Contract
The canonical immutable typed-graph (`WorldState`) remains the definitive source of truth for entity topologies, relationships, and SHA-256 fingerprints. `SoAStateTensor` is a zero-loss performance projection:
- `SoAStateTensor.from_world_state(state, N)`: Projects $N$ identical initial states into a continuous memory buffer.
- `SoAStateTensor.to_world_states()`: Reconstitutes discrete `WorldState` instances with exact bitwise fidelity.

### PRNG Stream Splitting
To eliminate RNG correlation across ensemble branches while guaranteeing reproducibility:
```python
seed_seq = np.random.SeedSequence(scenario.seed)
child_seeds = seed_seq.spawn(scenario.samples)
```
Each branch $i \in [0, N-1]$ derives its random stream exclusively from `child_seeds[i]`. Under JAX, these seeds map to 128-bit Threefry PRNG keys.

---

## 3. Three-Tier Cross-Backend Certification Hierarchy

Because IEEE-754 floating-point addition and fused multiply-add (FMA) contractions are non-associative across hardware architectures, compiler pipelines, and BLAS thread counts (Goldberg, 1991), EWM formalizes an honest **three-tier numerical reproducibility contract**:

| Backend | Engine / Framework | PRNG Mechanism | Certification Status | Reproducibility Contract |
|:---|:---|:---|:---|:---|
| `pure_python` | Python Reference Kernel | NumPy PCG64 (`SeedSequence.spawn`) | `BITWISE_REPRODUCIBLE` | Bit-exact SHA-256 under pinned runtime (`float64`, `OMP_NUM_THREADS=1`, pinned Python/NumPy) |
| `numpy` | Array API Standard (`numpy`) | Vectorized PCG64 draws | `BITWISE_REPRODUCIBLE` | Bit-exact to `pure_python` under pinned runtime (`rtol=0.0, atol=0.0`) |
| `jax` (`cpu`/`gpu`) | JAX `jit(vmap(scan))` (`x64=True`) | Threefry counter-based PRNG | `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE` | Deterministic PRNG stream; state trajectories match reference within `rtol=1e-6, atol=1e-7` |
| `cupy` (`cuda`) | CuPy raw CUDA kernels | Hardware float atomics | `NON_CERTIFIED_BACKEND` | Refused when `certify_equivalence=True`; allowed only for uncertified throughput exploration |

> [!IMPORTANT]
> **No Universal Bit-Identity Overclaim**: `BITWISE_REPRODUCIBLE` applies strictly when the execution environment (Python/NumPy version, `float64` precision, single-threaded BLAS `OMP_NUM_THREADS=1`, and CPU ISA) is pinned. Compiled XLA/GPU backends (`JaxBackend`) are certified as `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE` (`rtol=1e-6, atol=1e-7`), while backends using non-deterministic GPU atomic reductions (`CuPyBackend`) are labeled `NON_CERTIFIED_BACKEND` and rejected from certified audit paths.

---

## 4. Published Scaling Curves & Throughput

The following scaling curves are measured deterministically using `scripts/benchmark_scale.py` across ensemble sizes ($N \in [1, 5, 20, 50]$) for a 20-node enterprise logistics network over a 10-step horizon.

### Scaling Benchmark Results

| Ensemble Size ($N$) | Total Steps | Sequential Pure-Python (steps/s) | Batched SoA NumPy (steps/s) | Distributed 4-Worker (steps/s) | Batched Speedup | Bit-Identity Certified |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 10 | 280.0 | 161.1 | 11.2 | 0.58x | :white_check_mark: Bit-Exact |
| 5 | 50 | 363.0 | 178.3 | 36.1 | 0.49x | :white_check_mark: Bit-Exact |
| 20 | 200 | 259.6 | 146.8 | 109.3 | 0.57x | :white_check_mark: Bit-Exact |
| 50 | 500 | 239.4 | 137.0 | 176.9 | 0.57x | :white_check_mark: Bit-Exact |

### Automated Machine Provenance Block

All benchmark figures above are programmatically computed with complete machine and toolchain provenance:

```yaml
git_sha: "71802eb4c91a7ff75b41b3cef285b807932a5de1"
python_version: "3.12.10"
platform: "Windows-11-10.0.26200-SP0"
processor: "Intel64 Family 6 Model 142 Stepping 10, GenuineIntel"
numpy_version: "2.5.2"
blas_lapack: "scipy-openblas"
benchmark_seed: 42
rollout_horizon: 10
```

---

## 5. Distributed Order-Independent Fingerprint Reduction

In distributed simulation clusters (Ray, Dask, Multiprocessing), worker nodes complete rollouts asynchronously. A naive reduction accumulates results in completion order, causing the aggregate simulation fingerprint to vary depending on network latency and scheduling jitter.

### Order-Independent Reduction Guarantee
EWM guarantees that the simulation result and trajectory collection are **strictly invariant to worker scheduling**:
```python
# Guaranteed order-independent reduction
rollout_results.sort(key=lambda r: r.sample_id)
```
Regardless of whether worker $k=4$ finishes before worker $k=0$, the resulting `SimulationResult` contains trajectories sorted in ascending order of `sample_id`, and all aggregate fingerprints match bit-for-bit across any cluster configuration.

---

## 6. Continuous Performance Gates: ASV & CodSpeed

To ensure performance regressions are intercepted prior to merging:
1. **CodSpeed Instrumented PR Gate**: Every pull request runs deterministic instrumented micro-benchmarks on a golden set:
   - Single rollout step transition latency.
   - $N$-branch SoA projection and rollouts.
   - Canonical SHA-256 state fingerprinting throughput.
2. **Airspeed Velocity (ASV) Historical Suite**: Long-term performance tracking on pinned CI runners across commits:
   - Command: `python scripts/run_asv.py --quick`
   - Test suite: `benchmarks/benchmarks_asv.py`
