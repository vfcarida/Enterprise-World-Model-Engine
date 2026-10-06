# Numerical Reproducibility and Deterministic Performance Policy

> **Discipline**: Performance regression testing, deterministic gating, and numerical quality assurance.  
> **Research Anchors**: CodSpeed, Airspeed Velocity (asv), `pytest-benchmark`, `pytest-memray`, `numpy.testing`, David Goldberg (*"What Every Computer Scientist Should Know About Floating-Point Arithmetic"*, ACM Computing Surveys, 1991).  
> **Normative Standards**: AC-004 (Reproducibility Contract), ADR-005, ADR-012, ADR-032.

---

## 1. Executive Summary & Philosophy

In an enterprise world model and scientific simulation framework, an absence of performance and numerical regression gates creates a severe credibility gap. If simulation throughput degrades or floating-point rounding causes boundary constraint flips across runs or platforms, decision-makers cannot trust counterfactual evaluations.

This policy codifies two fundamental operational principles:
1. **Deterministic Signals Gate; Wall-Clock Signals Advise**: Wall-clock execution time on shared cloud CI runners suffers from hyperthread contention, CPU thermal throttling, and VM scheduling jitter. Wall-clock runs stay advisory. We gate pull requests exclusively on **deterministic proxies**: CPU-instruction counts via **CodSpeed** instrumentation and memory allocation bounds via **pytest-memray** / `tracemalloc`.
2. **Numerical Tolerance Discipline**: Floating-point arithmetic is non-associative ($ (a + b) + c \neq a + (b + c) $) and non-exact. We codify rigorous tolerance standards across every simulation domain, eliminate the classic `assert_allclose(atol=0)`-vs-zero bug, mandate `SeedSequence.spawn` for parallel Monte Carlo streams, and define differential testing oracles.

---

## 2. Deterministic Performance Gates (CodSpeed & pytest-benchmark)

### 2.1 The Problem with Wall-Clock CI Gates

Traditional benchmark gates in CI use wall-clock timing (`time.perf_counter()`). On shared runners (e.g. GitHub Actions `ubuntu-latest`), wall-clock variance frequently exceeds $15\text{--}30\%$. Setting tight gates causes continuous false-positive CI failures, while loose gates allow substantial real regressions to escape into production.

### 2.2 CodSpeed Instrumentation Mode

We integrate **CodSpeed** (`pytest-codspeed`, `CodSpeedHQ/action`) in **instrumentation mode**. Rather than timing wall-clock duration, CodSpeed measures exact **CPU instructions, branch misses, and cache interactions** consumed during function execution using low-overhead Valgrind/Callgrind-derived instrumentation.

- **Deterministic & Noise-Free**: Instruction counts remain virtually constant regardless of runner CPU frequency, hyperthreading state, or background load.
- **Rollout Strategy**:
  - *Advisory Period (1 release)*: CodSpeed posts differential PR comments highlighting instruction changes without blocking merges.
  - *Blocking Gate*: After calibration, any unintended instruction increase $> 5\%$ on curated benchmarks halts PR merging.

### 2.3 Curated Performance-Critical Subset

Benchmarks are defined in [`benchmarks/test_curated_benchmarks.py`](file:///benchmarks/test_curated_benchmarks.py) and executed via pytest's standard `benchmark` fixture:

| Benchmark Target | Method / Operation | Target Budget (Instructions / Ops) | Rationale |
|---|---|---|---|
| **1. Rollout Step** | `execute_step(world, state, ...)` | $\approx 430\ \mu\text{s}$ ($>90\ \text{ops/s}$) | Single discrete simulation step: dynamics evaluation, event dispatch, and constraint validation. |
| **2. Monte Carlo Spawn** | `SeedSequence.spawn(n)` + state clone | $\approx 2.1\ \text{ms}$ ($>50\ \text{ops/s}$) | Parallel trajectory stream initialization and initial state replication. |
| **3. Canonical Fingerprint** | `state.fingerprint` (SHA-256) | $\approx 40\ \mu\text{s}$ ($>450\ \text{ops/s}$) | Cryptographic state hashing ensuring deterministic caching and provenance. |
| **4. Constraint Evaluation** | `world.constraints.validate_actions` | $\approx 94\ \text{ms}$ ($>10\ \text{ops/s}$) | Multi-action boundary validation and invariant predicate checks. |
| **5. Serialization Round-Trip** | `to_dict()` $\to$ `from_dict()` | $\approx 9.5\ \mu\text{s}$ ($>1{,}000\ \text{ops/s}$) | Pydantic JSON/dict state export and model validation round-trip. |
| **6. Systemic Trace Build** | `SystemicTrace` DAG construction | $\approx 1.0\ \mu\text{s}$ ($>3{,}000\ \text{ops/s}$) | Causal/structural trace node registration and edge linking. |

### 2.4 Wall-Clock Sanity (pytest-benchmark)

In parallel with CodSpeed, the same benchmark file runs locally and in nightly pipelines via `pytest-benchmark`:
```bash
pytest benchmarks/test_curated_benchmarks.py --benchmark-only
```
This produces detailed statistics (Min, Max, Mean, StdDev, Median, IQR, OPS) for advisory wall-clock profiling.

---

## 3. Long-Term Performance History (Airspeed Velocity - ASV)

While CodSpeed gates individual PRs, long-term trends and historical bisections require a dedicated system-of-record.

We configure **Airspeed Velocity (ASV)** (`asv.conf.json`, [`benchmarks/benchmarks_asv.py`](file:///benchmarks/benchmarks_asv.py)):
- **Commit-Range History**: Tracks performance evolution across all commits on `main`.
- **Automatic Change-Point Detection**: Detects statistically significant regression steps across releases.
- **Bisecting Regressions (`asv find`)**: Automatically bisects git commit ranges to identify the exact commit introducing a throughput dip.
- **Static HTML Dashboard**: Collates benchmark trends into an interactive HTML dashboard published to `.asv/html/`.
- **Running Locally**:
  ```bash
  python scripts/run_asv.py --quick
  python scripts/run_asv.py --publish
  ```

---

## 4. Deterministic Memory Gates (pytest-memray)

### 4.1 Memory as a Deterministic Quality Gate

Unlike wall-clock execution time, peak heap memory allocation and object counts are deterministic for a fixed workload. A simulation step either allocates $10\text{ MB}$ or it leaks memory; variance across runs is negligible. Therefore, **memory limits are safe, hard CI gates**.

### 4.2 Enforcement Strategy

We implement a dual enforcement architecture in [`tests/performance/test_memory_gates.py`](file:///tests/performance/test_memory_gates.py):
1. **Linux CI (C Allocator Instrumentation)**: Uses `pytest-memray` with `@pytest.mark.limit_memory("N MB")` and `@pytest.mark.limit_leaks`. `memray` tracks every `malloc`, `calloc`, `realloc`, and `free` call at the native C runtime level.
2. **Cross-Platform / Windows (tracemalloc)**: Standard library `tracemalloc` budget assertions (`assert_memory_budget`, `assert_no_memory_leak`) ensure deterministic memory enforcement runs smoothly on Windows and macOS developer machines.

### 4.3 Memory Budgets

| Test Case | Ceiling | Verification Target |
|---|---|---|
| Single Step Rollout | $\le 25\ \text{MB}$ | Verifies transient action/event memory is released immediately. |
| 50-Step Trajectory | $\le 50\ \text{MB}$ | Verifies memory scales sub-linearly with time horizon. |
| 10 Monte Carlo Replicas | $\le 60\ \text{MB}$ | Verifies trajectory storage footprint across samples. |
| State Branching | $\le 15\ \text{MB}$ | Verifies immutable structural sharing across 50 branches. |
| 500-Node Systemic Trace | $\le 20\ \text{MB}$ | Verifies DAG edge/node dictionary scaling. |
| Simulation Loop Leak Gate | $\le 250\ \text{KB}$ net growth | Verifies zero memory leakage across 40 simulation iterations. |

---

## 5. Numerical Reproducibility Discipline

### 5.1 The Goldberg Zero-Hazard Trap

In David Goldberg's seminal paper *"What Every Computer Scientist Should Know About Floating-Point Arithmetic"* (1991), floating-point comparisons are shown to fail catastrophically when relative tolerance is used against zero.

In `numpy.testing.assert_allclose(actual, desired, rtol=..., atol=...)`, the underlying condition checked is:
$$\lvert \text{actual} - \text{desired} \rvert \le \text{atol} + \text{rtol} \times \lvert \text{desired} \rvert$$

When `desired == 0.0`:
$$\lvert \text{actual} - 0.0 \rvert \le \text{atol} + \text{rtol} \times 0.0 = \text{atol}$$

**If `atol` is left at default `0.0`, the allowable error is exactly $0.0$!**  
Any infinitesimal IEEE 754 floating-point epsilon (e.g. $10^{-16}$) will immediately raise an `AssertionError`.

> **MANDATORY POLICY**:  
> Every numeric assertion where the expected value can be zero MUST supply an explicit, non-zero `atol`. Default `atol=0.0` is strictly forbidden for zero-bounded tests.

### 5.2 Domain-Specific Tolerance Table

All numeric tests across EWM Engine standardize on domain-tailored tolerances:

| Domain | `rtol` | `atol` | Description & Justification |
|---|---|---|---|
| **`physical_quantities`** | `1e-6` | `1e-7` | Mass balances, inventory counts, discrete transfers. Absorbs standard double-precision accumulation noise. |
| **`probabilities`** | `1e-5` | `1e-7` | Probability simplex sums ($\sum p_i = 1.0$), transition CDFs, Dirichlet distributions. |
| **`financial_currency`** | `1e-4` | `1e-4` | Financial ledger balances, revenues, costs. Tolerates fractional-cent rounding. |
| **`constraint_margins`** | `1e-5` | `1e-6` | Robustness degrees, STL satisfaction metrics, constraint slack margins. |
| **`zero_bounded`** | `1e-5` | `1e-6` | Any test comparing against an analytical zero target. |

---

## 6. Differential Testing Architecture

To detect silent algorithmic degradation, fast-path bugs, or solver drift, the engine uses **differential testing**: asserting agreement within tolerance between two independent implementations:

1. **Analytical Reference vs Dynamics Model**:  
   Direct closed-form arithmetic balances are compared against `DeterministicTransferDynamics.transition(...)`.
2. **Direct Boundary Oracle vs Constraint Registry**:  
   A direct analytical boundary filter is evaluated against `ConstraintRegistry.validate_actions(...)` to ensure identical action filtering and violation masks.
3. **Serial vs Distributed Rollout Determinism**:  
   `SerialExecutor` and `MultiprocessingExecutor` must produce bit-for-bit identical final state fingerprints and resource trajectories under the same master seed.

All differential tests are codified in [`tests/numerical/test_differential_and_tolerances.py`](file:///tests/numerical/test_differential_and_tolerances.py).

---

## 7. Random Number Generation (RNG) & Parallel Stream Splitting

### 7.1 The `seed + i` Anti-Pattern

A common anti-pattern in parallel stochastic simulations is initializing worker $i$ with `seed + i`:
```python
# FORBIDDEN ANTI-PATTERN:
worker_rng = np.random.default_rng(master_seed + i)
```
**Why this is prohibited**:
- Incrementing integer seeds modifies only low-order bits.
- In pseudorandom generators (including linear congruential and Mersenne Twister variants), close seeds can project into correlated hyperplanes (Marsaglia's theorem), creating spurious statistical coupling between supposedly independent Monte Carlo rollouts.

### 7.2 The Gold Standard: `SeedSequence.spawn`

Parallel trajectory streams must always be spawned via `np.random.SeedSequence`:
```python
# MANDATORY STANDARD (AC-004 / R04):
seed_seq = np.random.SeedSequence(master_seed)
child_sequences = seed_seq.spawn(n_workers)
worker_rngs = [np.random.default_rng(child) for child in child_sequences]
```
`SeedSequence` uses cryptographic-grade hashing (SplitMix64) to advance the generator state, guaranteeing statistical independence and bit-for-bit cross-run reproducibility.

### 7.3 NumPy Generator Bitstream Policy

- NumPy's `Generator` API guarantees algorithmic stability for standard distributions.
- However, NumPy does **not** guarantee bitstream equality across major NumPy releases.
- Simulation pipelines requiring multi-year frozen bitstreams must pin NumPy minor versions in lockfiles.

---

## 8. BLAS Threading Pinning & Non-Associativity

### 8.1 Floating-Point Non-Associativity

In IEEE 754 arithmetic:
$$(10^{16} + (-10^{16})) + 1.0 = 0.0 + 1.0 = 1.0$$
$$10^{16} + ((-10^{16}) + 1.0) = 10^{16} - 10^{16} = 0.0$$

In multi-threaded BLAS/LAPACK libraries (OpenBLAS, Intel MKL, Apple Accelerate), matrix multiplications and vector dot products chunk arrays dynamically across worker threads. Because the summation reduction tree varies dynamically depending on thread scheduling, **multi-threaded BLAS operations can produce slight floating-point discrepancies across runs on the same hardware**.

### 8.2 Pinning BLAS in Numeric Testing

Critical numeric regression tests pin single-threaded BLAS execution:
```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```
This forces deterministic sequential summation order across all vector reductions.

### 8.3 Cross-Hardware Bit-Exact Invariance is Impossible

As documented by Goldberg, bit-exact cross-platform floating-point reproducibility across Intel, AMD, ARM, and Apple Silicon is mathematically unachievable due to:
- Fused Multiply-Add (FMA) contraction instructions ($a \times b + c$ computed with a single rounding step vs two).
- Differing extended precision registers (x87 80-bit vs SSE2 64-bit).
- Differing compiler optimization flags (`-ffast-math` breaks IEEE 754 compliance).

Therefore, EWM Engine guarantees **bitwise reproducibility across identical hardware/software environments**, and **tolerance-bounded reproducibility across heterogeneous platforms**.

---

## 9. Machine Learning & PyTorch Deterministic Recipe (`[ml]` Extra)

When using learned neural dynamics or reinforcement learning adapters with PyTorch, the following configuration must be applied:

```python
import os
import torch

# 1. Pinned master seed
torch.manual_seed(master_seed)

# 2. Force deterministic algorithms
torch.use_deterministic_algorithms(True)

# 3. Deterministic cuDNN backends
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Required for deterministic CUDA convolutions / scatter operations:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
```

**Caveat**: Bitwise cross-hardware reproducibility across different GPU architectures (e.g. Hopper vs Ampere vs Ada Lovelace) is outside the determinism guarantee due to varying CUDA warp scheduling and hardware-level tensor core reductions.
