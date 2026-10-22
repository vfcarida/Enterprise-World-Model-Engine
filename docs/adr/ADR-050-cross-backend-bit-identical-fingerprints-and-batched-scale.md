# ADR-050: Cross-Backend Reproducibility Contract, Structure-of-Arrays Batched Kernel, and Scalable Execution

## Context and Problem Statement

Enterprise World Model (EWM) simulations must scale across three independent operational dimensions:
1. **Vectorization**: Batching $N$ Monte Carlo rollouts simultaneously rather than sequentially stepping per-instance state objects.
2. **Compilation**: Leveraging array compiler toolchains (JAX `jit(vmap(scan(step)))` and native C/Rust extensions) to minimize Python dispatch overhead.
3. **Distribution**: Scaling out independent parallel rollouts across multicore clusters via Ray and Dask.

In scientific and enterprise computing, moving from single-thread CPU execution to GPU compilers or distributed cluster execution introduces floating-point rounding differences (FMA instruction contraction, parallel tree reductions), thread-scheduling variance, and out-of-order reduction hazards (Goldberg, 1991).

For EWM Engine, claiming universal bit-for-bit floating-point identity across arbitrary hardware, compilers, BLAS thread counts, and dtypes would be scientifically dishonest, while ignoring reproducibility altogether would break **Invariant #5: Deterministic Reproducibility & Canonical Fingerprinting**.

Therefore, EWM establishes an explicit **three-tier numerical reproducibility contract** (`BITWISE_REPRODUCIBLE`, `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE`, `NON_CERTIFIED_BACKEND`) paired with order-independent distributed reduction.

## Considered Options

1. **Universal Cross-Hardware Bit-Identity Claim**:
   - Promise identical SHA-256 state fingerprints across all CPUs, GPUs, compilers, BLAS libraries, and dtypes.
   - *Rejected*: Scientifically invalid. Even with a counter-based PRNG (Threefry), XLA compiler fusion, FMA contractions, and SIMD/GPU reduction trees alter last-bit rounding unless the hardware, runtime, dtype (`float64`), and single-thread BLAS (`OMP_NUM_THREADS=1`) are strictly pinned.

2. **GPU-Centric Complete Rewrite**:
   - Rewrite the core simulation kernel into a specialized CUDA/C++ framework, abandoning pure-Python fallback and lightweight installs.
   - *Rejected*: Violates the minimal-core dependency constraint. EWM core must remain installable on standard Python (`pydantic`, `numpy`, `pyyaml`) without requiring system compilers, CUDA drivers, or heavy ML toolchains.

3. **Layered Mathematical Projection (Structure-of-Arrays) with Three-Tier Certified Array API Backends**:
   - Keep the canonical immutable typed-graph (`WorldState`) as the ground-truth oracle for state representation and SHA-256 fingerprinting.
   - Project the ensemble of $N$ rollouts into a Structure-of-Arrays (`SoAStateTensor`) where each resource and metric field is advanced via vectorized operations conforming to the Python Array API standard (2025.12).
   - Use `SeedSequence(scenario.seed).spawn(scenario.samples)` to derive independent, deterministic PRNG streams.
   - Formalize a three-tier **Backend Certification Contract**:
     - `BITWISE_REPRODUCIBLE`: Pure-Python and NumPy (`float64`, `OMP_NUM_THREADS=1`, pinned runtime/BLAS/Python/NumPy versions).
     - `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE`: JAX (`jax_enable_x64=True`, Threefry PRNG), certified to match reference trajectories within explicit `(rtol=1e-6, atol=1e-7)` tolerances rather than universal cross-hardware bit identity.
     - `NON_CERTIFIED_BACKEND`: CuPy and raw CUDA floating-point atomic reductions that introduce hardware non-determinism, explicitly refused when `certify_equivalence=True`.
   - Enforce **Order-Independent Distributed Reduction**: All distributed execution backends (Ray, Dask, Multiprocessing) sort returned rollouts strictly by `sample_id` before fingerprint reduction, guaranteeing scheduling invariance.

## Decision Outcome

Chosen option: **Option 3 (Layered Mathematical Projection with Three-Tier Certified Array API Backends)**.

### Architectural Rules & Invariants

1. **Source of Truth vs. Mathematical Projection**:
   - The immutable `WorldState` graph is the canonical source of truth for state semantics, constraint evaluation, and SHA-256 fingerprinting.
   - `SoAStateTensor` is a high-performance mathematical projection representing state variables as continuous $(N, D_R)$ arrays.
   - Projection round-trips (`from_world_state` and `to_world_states`) preserve identical entity identities, metadata, and numerical values.

2. **PRNG Derivation & Threefry Determinism**:
   - All pseudo-random number streams originate from a single master seed:
     $$\text{child\_seeds} = \text{SeedSequence}(\text{seed}).\text{spawn}(N)$$
   - Each branch $i \in [0, N-1]$ consumes an independent PCG64 stream or Threefry counter key derived from `child_seeds[i]`.

3. **Three-Tier Backend Certification Hierarchy**:
   | Backend | Implementation | Certification Status | Tolerance Contract |
   |:---|:---|:---|:---|
   | **Pure-Python** | Python loop reference kernel | `BITWISE_REPRODUCIBLE` (Oracle) | `rtol=0.0, atol=0.0` (pinned runtime) |
   | **NumPy** | Array API standard (`numpy`, `float64`) | `BITWISE_REPRODUCIBLE` | `rtol=0.0, atol=0.0` (pinned runtime) |
   | **JAX** (`CPU`/`GPU`) | JAX `vmap`/`scan` (`x64=True`, Threefry) | `NUMERICALLY_EQUIVALENT_WITH_TOLERANCE` | `rtol=1e-6, atol=1e-7` |
   | **CuPy / CUDA** | Raw CUDA float atomics | `NON_CERTIFIED_BACKEND` | `None` (refused if `certify_equivalence=True`) |

4. **Order-Independent Distributed Reduction**:
   - In distributed computing (Ray / Dask / Multiprocessing), worker rollouts finish asynchronously out of order.
   - All executors must collect and sort `RolloutResult` items by `sample_id` ascending prior to calculating summary statistics, trajectories, and ensemble fingerprints:
     $$\mathcal{R}_{\text{reduced}} = \text{sort\_by\_sample\_id}(\{\mathcal{R}_{\text{worker}}\})$$
   - This ensures the simulation result fingerprint is mathematically invariant to task arrival order, worker latency, or cluster topology on homogeneous worker nodes.

5. **Pure-Python Fallback & Optional Toolchains**:
   - Core EWM installation requires zero C compilers, CUDA toolchains, or heavy frameworks.
   - High-performance backends (`jax`, `cupy`, `dask`, `ray`) are configured as optional extras in `pyproject.toml` (`[jax]`, `[cuda]`, `[distributed]`).
   - If an optional accelerator is unavailable, clear diagnostic errors are surfaced with installation guidance.

## Consequences

### Positive
- **Scientific Honesty & Auditability**: Bit-exact SHA-256 equivalence between local sequential pure-Python and batched NumPy under pinned runtimes (`BITWISE_REPRODUCIBLE`), explicit `(rtol, atol)` certification for compiled XLA backends (`NUMERICALLY_EQUIVALENT_WITH_TOLERANCE`), and fail-closed refusal of non-certified backends (`NON_CERTIFIED_BACKEND`) on certified paths.
- **Reproducible Distributed Parallelism**: Distributed Monte Carlo rollouts under Ray, Dask, and Multiprocessing produce identical results regardless of worker completion order.
- **Clean Extensibility**: The Python Array API standard provides a uniform tensor abstraction across NumPy, JAX, and CuPy without vendor lock-in.

### Negative
- **Projection Overhead on Small Batches**: For $N=1$ or trivial models, projecting into SoA arrays and back carries minor memory copying overhead compared to raw in-place mutations.
- **Pinned Environment Requirement for Bitwise Reproduction**: Bitwise SHA-256 identity across machines requires pinning Python, NumPy, `float64`, and `OMP_NUM_THREADS=1` (as provided by the repository's Docker/lockfile reproduction environment).
