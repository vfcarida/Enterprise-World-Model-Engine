# ADR-050: Cross-Backend Bit-Identical Fingerprints, Structure-of-Arrays Batched Kernel, and Scalable Execution

## Context and Problem Statement

Enterprise World Model (EWM) simulations must scale across three independent operational dimensions:
1. **Vectorization**: Batching $N$ Monte Carlo rollouts simultaneously rather than sequentially stepping per-instance state objects.
2. **Compilation**: Leveraging array compiler toolchains (JAX `jit(vmap(scan(step)))` and native C/Rust extensions) to minimize Python dispatch overhead.
3. **Distribution**: Scaling out independent parallel rollouts across multicore clusters via Ray and Dask.

In scientific and enterprise computing, nearly every simulator sacrifices determinism when scaling. Moving from single-thread CPU execution to GPU or distributed cluster execution frequently yields diverging floating-point rounding, non-deterministic race conditions, and scheduling-dependent reduction orders.

For EWM Engine, non-deterministic drift violates **Invariant #5: Bit-Identical Determinism & Canonical Fingerprinting**. If a policy decision or risk assessment computed on a laptop cannot be reproduced bit-for-bit on a multi-GPU cluster or Ray cluster, auditability and regulatory compliance collapse.

EWM's defining, citable differentiator is: **bit-identical SHA-256 fingerprints across computational backends** (pure-Python ⇄ NumPy ⇄ JAX-CPU ⇄ JAX-GPU ⇄ Ray/Dask-distributed).

## Considered Options

1. **Ad-Hoc Floating-Point Tolerance Bands**:
   - Allow slight float deviations across backends ($\epsilon \le 10^{-6}$) and use fuzzy hashing or approximate trajectory matching.
   - *Rejected*: Violates Invariant #5. Approximate matching makes digital twin audit trails unverifiable and introduces subtle, compounding policy divergence over long horizons.

2. **GPU-Centric Complete Rewrite**:
   - Rewrite the core simulation kernel into a specialized CUDA/C++ framework, abandoning pure-Python fallback and lightweight installs.
   - *Rejected*: Violates the zero-forced-dependency core constraint. EWM core must remain installable on standard Python (`pydantic`, `numpy`, `pyyaml`) without requiring system compilers, CUDA drivers, or heavy ML toolchains.

3. **Layered Mathematical Projection (Structure-of-Arrays) with Certified Array API Backends**:
   - Keep the canonical immutable typed-graph (`WorldState`) as the ground-truth oracle for state representation and SHA-256 fingerprinting.
   - Project the ensemble of $N$ rollouts into a Structure-of-Arrays (`SoAStateTensor`) where each resource and metric field is advanced via vectorized operations conforming to the Python Array API standard (2025.12).
   - Use `SeedSequence(scenario.seed).spawn(scenario.samples)` to derive independent, deterministic PRNG streams.
   - Formalize a strict **Backend Certification Contract**:
     - `CERTIFIED_BIT_IDENTICAL`: NumPy and JAX (using Threefry counter-based PRNG, documented CPU/GPU bit-identical).
     - `THROUGHPUT_ONLY`: CuPy and CUDA floating-point atomic reductions that introduce hardware non-determinism are explicitly flagged as throughput-only and refused from certified audit paths.
   - Enforce **Order-Independent Distributed Reduction**: All distributed execution backends (Ray, Dask, Multiprocessing) sort returned rollouts strictly by `sample_id` before fingerprint reduction, guaranteeing scheduling invariance.

## Decision Outcome

Chosen option: **Option 3 (Layered Mathematical Projection with Certified Array API Backends)**.

### Architectural Rules & Invariants

1. **Source of Truth vs. Mathematical Projection**:
   - The immutable `WorldState` graph is the canonical source of truth for state semantics, constraint evaluation, and SHA-256 fingerprinting.
   - `SoAStateTensor` is a high-performance mathematical projection representing state variables as continuous $(N, D_R)$ arrays.
   - Projection round-trips (`from_world_state` and `to_world_states`) preserve identical entity identities, metadata, and numerical values.

2. **PRNG Derivation & Threefry Determinism**:
   - All pseudo-random number streams originate from a single master seed:
     $$\text{child\_seeds} = \text{SeedSequence}(\text{seed}).\text{spawn}(N)$$
   - Each branch $i \in [0, N-1]$ consumes an independent PCG64 stream or Threefry counter key derived from `child_seeds[i]`.
   - JAX implementations exploit Threefry's counter-based PRNG architecture, which guarantees identical random bit sequences across CPU and GPU hardware.

3. **Backend Certification Hierarchy**:
   | Backend | Implementation | Certification Status | Use Case |
   |:---|:---|:---|:---|
   | **Pure-Python** | Python loop reference kernel | `CERTIFIED_BIT_IDENTICAL` (Oracle) | Canonical audit baseline |
   | **NumPy** | Array API standard (`numpy`) | `CERTIFIED_BIT_IDENTICAL` | Default local batched engine |
   | **JAX-CPU** | JAX `vmap`/`scan` Threefry | `CERTIFIED_BIT_IDENTICAL` | High-throughput compiled rollouts |
   | **JAX-GPU** | JAX GPU with Threefry PRNG | `CERTIFIED_BIT_IDENTICAL` | Accelerated certified rollouts |
   | **CuPy / CUDA** | Raw CUDA float atomics | `THROUGHPUT_ONLY` | Massive non-certified exploration |

   Any backend with non-deterministic floating-point reduction or hardware variance is strictly classified as `THROUGHPUT_ONLY` and forbidden from emitting certified audit reports.

4. **Order-Independent Distributed Reduction**:
   - In distributed computing (Ray / Dask / Multiprocessing), worker rollouts finish asynchronously out of order.
   - All executors must collect and sort `RolloutResult` items by `sample_id` ascending prior to calculating summary statistics, trajectories, and ensemble fingerprints:
     $$\mathcal{R}_{\text{reduced}} = \text{sort\_by\_sample\_id}(\{\mathcal{R}_{\text{worker}}\})$$
   - This ensures the simulation result fingerprint is mathematically invariant to task arrival order, worker latency, or cluster topology.

5. **Pure-Python Fallback & Optional Toolchains**:
   - Core EWM installation requires zero C compilers, CUDA toolchains, or heavy frameworks.
   - High-performance backends (`jax`, `cupy`, `dask`, `ray`) are configured as optional extras in `pyproject.toml` (`[jax]`, `[cuda]`, `[distributed]`).
   - If an optional accelerator is unavailable, clear diagnostic errors are surfaced with installation guidance.

## Consequences

### Positive
- **Uncompromised Auditability**: 100% bit-exact SHA-256 equivalence between a local single-sample pure-Python run and batched NumPy or JAX runs.
- **Reproducible Distributed Parallelism**: Distributed Monte Carlo rollouts under Ray and Dask produce identical results regardless of worker scheduling or cluster node count.
- **Clean Extensibility**: The Python Array API standard provides a uniform tensor abstraction across NumPy, JAX, and CuPy without vendor lock-in.

### Negative
- **Projection Overhead on Small Batches**: For $N=1$ or trivial models, projecting into SoA arrays and back carries minor memory copying overhead compared to raw in-place mutations.
- **GPU Determinism Discipline**: Requiring certified bit-identity on GPU necessitates using deterministic algorithms (such as Threefry PRNG) and avoiding non-deterministic CUDA atomic float additions.
