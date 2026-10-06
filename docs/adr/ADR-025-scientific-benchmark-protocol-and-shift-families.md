# ADR-025: Scientific Benchmark Protocol and Structural Shift Families

## Status
Accepted

## Date
2026-10-05

## Context
A convergent finding across 2024–2026 AI research is that surface or observational predictive accuracy does not imply systemic understanding (Physics-IQ, arXiv:2501.09038; Action-Conditioned World Models critique, arXiv:2610.00012). Models that perform well on in-distribution forecasting frequently fail catastrophically when an organization enacts structural interventions, modifies business rules, or operates under acute constraint stress.

The authoritative specification (ewm.md §63) reserves five scientific benchmark families designed to measure how any `DynamicsModel` or decision policy behaves under structural distribution shift. Rather than creating a standalone benchmark product, the engine requires a clean, typed protocol in `benchmarks/` to serve as a reproducible **measurement instrument** for world-model research.

## Decision

### 1. Benchmark Harness Protocol (`benchmarks/protocol.py`)
We define a lightweight, typed benchmark protocol in `benchmarks/`:
- `BenchmarkCase`: Declarative specification combining `(world_factory, scenario_family, metric_set, reference_result)`.
- `BenchmarkFamily`: Protocol providing `name`, `description`, `generate_cases()`, and `evaluate(model)`.
- `BenchmarkReport`: Machine-readable, frozen Pydantic model capturing execution metrics, pass/fail thresholds, and cryptographic state fingerprints.

### 2. Five Canonical Shift Families (`benchmarks/families/`)
We implement the five synthetic benchmark families specified in ewm.md §63:

1. **`InterventionShiftBenchmark`**:
   - **Probe:** Measures model generalization when intervention actions diverge from training distributions ($a_{\text{eval}} \notin \text{Support}(A_{\text{train}})$).
   - **Metric:** Interventional generalization gap ($\Delta MAE = MAE_{\text{interventional}} - MAE_{\text{in-distribution}}$).
2. **`RuleShiftBenchmark`**:
   - **Probe:** Evaluates whether dynamics adapt when organizational rules or tax/transit laws shift dynamically mid-rollout.
   - **Metric:** Error jump immediately following rule activation step.
3. **`ConstraintStressBenchmark`**:
   - **Probe:** Pushes resource levels toward critical boundaries ($[min, max]$) under heavy stochastic shocks.
   - **Metric:** Rate of illegal state transitions and invariant breach violations.
4. **`LongHorizonBenchmark`**:
   - **Probe:** Steps rollouts over extended horizons ($H \ge 100$ steps) to measure compounding autoregressive drift and variance explosion.
   - **Metric:** Horizon error scaling coefficient $\alpha$ ($Error(t) \sim t^\alpha$).
5. **`MultiAgentCascadeBenchmark`**:
   - **Probe:** Evaluates systemic cascade failures across interdependent decision actors and supply depots.
   - **Metric:** Cascade propagation velocity and systemic unserved demand fraction.

### 3. Reproducibility & FAIR Principles
Every benchmark execution:
- Seeds PRNGs deterministically via NumPy `SeedSequence`.
- Records component versions and environment metadata.
- Outputs machine-readable JSON reports matching the Croissant metadata specification for FAIR scientific reproducibility.

### 4. Non-Blocking CI Governance
Benchmarks are research instruments and performance monitors. They run as automated verification tests (`tests/benchmark/test_scientific_benchmarks.py`) to verify execution reproducibility, but throughput and research scores do not block functional library releases.

## Consequences
- **Positive:** Positions EWM Engine as a standard scientific evaluation instrument for world-model research.
- **Positive:** Fully reproducible and seed-locked across runs and machines.
- **Positive:** Keeps research benchmark suites cleanly partitioned in `benchmarks/` without polluting `src/ewm_engine`.
