# ADR-013: Observability Layer, Lifecycle Hook Protocol, and Programmatic RunMetrics

## Status
Accepted

## Date
2026-10-02

## Context
Enterprise World Model Engine (EWM Engine) requires rich, extensible runtime visibility into simulation lifecycles, constraint evaluations, discrete step transitions, and Monte Carlo rollout batches. Observability is essential for host applications, experiment runners, distributed orchestrators, and enterprise monitoring stacks.

However, EWM Engine's core kernel is strictly domain-independent and lightweight. Introducing heavy telemetry dependencies (such as the OpenTelemetry SDK, Prometheus clients, or MLflow) into the mandatory package requirements would violate the zero-external-dependencies principle, add significant installation overhead, and complicate embedded deployments.

Furthermore, observational components must not violate core simulation invariants:
1. **Determinism and Reproducibility (AC-004)**: Registering hooks or monitoring agents must never alter the pseudo-random stream, state progression, or simulation trajectory.
2. **Fault Isolation**: A buggy, slow, or throwing monitoring hook must never crash an ongoing simulation experiment.
3. **Epistemic Honesty and Data Minimization**: Runtime metrics must expose operational and performance counters without leaking internal entity payloads, secrets, or unverified causal claims.

## Decision

### 1. Observational Lifecycle Hook Protocol
We introduce `ewm_engine.hooks.protocol.Hook` as a runtime-checkable Protocol defining a single observation method:
```python
@runtime_checkable
class Hook(Protocol):
    def on_event(self, event: HookEvent) -> None: ...
```
Any object providing `on_event(event: HookEvent) -> None`, or any single-argument callable function, can be registered with a `HookRegistry`.

### 2. Immutable Lifecycle Event Hierarchy
Five typed, frozen lifecycle events are emitted by `SimulationEngine` during execution:
- `SimulationStarted`: Emitted when `SimulationEngine.run` begins, carrying scenario identifiers, horizon, sample count, seed, and initial state fingerprint.
- `ConstraintEvaluated`: Emitted each time an individual constraint rule is evaluated in pre-action or post-transition phase, carrying rule severity, satisfaction status, and evaluation results.
- `StepCompleted`: Emitted after each discrete step transition, carrying step index, pre-transition state hash, step record, and invalidation flag.
- `RolloutCompleted`: Emitted upon rollout completion or early invalidation, carrying trajectory status and step count.
- `SimulationFinished`: Emitted when all rollouts conclude, carrying final `SimulationResult`, summary `RunMetrics`, and wall-clock execution duration.

### 3. Fault Isolation and Zero-Overhead Policy
- **Fault Isolation**: `HookRegistry.emit(event)` dispatches events to each registered hook within a guarded `try...except Exception` block. Any exception is caught, logged with full traceback via `logging.getLogger("ewm_engine.hooks")`, and swallowed so simulation execution proceeds unhindered.
- **Zero Overhead**: When no hooks are registered (`len(hooks) == 0`), event dispatch returns immediately with zero measurable overhead.
- **Observational Only**: Hooks have read-only access to deeply immutable `WorldState` instances and frozen event models.

### 4. Library-Correct Logging and No Print Invariant
- The package root logger `ewm_engine` is configured with `logging.NullHandler()` at import time to prevent spurious "No handler found" warnings in host applications.
- Zero `print()` statements are permitted anywhere in library code (`src/ewm_engine/**`), verified by an automated AST-scanning test (`tests/unit/test_no_print.py`).

### 5. Programmatic `RunMetrics`
A dedicated, typed `RunMetrics` model is captured across all rollouts and attached to `SimulationResult.run_metrics` and `Provenance.runtime_metadata["metrics"]`:
- `simulation_duration_seconds: float`
- `rollout_count: int`
- `completed_rollout_count: int`
- `invalid_rollout_count: int`
- `step_count: int`
- `constraint_evaluation_count: int`
- `hard_violation_count: int`
- `soft_violation_count: int`
- `dynamics_transition_count: int`
- `trace_edge_count: int`

### 6. Future OpenTelemetry & Telemetry Adapters
An OpenTelemetry adapter is declared as a **FUTURE optional extra package** (e.g. `ewm-engine-otel`). The core engine provides the hook event protocol; the adapter implements `Hook` to map events to OpenTelemetry spans, metrics, and trace contexts. The OpenTelemetry SDK and collector clients remain entirely outside the core library.

## Consequences
- Clean, decoupled lifecycle observability with zero mandatory dependencies.
- Host applications can attach logging, progress bars, or metrics collectors via simple callbacks or `Hook` classes.
- Bitwise determinism and branch isolation remain preserved with 100% mathematical equality between hooked and unhooked runs.
