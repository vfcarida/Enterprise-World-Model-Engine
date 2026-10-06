# ADR-024: OpenTelemetry Observability Adapter (API-Only & Zero Core Coupling)

## Status
Accepted

## Date
2026-10-05

## Context
Production-grade simulation workflows in enterprise environments require standard distributed tracing and metric telemetry. Observability engineers need to monitor long-running Monte Carlo experiments, trace the execution latency of individual discrete steps, record constraint violation rates, and inspect simulation lifecycle events in APM platforms (such as Datadog, Honeycomb, Grafana Tempo, or Jaeger).

However, introducing OpenTelemetry into the core kernel of the Enterprise World Model Engine (EWM Engine) presents significant risks:
1. **Dependency Bloat**: The OpenTelemetry SDK and its gRPC/protobuf export pipeline are heavy dependencies that violate the core engine's zero-external-dependencies principle (AC-019).
2. **Library vs. Application Responsibilities**: Standard OpenTelemetry guidance for libraries dictates that libraries should only depend on the OpenTelemetry API, never initialize the SDK or configure exporters. The host application must own SDK initialization, resource attribution, sampling policies, and backend exporters.
3. **Execution Invariants**: Monitoring hooks must never alter simulation determinism (AC-004), mutate state (AC-005), or fail an ongoing simulation run if telemetry transmission fails.

## Decision

### 1. API-Only Observability Adapter (`OpenTelemetryHookListener`)
We implement [`ewm_engine.integrations.otel.OpenTelemetryHookListener`](file:///c:/Users/vinicius/Documents/GeminiCodes/Enterprise-World-Model-Engine/src/ewm_engine/integrations/otel.py) as a dedicated adapter that implements the existing, frozen `Hook` protocol (ADR-013).

The adapter:
- Binds to the engine exclusively via `HookRegistry.register(listener)`.
- Uses only `opentelemetry.trace` and `opentelemetry.metrics` API packages.
- Never imports or instantiates the OpenTelemetry SDK or exporter backends.
- Leaves tracer and meter provider configuration entirely to the host application or test fixture.

### 2. Hierarchical Span Mapping
Lifecycle events emitted by `SimulationEngine` map to an intuitive nested span hierarchy:
- `SimulationStarted` $\to$ Root span: `ewm.simulation` (attributes: `scenario.id`, `scenario.seed`, `scenario.samples`, `scenario.horizon`).
- `SimulationEngine` rollout execution $\to$ Child span: `ewm.rollout` (attributes: `sample_id`, `trajectory.status`).
- `StepCompleted` $\to$ Child span: `ewm.step` (attributes: `step.index`, `actions.count`, `is_invalid`).
- `ConstraintEvaluated` $\to$ Span event on current step: `constraint_evaluated` (attributes: `constraint.id`, `phase`, `severity`, `satisfied`).

### 3. Metric Instruments & `RunMetrics` Mirroring
The adapter establishes a Meter (`ewm.engine`) with standard telemetry instruments:
- `ewm.simulations.total` (Counter): Total simulations started/finished.
- `ewm.rollouts.total` (Counter): Completed and invalid rollout counts.
- `ewm.steps.total` (Counter): Total discrete simulation steps transitioned.
- `ewm.constraint_violations.total` (Counter): Hard and soft invariant violations flagged.
- `ewm.simulation.duration` (Histogram): Wall-clock execution duration in seconds.

### 4. Graceful Degradation & No-Op Fallback
When the optional `otel` extra is not installed:
- `ewm_engine.integrations.otel` can be imported safely without throwing an immediate `ImportError`.
- If OpenTelemetry is absent, `OpenTelemetryHookListener` functions as a silent no-op listener with zero runtime overhead and zero impact on simulation execution.

### 5. Packaging & Optional Extra
The OpenTelemetry adapter is packaged under the optional dependency extra:
```bash
pip install 'ewm-engine[otel]'
```
The core `pyproject.toml` dependencies remain completely free of OpenTelemetry packages.

## Consequences
- **Positive:** Full compatibility with modern enterprise observability stacks without adding a single byte to the core package.
- **Positive:** Complies strictly with the OpenTelemetry library specification.
- **Positive:** Preserves bitwise simulation determinism and fault isolation (ADR-013).
- **Negative:** Telemetry spans are only captured if the host application configures an active OpenTelemetry SDK TracerProvider.
