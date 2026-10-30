# Dataset Card: AIOps Microservice Telemetry (Prometheus, Jaeger & Logs)

**Author:** Vinícius Caridá  
**Ingestion Modules:** `ewm_engine.aiops.data.PrometheusCollector`, `ewm_engine.aiops.data.JaegerCollector`, `ewm_engine.aiops.data.LogParser`  
**Maturity Tier:** **Experimental (`Synthetic Telemetry Schema & Collectors`)**  
**License:** Apache-2.0

> [!WARNING]
> **Scope & Data Availability Notice (`EWM-007` Audit):** This card documents the telemetry schema consumed by `ewm_engine.aiops.data`. Live production Prometheus/Jaeger scraping is disabled in repository test suites; all tests run offline against deterministic synthetic microservice failure traces.

---

## 1. Dataset Description

- **Domain:** Multimodal cloud-native Kubernetes microservice observability combining 1-minute time-series metrics, distributed tracing call graphs, and unstructured container logs.
- **Modalities & Channels:**
  1. **Prometheus Metrics (`PrometheusCollector`, 5 channels at `1m` step):**
     - `cpu`: `container_cpu_usage_seconds_total`
     - `memory`: `container_memory_working_set_bytes`
     - `error_rate`: `http_requests_total{status=~"5.."}`
     - `latency_p95`: `http_request_duration_seconds{quantile="0.95"}`
     - `active_connections`: Active socket/RPC concurrency
  2. **Jaeger Call Graph (`JaegerCollector`):**
     - Directed edges `(parent_service, child_service, avg_latency_ms)` extracted from `/api/dependencies`.
  3. **Unstructured Application Logs (`LogParser`):**
     - Raw container log streams pre-filtered via regex (`ERROR`, `FATAL`, `Exception`, `TimeoutException`, `OutOfMemoryError`, `CrashLoopBackOff`) and converted into `CausalInterventionVector` objects when high-severity faults occur.

## 2. Bitemporal & Privacy Guarantees

- All Prometheus metric queries attach `event_time` (scrape timestamp) and `knowledge_time` (query execution timestamp) with `knowledge_time >= event_time`.
- Log parsing strips or ignores personally identifiable information (PII) and credentials, extracting only service topology names, directional degradation sentiment, and contagion pathways.
