# Simulation-as-a-Service: LocalJobRunner & FastAPI Serving

This guide demonstrates how to execute asynchronous simulation jobs using `LocalJobRunner`, serve simulation endpoints over REST with FastAPI, and configure orchestrator cache keys for Prefect and Dagster.

---

## 1. Concurrency with LocalJobRunner

`LocalJobRunner` manages concurrent simulations with thread pools or process pools, guarantees bitwise reproducible seeding via `SeedSequence.spawn()`, and avoids redundant computation via exact `ResultStore` caching:

```python
from ewm_engine.service import LocalJobRunner
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.core.world import World

runner = LocalJobRunner(max_workers=4)

# Submit simulation asynchronously
job = runner.submit_simulation(world=world, scenario=scenario)
print(f"Job submitted with ID: {job.job_id}")

# Wait for completion or poll status
completed_job = runner.wait_for_job(job.job_id, timeout=30.0)
print(f"Status: {completed_job.status}, Fingerprint: {completed_job.fingerprint}")

# Fetch result
result = runner.get_result(completed_job.fingerprint)
```

---

## 2. FastAPI REST Serving

Expose EWM Engine over high-performance HTTP endpoints using `[serve]` (FastAPI & Starlette):

```python
from ewm_engine.service import create_simulation_app

# Create FastAPI application instance
app = create_simulation_app()

# Run with uvicorn:
# uvicorn app:app --host 0.0.0.0 --port 8000
```

### Endpoints Available:
- **`POST /simulate`**: Accepts scenario specification JSON; returns job ID, status, and predicted fingerprint.
- **`GET /jobs/{job_id}`**: Returns current execution status (`pending`, `running`, `completed`, `failed`).
- **`GET /results/{fingerprint}`**: Returns cached `SimulationResult` payload by canonical fingerprint.

---

## 3. Workflow Orchestration (Prefect & Dagster)

EWM canonical fingerprints provide the optimal cache key for workflow orchestrators:

```python
from ewm_engine.service.orchestration import get_prefect_cache_key, get_dagster_cache_key

# Deterministic cache key based on world initial state, rules, and scenario
cache_key = get_prefect_cache_key(world=world, scenario=scenario)
print("Prefect cache key:", cache_key)
```
