# Simulation-as-a-Service API Reference

This module provides concurrent background simulation runners, FastAPI REST endpoints, and orchestrator caching integrations (Track T8).

---

## Job Runners & Background Execution

::: ewm_engine.service.runner
    options:
      show_root_heading: true
      show_source: false
      members:
        - JobRunner
        - LocalJobRunner
        - JobRecord
        - JobStatus

---

## REST Service (FastAPI)

::: ewm_engine.service.fastapi_app
    options:
      show_root_heading: true
      show_source: false
      members:
        - create_simulation_app
        - SimulateRequest

---

## Orchestration Cache Adapters

::: ewm_engine.service.orchestration
    options:
      show_root_heading: true
      show_source: false
      members:
        - PrefectTaskAdapter
        - DagsterAssetAdapter
