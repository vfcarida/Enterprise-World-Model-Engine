"""Simulation-as-a-service and orchestration module (T8).

Provides JobRunner protocol, concurrent LocalJobRunner, FastAPI REST service,
and Prefect/Dagster orchestration cache adapters.
"""

from __future__ import annotations

from ewm_engine.service.fastapi_app import (
    SimulateRequest,
    create_simulation_app,
    is_fastapi_available,
)
from ewm_engine.service.orchestration import (
    DagsterAssetAdapter,
    PrefectTaskAdapter,
    is_dagster_available,
    is_prefect_available,
)
from ewm_engine.service.runner import (
    JobRecord,
    JobRunner,
    JobStatus,
    LocalJobRunner,
)

__all__ = [
    "DagsterAssetAdapter",
    "JobRecord",
    "JobRunner",
    "JobStatus",
    "LocalJobRunner",
    "PrefectTaskAdapter",
    "SimulateRequest",
    "create_simulation_app",
    "is_dagster_available",
    "is_fastapi_available",
    "is_prefect_available",
]
