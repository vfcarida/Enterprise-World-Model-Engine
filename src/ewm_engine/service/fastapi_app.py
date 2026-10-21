"""FastAPI simulation REST service (SERVE extra).

Provides idempotent, fingerprint-cached HTTP endpoints delegating execution
to JobRunner (never using FastAPI BackgroundTasks for compute).
"""

from __future__ import annotations

import importlib.util
from typing import Any

from pydantic import BaseModel, Field

from ewm_engine.core.world import World
from ewm_engine.durability.protocol import ResultStore
from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.service.runner import JobRunner
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import SimulationResult


def is_fastapi_available() -> bool:
    """Check if FastAPI is installed."""
    return importlib.util.find_spec("fastapi") is not None


class SimulateRequest(BaseModel):
    """Payload for submitting a simulation rollout."""

    scenario: Scenario = Field(description="Simulation scenario definition.")


class _AppResponse:
    """Lightweight HTTP response wrapper for zero-dependency service testing."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class BuiltInSimulationApp:
    """Zero-dependency simulation service dispatcher when FastAPI is not installed."""

    def __init__(
        self,
        runner: JobRunner,
        world: World,
        result_store: ResultStore | None = None,
    ) -> None:
        self.runner = runner
        self.world = world
        self.result_store = result_store

    def _dump_result(self, r: SimulationResult) -> dict[str, Any]:
        return {
            "fingerprint": r.fingerprint,
            "scenario_id": r.scenario.scenario_id,
            "horizon": r.scenario.horizon,
            "sample_count": len(r.trajectories),
            "provenance": r.provenance.model_dump(),
            "metrics": r.run_metrics.model_dump(),
        }

    def get(self, path: str) -> _AppResponse:
        if path == "/health":
            return _AppResponse(200, {"status": "healthy", "engine": "ewm-engine"})
        if path.startswith("/jobs/"):
            job_id = path[len("/jobs/") :]
            try:
                rec = self.runner.get_job(job_id)
                return _AppResponse(200, rec.model_dump())
            except KeyError:
                return _AppResponse(404, {"detail": f"Job '{job_id}' not found."})
        if path.startswith("/results/"):
            fp = path[len("/results/") :]
            res = self.runner.get_result(fp)
            if res is not None:
                return _AppResponse(200, self._dump_result(res))
            if self.result_store is not None:
                stored = self.result_store.get(fp)
                if stored is not None:
                    return _AppResponse(200, self._dump_result(stored))
            return _AppResponse(
                404, {"detail": f"Simulation result for fingerprint '{fp}' not found."}
            )
        return _AppResponse(404, {"detail": f"Route '{path}' not found."})

    def post(self, path: str, json: dict[str, Any]) -> _AppResponse:
        if path == "/simulate":
            try:
                req = SimulateRequest.model_validate(json)
                job_id = self.runner.submit_job(self.world, req.scenario)
                rec = self.runner.get_job(job_id)
                return _AppResponse(
                    200,
                    {
                        "job_id": rec.job_id,
                        "fingerprint": rec.fingerprint,
                        "status": rec.status.value,
                        "scenario_id": rec.scenario_id,
                    },
                )
            except Exception as exc:
                return _AppResponse(400, {"detail": str(exc)})
        return _AppResponse(404, {"detail": f"Route '{path}' not found."})


def create_simulation_app(
    runner: JobRunner,
    world: World,
    result_store: ResultStore | None = None,
    require_extra: bool = False,
) -> Any:
    """Factory creating a FastAPI application (or zero-dependency dispatcher) backed by JobRunner."""
    if require_extra and not is_fastapi_available():
        raise SimulationConfigurationError(
            "FastAPI is required for create_simulation_app. Install via `pip install ewm-engine[serve]`."
        )

    if not is_fastapi_available():
        return BuiltInSimulationApp(runner=runner, world=world, result_store=result_store)

    fastapi_mod = importlib.import_module("fastapi")
    FastAPI = fastapi_mod.FastAPI
    HTTPException = fastapi_mod.HTTPException

    app = FastAPI(
        title="Enterprise World Model Engine - Simulation API",
        version="1.5.0",
        description="Idempotent, fingerprint-cached simulation service.",
    )

    def _dump_result(r: SimulationResult) -> dict[str, Any]:
        return {
            "fingerprint": r.fingerprint,
            "scenario_id": r.scenario.scenario_id,
            "horizon": r.scenario.horizon,
            "sample_count": len(r.trajectories),
            "provenance": r.provenance.model_dump(),
            "metrics": r.run_metrics.model_dump(),
        }

    @app.get("/health")  # type: ignore[untyped-decorator]
    def health_check() -> dict[str, str]:
        return {"status": "healthy", "engine": "ewm-engine"}

    @app.post("/simulate")  # type: ignore[untyped-decorator]
    def submit_simulation(req: SimulateRequest) -> dict[str, Any]:
        """Submit scenario to JobRunner and return tracking job_id and predicted fingerprint."""
        try:
            job_id = runner.submit_job(world, req.scenario)
            rec = runner.get_job(job_id)
            return {
                "job_id": rec.job_id,
                "fingerprint": rec.fingerprint,
                "status": rec.status.value,
                "scenario_id": rec.scenario_id,
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/jobs/{job_id}")  # type: ignore[untyped-decorator]
    def get_job_status(job_id: str) -> dict[str, Any]:
        """Retrieve execution status and metadata for a job."""
        try:
            rec = runner.get_job(job_id)
            return rec.model_dump()
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.") from None

    @app.get("/results/{fingerprint}")  # type: ignore[untyped-decorator]
    def get_simulation_result(fingerprint: str) -> dict[str, Any]:
        """Retrieve completed simulation result by canonical SHA-256 fingerprint."""
        res = runner.get_result(fingerprint)
        if res is not None:
            return _dump_result(res)
        if result_store is not None:
            stored = result_store.get(fingerprint)
            if stored is not None:
                return _dump_result(stored)

        raise HTTPException(
            status_code=404, detail=f"Simulation result for fingerprint '{fingerprint}' not found."
        )

    return app
