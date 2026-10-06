"""Local JobRunner and concurrency protocols for simulation-as-a-service.

Provides asynchronous, memoized simulation job execution keyed by SeedSequence
and cached via P13 ResultStore.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import threading
import uuid
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.world import World
from ewm_engine.durability.memoize import compute_simulation_fingerprint
from ewm_engine.durability.protocol import ResultStore
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import SimulationResult


class JobStatus(StrEnum):
    """Lifecycle status of an asynchronous simulation job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobRecord(BaseModel):
    """Tracking record for a simulation execution job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(description="Unique identifier for the job.")
    scenario_id: str = Field(description="Scenario identifier.")
    fingerprint: str = Field(description="Predicted or evaluated simulation fingerprint.")
    status: JobStatus = Field(description="Current job lifecycle status.")
    created_at: str = Field(description="ISO timestamp of job creation.")
    finished_at: str | None = Field(default=None, description="ISO timestamp of completion.")
    result_summary: dict[str, Any] | None = Field(
        default=None, description="Summary of simulation outcomes upon completion."
    )
    error_message: str | None = Field(default=None, description="Error details if failed.")


@runtime_checkable
class JobRunner(Protocol):
    """Protocol for asynchronous simulation job runners."""

    def submit_job(self, world: World, scenario: Scenario) -> str:
        """Submit a simulation job and return its job_id."""
        ...

    def get_job(self, job_id: str) -> JobRecord:
        """Retrieve tracking record for a job."""
        ...

    def get_result(self, fingerprint: str) -> SimulationResult | None:
        """Retrieve cached result by canonical fingerprint."""
        ...

    def cancel_job(self, job_id: str) -> bool:
        """Attempt to cancel an ongoing or pending job."""
        ...


class LocalJobRunner(JobRunner):
    """Zero-dependency local concurrent job runner with exact ResultStore caching."""

    def __init__(
        self,
        max_workers: int = 4,
        result_store: ResultStore | None = None,
        engine: SimulationEngine | None = None,
    ) -> None:
        self.max_workers = max_workers
        self.result_store = result_store
        self.engine = engine or SimulationEngine()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._results: dict[str, SimulationResult] = {}
        self._futures: dict[str, concurrent.futures.Future[SimulationResult]] = {}
        self._lock = threading.Lock()

    def submit_job(self, world: World, scenario: Scenario) -> str:
        """Submit simulation job, checking ResultStore for instant cache hits."""
        fp = compute_simulation_fingerprint(world, scenario)
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.UTC).isoformat()

        # Check ResultStore cache hit
        if self.result_store is not None:
            cached_res = self.result_store.get(fp)
            if cached_res is not None:
                with self._lock:
                    self._results[fp] = cached_res
                    rec = JobRecord(
                        job_id=job_id,
                        scenario_id=scenario.scenario_id,
                        fingerprint=fp,
                        status=JobStatus.COMPLETED,
                        created_at=now,
                        finished_at=now,
                        result_summary={
                            "trajectories_count": len(cached_res.trajectories),
                            "cached": True,
                        },
                    )
                    self._jobs[job_id] = rec
                return job_id

        # Submit to background executor
        with self._lock:
            rec = JobRecord(
                job_id=job_id,
                scenario_id=scenario.scenario_id,
                fingerprint=fp,
                status=JobStatus.PENDING,
                created_at=now,
            )
            self._jobs[job_id] = rec

        future = self._executor.submit(self._execute_simulation, job_id, world, scenario, fp)
        with self._lock:
            self._futures[job_id] = future

        return job_id

    def _execute_simulation(
        self,
        job_id: str,
        world: World,
        scenario: Scenario,
        fp: str,
    ) -> SimulationResult:
        """Background worker executing the simulation rollout."""
        with self._lock:
            old_rec = self._jobs[job_id]
            self._jobs[job_id] = old_rec.model_copy(update={"status": JobStatus.RUNNING})

        try:
            # SeedSequence spawn guarantees reproducibility across threads
            ss = np.random.SeedSequence(scenario.seed)
            worker_seed = int(ss.spawn(1)[0].generate_state(1)[0])
            seeded_sc = scenario.model_copy(update={"seed": worker_seed})

            res = self.engine.run(world, seeded_sc)
            finished_now = datetime.datetime.now(datetime.UTC).isoformat()

            with self._lock:
                self._results[fp] = res
                if self.result_store is not None:
                    self.result_store.put(fp, res)

                self._jobs[job_id] = self._jobs[job_id].model_copy(
                    update={
                        "status": JobStatus.COMPLETED,
                        "finished_at": finished_now,
                        "result_summary": {
                            "trajectories_count": len(res.trajectories),
                            "cached": False,
                        },
                    }
                )
            return res
        except Exception as exc:
            finished_now = datetime.datetime.now(datetime.UTC).isoformat()
            with self._lock:
                self._jobs[job_id] = self._jobs[job_id].model_copy(
                    update={
                        "status": JobStatus.FAILED,
                        "finished_at": finished_now,
                        "error_message": str(exc),
                    }
                )
            raise

    def get_job(self, job_id: str) -> JobRecord:
        """Retrieve current record of a job."""
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Job with id '{job_id}' not found.")
            return self._jobs[job_id]

    def get_result(self, fingerprint: str) -> SimulationResult | None:
        """Retrieve simulation result by fingerprint."""
        with self._lock:
            if fingerprint in self._results:
                return self._results[fingerprint]
        if self.result_store is not None:
            return self.result_store.get(fingerprint)
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        with self._lock:
            if job_id not in self._jobs:
                return False
            fut = self._futures.get(job_id)
            if fut is not None and fut.cancel():
                self._jobs[job_id] = self._jobs[job_id].model_copy(
                    update={"status": JobStatus.CANCELLED}
                )
                return True
        return False

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown underlying thread pool executor."""
        self._executor.shutdown(wait=wait)
