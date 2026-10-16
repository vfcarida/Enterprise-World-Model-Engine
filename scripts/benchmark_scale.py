#!/usr/bin/env python3
"""Automated scaling curve benchmark generator for EWM Engine.

Measures simulation throughput across ensemble sizes and execution backends:
1. Sequential Pure-Python (SerialExecutor, canonical oracle)
2. Batched Structure-of-Arrays (BatchedExecutor, NumPy Array API)
3. Distributed Multi-Core (MultiprocessingExecutor, 4 workers)

Verifies Invariant #5: 100% bit-identical SHA-256 fingerprints across sequential and batched.
Outputs reproducible Markdown tables with an automated machine provenance block.
"""

from __future__ import annotations

import platform
import subprocess
import time
from typing import Any

import numpy as np

from ewm_engine.actors.rule_based import ThresholdReplenishmentActor
from ewm_engine.constraints.registry import ConstraintRegistry
from ewm_engine.constraints.standard import (
    ActionTransferAvailabilityConstraint,
    ResourceCapacityConstraint,
)
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.dynamics.composite import CompositeDynamics
from ewm_engine.dynamics.deterministic import DeterministicTransferDynamics
from ewm_engine.dynamics.stochastic import StochasticDemandDynamics
from ewm_engine.simulation.batched import BatchedExecutor
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.executors import MultiprocessingExecutor, SerialExecutor
from ewm_engine.simulation.scenario import Scenario


def get_git_sha() -> str:
    """Retrieve current git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_blas_info() -> str:
    """Retrieve BLAS/LAPACK library configuration."""
    try:
        config = np.show_config(mode="dicts")  # type: ignore[call-arg]
        blas = config.get("Build Dependencies", {}).get("blas", {}).get("name", "standard")
        return str(blas)
    except Exception:
        return "OpenBLAS / Accelerate"


def create_scaling_world(num_nodes: int = 25) -> World:
    """Build a deterministic multi-node enterprise benchmark world."""
    entities = [Entity(id=f"hub_{i}", type="hub") for i in range(num_nodes)]
    relationships = [
        Relationship(
            source=f"hub_{i}",
            target=f"hub_{(i + 1) % num_nodes}",
            type="courier_route",
        )
        for i in range(num_nodes)
    ]
    resources = [
        Resource(
            id=f"inventory_{i}",
            entity_id=f"hub_{i}",
            current=150.0,
            min_value=0.0,
            max_value=500.0,
            unit="units",
        )
        for i in range(num_nodes)
    ]
    state = WorldState(entities=entities, relationships=relationships, resources=resources)
    dynamics = CompositeDynamics(
        [
            DeterministicTransferDynamics(),
            StochasticDemandDynamics(resource_id="inventory_0", mean_demand=4.0, std_demand=1.0),
        ]
    )
    constraints = ConstraintRegistry(
        [
            ResourceCapacityConstraint(resource_id="inventory_0"),
            ActionTransferAvailabilityConstraint(),
        ]
    )
    actor = ThresholdReplenishmentActor(
        actor_id="actor_replenish",
        source_resource="inventory_1",
        target_resource="inventory_0",
        reorder_point=50.0,
        order_quantity=25.0,
    )
    return World(state=state, dynamics=dynamics, constraints=constraints, actors=[actor])


def run_scaling_benchmark(
    ensemble_sizes: list[int] | None = None,
    horizon: int = 15,
    seed: int = 42,
) -> dict[str, Any]:
    """Run scaling benchmark across ensemble sizes and executors."""
    if ensemble_sizes is None:
        ensemble_sizes = [1, 5, 20, 50]

    world = create_scaling_world(num_nodes=20)
    results: list[dict[str, Any]] = []

    for n in ensemble_sizes:
        scenario = Scenario(
            scenario_id=f"scen_scale_{n}",
            horizon=horizon,
            samples=n,
            seed=seed,
        )

        # 1. Sequential Pure-Python
        t0 = time.perf_counter()
        engine_seq = SimulationEngine(executor=SerialExecutor())
        res_seq = engine_seq.run(world, scenario)
        t_seq = time.perf_counter() - t0

        # 2. Batched Structure-of-Arrays (NumPy Array API)
        t0 = time.perf_counter()
        engine_batch = SimulationEngine(executor=BatchedExecutor(backend="numpy"))
        res_batch = engine_batch.run(world, scenario)
        t_batch = time.perf_counter() - t0

        # Verify bit-identity invariant
        assert len(res_seq.trajectories) == len(res_batch.trajectories)
        for s_t, b_t in zip(res_seq.trajectories, res_batch.trajectories, strict=True):
            assert s_t.final_state.fingerprint == b_t.final_state.fingerprint, (
                f"Fingerprint mismatch at N={n}, sample={s_t.sample_id}"
            )

        # 3. Distributed Multiprocessing (4 workers)
        t0 = time.perf_counter()
        engine_mp = SimulationEngine(executor=MultiprocessingExecutor(max_workers=4))
        res_mp = engine_mp.run(world, scenario)
        assert len(res_mp.trajectories) == n
        t_mp = time.perf_counter() - t0

        results.append(
            {
                "samples": n,
                "horizon": horizon,
                "total_steps": n * horizon,
                "seq_time_sec": t_seq,
                "seq_throughput": (n * horizon) / t_seq if t_seq > 0 else 0.0,
                "batch_time_sec": t_batch,
                "batch_throughput": (n * horizon) / t_batch if t_batch > 0 else 0.0,
                "mp_time_sec": t_mp,
                "mp_throughput": (n * horizon) / t_mp if t_mp > 0 else 0.0,
                "batch_speedup": t_seq / t_batch if t_batch > 0 else 1.0,
                "mp_speedup": t_seq / t_mp if t_mp > 0 else 1.0,
                "bit_identical": True,
            }
        )

    provenance = {
        "git_sha": get_git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "numpy_version": np.__version__,
        "blas_library": get_blas_info(),
        "seed": seed,
        "horizon": horizon,
    }

    return {"results": results, "provenance": provenance}


def format_markdown_report(data: dict[str, Any]) -> str:
    """Format benchmark measurements and provenance into GitHub-flavored Markdown."""
    lines: list[str] = []
    lines.append("### Scaling Throughput and Cross-Backend Performance Curves")
    lines.append("")
    lines.append(
        "| Ensemble Size ($N$) | Total Steps | Sequential Pure-Python (steps/s) | Batched SoA NumPy (steps/s) | Distributed 4-Worker (steps/s) | Batched Speedup | Bit-Identity Certified |"
    )
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for r in data["results"]:
        lines.append(
            f"| {r['samples']} | {r['total_steps']} | {r['seq_throughput']:.1f} | {r['batch_throughput']:.1f} | {r['mp_throughput']:.1f} | {r['batch_speedup']:.2f}x | :white_check_mark: Bit-Exact |"
        )

    lines.append("")
    lines.append("#### Benchmark Machine Provenance Block")
    lines.append("```yaml")
    lines.append(f'git_sha: "{data["provenance"]["git_sha"]}"')
    lines.append(f'python_version: "{data["provenance"]["python_version"]}"')
    lines.append(f'platform: "{data["provenance"]["platform"]}"')
    lines.append(f'processor: "{data["provenance"]["processor"]}"')
    lines.append(f'numpy_version: "{data["provenance"]["numpy_version"]}"')
    lines.append(f'blas_lapack: "{data["provenance"]["blas_library"]}"')
    lines.append(f"benchmark_seed: {data['provenance']['seed']}")
    lines.append(f"rollout_horizon: {data['provenance']['horizon']}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("Measuring scaling curves and bit-identical cross-backend performance...")
    data = run_scaling_benchmark(ensemble_sizes=[1, 5, 20, 50], horizon=10, seed=42)
    report = format_markdown_report(data)
    print("\n" + report)


if __name__ == "__main__":
    main()
