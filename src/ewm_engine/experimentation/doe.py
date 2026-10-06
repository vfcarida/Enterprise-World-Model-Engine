"""Design of Experiments (DoE) and parameter sweep harness.

Conforms to Track T4: Core Substrate (Zero-dep, pure NumPy/stdlib).
Generates full factorial, One-At-a-Time (OAT), Latin Hypercube Sampling (LHS),
and Halton low-discrepancy designs; executes reproducible seeded rollouts;
and produces tidy result tables and tornado-diagram sensitivity rankings.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.core.world import World
from ewm_engine.durability.protocol import ResultStore
from ewm_engine.experimentation.params import ParameterSpace
from ewm_engine.simulation.engine import SimulationEngine
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import Trajectory

DesignType = Literal["full_factorial", "oat", "lhs", "halton", "custom"]


def _van_der_corput(n: int, base: int) -> list[float]:
    """Generate first n terms of Van der Corput sequence in base."""
    sequence: list[float] = []
    for i in range(1, n + 1):
        num = i
        f = 1.0 / base
        val = 0.0
        while num > 0:
            val += (num % base) * f
            num //= base
            f /= base
        sequence.append(val)
    return sequence


_PRIMES = [
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
    73,
    79,
    83,
    89,
    97,
    101,
    103,
    107,
    109,
    113,
    127,
    131,
    137,
    139,
    149,
    151,
    157,
    163,
    167,
    173,
    179,
    181,
    191,
    193,
    197,
    199,
    211,
    223,
    227,
    229,
    233,
    239,
    241,
    251,
    257,
    263,
    269,
    271,
    277,
    281,
    283,
    293,
    307,
]


def generate_halton(space: ParameterSpace, n_samples: int) -> list[dict[str, Any]]:
    """Generate low-discrepancy Halton sequence design over parameter space."""
    if space.dim > len(_PRIMES):
        raise ValueError(
            f"Halton generator supports up to {len(_PRIMES)} dimensions; got {space.dim}"
        )

    matrix = np.zeros((n_samples, space.dim), dtype=np.float64)
    for j in range(space.dim):
        matrix[:, j] = _van_der_corput(n_samples, _PRIMES[j])

    return [space.unit_to_point(matrix[i, :]) for i in range(n_samples)]


def generate_lhs(space: ParameterSpace, n_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate Latin Hypercube Sampling (LHS) design in pure NumPy."""
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_samples, space.dim), dtype=np.float64)

    for j in range(space.dim):
        # Stratified bins with uniform jitter within bin
        bins = np.arange(n_samples, dtype=np.float64)
        jitter = rng.uniform(0.0, 1.0, size=n_samples)
        coords = (bins + jitter) / n_samples
        rng.shuffle(coords)
        matrix[:, j] = coords

    return [space.unit_to_point(matrix[i, :]) for i in range(n_samples)]


def generate_full_factorial(
    space: ParameterSpace, levels_per_param: int = 3
) -> list[dict[str, Any]]:
    """Generate full factorial grid design across all parameters."""
    grid_coords: list[list[float]] = []
    for p in space.parameters:
        if p.type in ("continuous", "integer"):
            grid_coords.append(list(np.linspace(0.0, 1.0, levels_per_param)))
        elif p.type == "categorical":
            assert p.categories is not None
            k = len(p.categories)
            grid_coords.append(list((np.arange(k) + 0.5) / k))

    points: list[dict[str, Any]] = []
    for combo in itertools.product(*grid_coords):
        points.append(space.unit_to_point(combo))
    return points


def generate_oat(space: ParameterSpace, steps_per_param: int = 2) -> list[dict[str, Any]]:
    """Generate One-At-a-Time (OAT) design perturbing one parameter while fixing others at baseline."""
    baseline = space.get_baseline_point()
    points: list[dict[str, Any]] = [dict(baseline)]

    for p in space.parameters:
        if p.type in ("continuous", "integer"):
            assert p.bounds is not None
            low_val = p.bounds[0]
            high_val = p.bounds[1]
            values = (
                [low_val, high_val]
                if steps_per_param <= 2
                else list(np.linspace(low_val, high_val, steps_per_param))
            )
            for val in values:
                if val != baseline[p.name]:
                    pt = dict(baseline)
                    pt[p.name] = round(val) if p.type == "integer" else float(val)
                    points.append(pt)
        elif p.type == "categorical":
            assert p.categories is not None
            for cat in p.categories:
                if cat != baseline[p.name]:
                    pt = dict(baseline)
                    pt[p.name] = cat
                    points.append(pt)

    return points


class TornadoBar(BaseModel):
    """Sensitivity swing bar for an individual parameter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    parameter_name: str = Field(description="Parameter name.")
    baseline_value: Any = Field(description="Nominal baseline parameter value.")
    low_value: Any = Field(description="Parameter value producing the lower bound evaluation.")
    high_value: Any = Field(description="Parameter value producing the upper bound evaluation.")
    baseline_metric: float = Field(description="Metric value at nominal baseline.")
    low_metric: float = Field(description="Metric value observed at low parameter value.")
    high_metric: float = Field(description="Metric value observed at high parameter value.")
    swing: float = Field(ge=0.0, description="Absolute spread / sensitivity swing |high - low|.")


class TornadoData(BaseModel):
    """Ranked parameter sensitivity data for tornado diagram rendering."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric_name: str = Field(description="Evaluated target metric name.")
    baseline_metric: float = Field(description="Nominal baseline metric value.")
    bars: tuple[TornadoBar, ...] = Field(
        description="Ranked parameter sensitivity bars in descending swing order."
    )


class SweepResult(BaseModel):
    """Tidy outcomes and sensitivity diagnostics of a parameter sweep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    space_hash: str = Field(description="Cryptographic hash of the parameter space definition.")
    design_type: DesignType = Field(description="Design methodology used.")
    point_count: int = Field(ge=0, description="Number of design points evaluated.")
    design_points: list[dict[str, Any]] = Field(description="Input design points.")
    results_table: list[dict[str, Any]] = Field(
        description="Tidy results table containing inputs and metrics."
    )
    tornado_data: dict[str, TornadoData] = Field(
        default_factory=dict,
        description="Computed tornado diagrams keyed by metric name.",
    )
    fingerprint: str = Field(description="Canonical SHA-256 reproducibility fingerprint.")


def compute_tornado_data(
    results_table: Sequence[dict[str, Any]],
    space: ParameterSpace,
    metric_name: str,
    baseline_point: dict[str, Any] | None = None,
) -> TornadoData:
    """Compute tornado diagram sensitivity rankings for a given metric from sweep data."""
    if not results_table:
        return TornadoData(metric_name=metric_name, baseline_metric=0.0, bars=())

    base_pt = baseline_point or space.get_baseline_point()

    # Locate baseline metric
    baseline_val = 0.0
    for row in results_table:
        is_match = all(row.get(k) == v for k, v in base_pt.items() if k in row)
        if is_match and metric_name in row:
            baseline_val = float(row[metric_name])
            break
    else:
        # Default to first row or average if exact baseline not in table
        baseline_val = float(results_table[0].get(metric_name, 0.0))

    bars: list[TornadoBar] = []

    for p in space.parameters:
        if p.type in ("continuous", "integer"):
            assert p.bounds is not None
            # Find points where this parameter is min / max while other params are near baseline
            p_vals = [r[p.name] for r in results_table if p.name in r and metric_name in r]
            if not p_vals:
                continue

            min_val = min(p_vals)
            max_val = max(p_vals)

            min_metrics = [
                r[metric_name]
                for r in results_table
                if r.get(p.name) == min_val and metric_name in r
            ]
            max_metrics = [
                r[metric_name]
                for r in results_table
                if r.get(p.name) == max_val and metric_name in r
            ]

            low_m = float(np.mean(min_metrics)) if min_metrics else baseline_val
            high_m = float(np.mean(max_metrics)) if max_metrics else baseline_val
            swing = abs(high_m - low_m)

            bars.append(
                TornadoBar(
                    parameter_name=p.name,
                    baseline_value=base_pt.get(p.name, p.default),
                    low_value=min_val,
                    high_value=max_val,
                    baseline_metric=baseline_val,
                    low_metric=low_m,
                    high_metric=high_m,
                    swing=float(swing),
                )
            )

    # Sort descending by swing
    bars.sort(key=lambda b: b.swing, reverse=True)
    return TornadoData(metric_name=metric_name, baseline_metric=baseline_val, bars=tuple(bars))


def run_doe_sweep(
    world: World,
    scenario: Scenario,
    space: ParameterSpace,
    design_points: Sequence[dict[str, Any]],
    design_type: DesignType = "custom",
    metrics_extractor: Callable[[Trajectory], dict[str, float]] | None = None,
    engine: SimulationEngine | None = None,
    result_store: ResultStore | None = None,
    seed: int = 42,
) -> SweepResult:
    """Execute a reproducible DoE parameter sweep across a collection of design points."""
    sim_engine = engine or SimulationEngine()

    def default_extractor(traj: Trajectory) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for r_id, r in traj.final_state.resources.items():
            metrics[f"resource_{r_id}"] = float(r.current)
            metrics[r_id] = float(r.current)
        for s in traj.steps:
            for k, v in s.step_metrics.items():
                metrics[k] = float(v)
        return metrics

    extractor = metrics_extractor or default_extractor

    # Derive child seeds deterministically via SeedSequence
    ss = np.random.SeedSequence(seed)
    child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(len(design_points))]

    results_table: list[dict[str, Any]] = []

    for idx, (point, run_seed) in enumerate(zip(design_points, child_seeds, strict=True)):
        pt_world, pt_scenario = space.apply_to_world_and_scenario(point, world, scenario)
        pt_scenario = pt_scenario.model_copy(update={"seed": run_seed, "samples": 1})

        sim_res = sim_engine.run(pt_world, pt_scenario)
        traj = sim_res.trajectories[0]

        extracted = extractor(traj)
        row: dict[str, Any] = {
            "run_id": idx,
            "seed": run_seed,
            "trajectory_status": traj.status.value,
        }
        row.update(point)
        row.update(extracted)
        results_table.append(row)

    # Compute tornado data for all extracted numeric metrics
    metric_keys = [
        k
        for k, v in results_table[0].items()
        if isinstance(v, (int, float)) and k not in space.names and k not in ("run_id", "seed")
    ]
    tornado_dict: dict[str, TornadoData] = {}
    for m_key in metric_keys:
        tornado_dict[m_key] = compute_tornado_data(results_table, space, m_key)

    fingerprint = canonical_sha256(
        {
            "space_hash": space.space_hash,
            "design_type": design_type,
            "seed": seed,
            "point_count": len(design_points),
            "results_sample": results_table[:3],
        }
    )

    return SweepResult(
        space_hash=space.space_hash,
        design_type=design_type,
        point_count=len(design_points),
        design_points=list(design_points),
        results_table=results_table,
        tornado_data=tornado_dict,
        fingerprint=fingerprint,
    )
