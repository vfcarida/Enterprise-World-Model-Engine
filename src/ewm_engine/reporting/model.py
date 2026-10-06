"""Renderer-neutral data models for simulation reporting and visualization.

Pure data representations (Pydantic / stdlib / NumPy) capturing trajectories,
uncertainty quantiles, branch trees, and verification events.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.constraints.results import ConstraintSeverity
from ewm_engine.simulation.trajectory import SimulationResult


class SignalQuantiles(BaseModel):
    """Empirical quantiles and moments for a timeseries across Monte Carlo samples."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_name: str = Field(description="Name of the resource or metric.")
    steps: list[int] = Field(description="Time step indices.")
    p10: list[float] = Field(description="10th percentile timeseries.")
    p25: list[float] = Field(description="25th percentile timeseries.")
    p50: list[float] = Field(description="Median (50th percentile) timeseries.")
    p75: list[float] = Field(description="75th percentile timeseries.")
    p90: list[float] = Field(description="90th percentile timeseries.")
    mean: list[float] = Field(description="Mean timeseries.")


class BranchTreeNode(BaseModel):
    """Node in an event-sourced or scenario branching tree."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(description="Unique node identifier.")
    parent_id: str | None = Field(default=None, description="Parent node identifier.")
    step: int = Field(ge=0, description="Step index of this node.")
    state_fingerprint: str = Field(description="Canonical fingerprint of the state.")
    branch_tag: str = Field(default="main", description="Branch tag or intervention name.")


class ReportModel(BaseModel):
    """Renderer-neutral simulation report model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(default="1.0.0", description="Report schema version.")
    simulation_id: str = Field(description="Identifier of the simulation run.")
    fingerprint: str = Field(description="Canonical SHA-256 fingerprint of the simulation result.")
    scenario_id: str = Field(description="Scenario identifier.")
    horizon: int = Field(ge=1, description="Simulation horizon steps.")
    sample_count: int = Field(ge=1, description="Number of Monte Carlo rollout samples.")
    signals: dict[str, SignalQuantiles] = Field(
        default_factory=dict, description="Extracted signal quantiles."
    )
    branch_tree: list[BranchTreeNode] = Field(
        default_factory=list, description="Scenario branching DAG nodes."
    )
    violations_summary: list[dict[str, Any]] = Field(
        default_factory=list, description="Constraint violation events."
    )
    verification_verdicts: list[dict[str, Any]] = Field(
        default_factory=list, description="Temporal STL or Oracle-graph verification logs."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Simulation and engine metadata."
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to normalized dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: str | Path) -> None:
        """Write JSON report to disk atomically."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        tmp.replace(p)


def create_report_from_simulation(
    result: SimulationResult,
    verification_verdicts: Sequence[dict[str, Any]] | None = None,
) -> ReportModel:
    """Build a ReportModel by extracting quantiles and tree nodes from SimulationResult."""
    horizon = result.scenario.horizon
    samples = len(result.trajectories)

    # Collect signals across trajectories
    signal_trajs: dict[str, list[list[float]]] = {}
    branch_nodes: list[BranchTreeNode] = []
    seen_nodes: set[str] = set()

    for traj in result.trajectories:
        parent_id = None
        for step_rec in traj.steps:
            if step_rec.transition_result is not None:
                st = step_rec.transition_result.next_state
                node_id = f"{traj.sample_id}_{st.step}_{st.fingerprint[:8]}"
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    branch_nodes.append(
                        BranchTreeNode(
                            node_id=node_id,
                            parent_id=parent_id,
                            step=st.step,
                            state_fingerprint=st.fingerprint,
                            branch_tag=f"sample_{traj.sample_id}",
                        )
                    )
                parent_id = node_id

                # Collect resources
                for r_id, r in st.resources.items():
                    if r_id not in signal_trajs:
                        signal_trajs[r_id] = []
                    # ensure trajectory slot exists
                    while len(signal_trajs[r_id]) <= traj.sample_id:
                        signal_trajs[r_id].append([])
                    signal_trajs[r_id][traj.sample_id].append(float(r.current))

                # Collect memory numerical variables
                for m_k, m_v in st.memory.items():
                    if isinstance(m_v, (int, float)):
                        if m_k not in signal_trajs:
                            signal_trajs[m_k] = []
                        while len(signal_trajs[m_k]) <= traj.sample_id:
                            signal_trajs[m_k].append([])
                        signal_trajs[m_k][traj.sample_id].append(float(m_v))

    signals: dict[str, SignalQuantiles] = {}
    for sig_name, trajs in signal_trajs.items():
        if not trajs:
            continue
        # Truncate / pad to common length
        max_len = max(len(t) for t in trajs)
        if max_len == 0:
            continue
        matrix = np.zeros((len(trajs), max_len), dtype=np.float64)
        for i, t in enumerate(trajs):
            for step_idx in range(max_len):
                if step_idx < len(t):
                    matrix[i, step_idx] = t[step_idx]
                else:
                    matrix[i, step_idx] = t[-1] if t else 0.0

        p10 = np.percentile(matrix, 10.0, axis=0)
        p25 = np.percentile(matrix, 25.0, axis=0)
        p50 = np.percentile(matrix, 50.0, axis=0)
        p75 = np.percentile(matrix, 75.0, axis=0)
        p90 = np.percentile(matrix, 90.0, axis=0)
        mean_vals = np.mean(matrix, axis=0)
        steps_list = list(range(1, max_len + 1))

        signals[sig_name] = SignalQuantiles(
            signal_name=sig_name,
            steps=steps_list,
            p10=[float(v) for v in p10],
            p25=[float(v) for v in p25],
            p50=[float(v) for v in p50],
            p75=[float(v) for v in p75],
            p90=[float(v) for v in p90],
            mean=[float(v) for v in mean_vals],
        )

    # Collect violations
    violations: list[dict[str, Any]] = []
    for traj in result.trajectories:
        for step_rec in traj.steps:
            for v in step_rec.constraint_violations:
                violations.append(
                    {
                        "sample_id": traj.sample_id,
                        "step": step_rec.step,
                        "constraint_id": v.constraint_id,
                        "is_hard": v.severity == ConstraintSeverity.HARD,
                        "message": v.message,
                    }
                )

    return ReportModel(
        simulation_id=result.scenario.scenario_id,
        fingerprint=result.fingerprint,
        scenario_id=result.scenario.scenario_id,
        horizon=horizon,
        sample_count=samples,
        signals=signals,
        branch_tree=branch_nodes,
        violations_summary=violations,
        verification_verdicts=list(verification_verdicts or []),
        metadata=(
            result.metadata.model_dump()
            if hasattr(result.metadata, "model_dump")
            else (dict(result.metadata) if result.metadata else {})
        ),
    )
