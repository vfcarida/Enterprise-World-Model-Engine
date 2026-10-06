"""Co-simulation master orchestrator for multi-model synchronization.

Provides deterministic scheduling, typed port routing, and zero-order-hold
or linear-interpolated coupling between sub-models.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.cosim.envelope import (
    CoSimExchangeEnvelope,
    CouplingScheme,
    SubModel,
)


class PortConnection(BaseModel):
    """Routing connection between an output port of a source model and an input port of a destination model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    src_model: str = Field(description="Identifier of the source sub-model.")
    src_port: str = Field(description="Name of the source output port.")
    dst_model: str = Field(description="Identifier of the destination sub-model.")
    dst_port: str = Field(description="Name of the destination input port.")


class CoSimStepResult(BaseModel):
    """Result of a co-simulation master macro-step across all participating sub-models."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(ge=0.0, description="End time of the macro-step.")
    step_index: int = Field(ge=0, description="Macro-step index.")
    submodel_outputs: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Outputs emitted by each sub-model."
    )
    envelopes: list[CoSimExchangeEnvelope] = Field(
        default_factory=list, description="All exchanged envelopes during this step."
    )
    step_fingerprint: str = Field(description="Deterministic SHA-256 fingerprint of the step.")


class CoSimMaster:
    """Master orchestrator for co-simulation federations."""

    def __init__(
        self,
        coupling_scheme: CouplingScheme = CouplingScheme.ZERO_ORDER_HOLD,
        schedule: list[str] | None = None,
    ) -> None:
        self.coupling_scheme = coupling_scheme
        self._submodels: dict[str, SubModel] = {}
        self._connections: list[PortConnection] = []
        self._schedule: list[str] = schedule or []
        # History buffers for interpolation: model_id -> port_name -> [(t, val)]
        self._history: dict[str, dict[str, list[tuple[float, Any]]]] = {}
        self._step_counter: int = 0
        self._current_time: float = 0.0

    @property
    def submodels(self) -> dict[str, SubModel]:
        """Dictionary of registered sub-models."""
        return dict(self._submodels)

    @property
    def connections(self) -> tuple[PortConnection, ...]:
        """Registered port routing connections."""
        return tuple(self._connections)

    def register_submodel(self, model: SubModel) -> None:
        """Register a sub-model into the federation."""
        if model.model_id in self._submodels:
            raise ValueError(f"Sub-model with id '{model.model_id}' is already registered.")
        self._submodels[model.model_id] = model
        self._history[model.model_id] = {}
        if model.model_id not in self._schedule:
            self._schedule.append(model.model_id)

    def connect(
        self,
        src_model: str,
        src_port: str,
        dst_model: str,
        dst_port: str,
    ) -> None:
        """Create a directed dataflow connection between sub-model ports."""
        if src_model not in self._submodels:
            raise KeyError(f"Source sub-model '{src_model}' is not registered.")
        if dst_model not in self._submodels:
            raise KeyError(f"Destination sub-model '{dst_model}' is not registered.")

        conn = PortConnection(
            src_model=src_model,
            src_port=src_port,
            dst_model=dst_model,
            dst_port=dst_port,
        )
        self._connections.append(conn)

    def reset(self, seed: int | None = None) -> None:
        """Reset all registered sub-models and master scheduling state."""
        self._step_counter = 0
        self._current_time = 0.0
        self._history.clear()
        for idx, model_id in enumerate(self._schedule):
            sub_seed = (seed + idx * 1000) if seed is not None else None
            self._submodels[model_id].reset(seed=sub_seed)
            self._history[model_id] = {}

    def _resolve_input(
        self,
        dst_model: str,
        dst_port: str,
        t_eval: float,
        external_inputs: dict[str, Any],
    ) -> Any:
        """Resolve the input value for a destination port using connections and coupling scheme."""
        # Check if connected from another submodel
        for conn in self._connections:
            if conn.dst_model == dst_model and conn.dst_port == dst_port:
                src_hist = self._history.get(conn.src_model, {}).get(conn.src_port, [])
                if not src_hist:
                    # Fallback to current model output
                    curr_out = self._submodels[conn.src_model].get_outputs()
                    return curr_out.get(conn.src_port, 0.0)

                if self.coupling_scheme == CouplingScheme.ZERO_ORDER_HOLD or len(src_hist) < 2:
                    # Return latest recorded value
                    return src_hist[-1][1]

                # Linear interpolation between last two history points
                t0, v0 = src_hist[-2]
                t1, v1 = src_hist[-1]
                if t1 == t0 or not isinstance(v0, (int, float)) or not isinstance(v1, (int, float)):
                    return v1
                ratio = max(0.0, min(1.0, (t_eval - t0) / (t1 - t0)))
                return float(v0 + ratio * (v1 - v0))

        # Check external inputs
        qualified_key = f"{dst_model}.{dst_port}"
        if qualified_key in external_inputs:
            return external_inputs[qualified_key]
        if dst_port in external_inputs:
            return external_inputs[dst_port]

        return 0.0

    def step(
        self,
        t_step: float,
        external_inputs: dict[str, Any] | None = None,
    ) -> CoSimStepResult:
        """Advance the co-simulation federation by macro-step t_step."""
        ext_in = external_inputs or {}
        t_start = self._current_time
        t_end = t_start + t_step

        outputs: dict[str, dict[str, Any]] = {}
        envelopes: list[CoSimExchangeEnvelope] = []

        # Execute scheduled sub-models in sequence
        for model_id in self._schedule:
            sub = self._submodels[model_id]
            # Assemble input vector
            in_vec: dict[str, Any] = {}
            for port in sub.ports:
                if port.direction.value == "input":
                    in_vec[port.name] = self._resolve_input(model_id, port.name, t_end, ext_in)

            # Advance submodel
            sub_out = sub.step(t_start, t_step, in_vec)
            outputs[model_id] = sub_out

            # Record history
            for k_out, v_out in sub_out.items():
                if k_out not in self._history[model_id]:
                    self._history[model_id][k_out] = []
                self._history[model_id][k_out].append((t_end, v_out))
                # Keep last 5 points for memory efficiency
                if len(self._history[model_id][k_out]) > 5:
                    self._history[model_id][k_out].pop(0)

            env = CoSimExchangeEnvelope(
                timestamp=t_end,
                step_index=self._step_counter,
                sender_id=model_id,
                variables=sub_out,
            )
            envelopes.append(env)

        # Fingerprint of the step
        step_payload = {
            "step_index": self._step_counter,
            "t_end": round(t_end, 6),
            "envelopes": [e.envelope_hash for e in envelopes],
        }
        fp = canonical_sha256(step_payload)

        result = CoSimStepResult(
            timestamp=t_end,
            step_index=self._step_counter,
            submodel_outputs=outputs,
            envelopes=envelopes,
            step_fingerprint=fp,
        )

        self._step_counter += 1
        self._current_time = t_end
        return result
