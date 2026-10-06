"""DynamicsModel adapters for co-simulation.

Allows wrapping existing DynamicsModels as co-simulation sub-models, and
running an entire CoSimMaster federation as a top-level DynamicsModel inside World.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ewm_engine.core.actions import Action
from ewm_engine.core.events import ExogenousEvent
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import RandomGenerator
from ewm_engine.cosim.envelope import PortDirection, SubModel, VariablePortDef
from ewm_engine.cosim.master import CoSimMaster
from ewm_engine.dynamics.base import DynamicsModel, TransitionResult


class SubModelDynamicsAdapter(SubModel):
    """Adapts an existing DynamicsModel into a co-simulation SubModel."""

    def __init__(
        self,
        model_id: str,
        dynamics: DynamicsModel,
        input_ports: Sequence[str] = (),
        output_ports: Sequence[str] = (),
    ) -> None:
        self._model_id = model_id
        self._dynamics = dynamics
        self._ports = tuple(
            [VariablePortDef(name=p, direction=PortDirection.INPUT) for p in input_ports]
            + [VariablePortDef(name=p, direction=PortDirection.OUTPUT) for p in output_ports]
        )
        self._current_outputs: dict[str, Any] = {}
        self._internal_state: WorldState | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def ports(self) -> tuple[VariablePortDef, ...]:
        return self._ports

    def reset(self, seed: int | None = None) -> None:
        self._current_outputs.clear()
        self._internal_state = WorldState(step=0)

    def get_outputs(self) -> dict[str, Any]:
        return dict(self._current_outputs)

    def step(
        self,
        t_start: float,
        t_step: float,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        state = self._internal_state or WorldState(step=int(t_start))
        # Update state memory with incoming inputs
        new_mem = dict(state.memory)
        new_mem.update(inputs)
        st = state.model_copy(update={"memory": new_mem, "timestamp": t_start})

        rng = np.random.default_rng(int(t_start * 1000))
        res = self._dynamics.transition(st, (), (), rng)
        self._internal_state = res.next_state

        # Collect outputs from next state
        out: dict[str, Any] = {}
        for port in self._ports:
            if port.direction == PortDirection.OUTPUT:
                if port.name in res.next_state.resources:
                    out[port.name] = res.next_state.resources[port.name].current
                elif port.name in res.next_state.memory:
                    out[port.name] = res.next_state.memory[port.name]
                else:
                    out[port.name] = 0.0

        self._current_outputs = out
        return out


class CoSimDynamicsModel(DynamicsModel):
    """DynamicsModel that delegates state progression to a CoSimMaster federation."""

    def __init__(
        self,
        master: CoSimMaster,
        time_step: float = 1.0,
        resource_mappings: dict[str, tuple[str, str]] | None = None,
        memory_mappings: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        """Initialize CoSimDynamicsModel.

        Args:
            master: Configured CoSimMaster instance.
            time_step: Macro-step duration per World transition.
            resource_mappings: Mapping from World resource ID -> (submodel_id, output_port).
            memory_mappings: Mapping from World memory key -> (submodel_id, output_port).
        """
        self.master = master
        self.time_step = time_step
        self.resource_mappings = resource_mappings or {}
        self.memory_mappings = memory_mappings or {}

    def transition(
        self,
        state: WorldState,
        actions: Sequence[Action] = (),
        exogenous_events: Sequence[ExogenousEvent] = (),
        rng: RandomGenerator | None = None,
    ) -> TransitionResult:
        """Advance the co-simulation federation by time_step and project onto WorldState."""
        # Convert state resources and memory into external inputs
        ext_inputs: dict[str, Any] = {}
        for r_id, r in state.resources.items():
            ext_inputs[r_id] = r.current
        for m_k, m_v in state.memory.items():
            ext_inputs[m_k] = m_v

        step_res = self.master.step(self.time_step, external_inputs=ext_inputs)

        # Update next state resources and memory
        new_resources = dict(state.resources)
        for res_id, (m_id, port_name) in self.resource_mappings.items():
            val = step_res.submodel_outputs.get(m_id, {}).get(port_name)
            if val is not None and isinstance(val, (int, float)):
                new_resources[res_id] = Resource(id=res_id, current=float(val))

        new_memory = dict(state.memory)
        for mem_key, (m_id, port_name) in self.memory_mappings.items():
            val = step_res.submodel_outputs.get(m_id, {}).get(port_name)
            if val is not None:
                new_memory[mem_key] = val

        next_state = state.model_copy(
            update={
                "step": state.step + 1,
                "timestamp": step_res.timestamp,
                "resources": new_resources,
                "memory": new_memory,
            }
        )

        return TransitionResult(
            next_state=next_state,
            applied_changes={"resources": new_resources, "memory": new_memory},
        )
