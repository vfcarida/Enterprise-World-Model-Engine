"""Unit tests for co-simulation master, exchange envelopes, and dynamics adapters (T5)."""

from __future__ import annotations

import numpy as np

from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.world import World
from ewm_engine.cosim.dynamics_adapter import (
    CoSimDynamicsModel,
)
from ewm_engine.cosim.envelope import (
    CoSimExchangeEnvelope,
    CouplingScheme,
    PortDirection,
    SubModel,
    VariablePortDef,
)
from ewm_engine.cosim.master import CoSimMaster


class MockSubModel(SubModel):
    """Simple deterministic sub-model for testing co-simulation scheduling."""

    def __init__(self, model_id: str, multiplier: float = 2.0) -> None:
        self._model_id = model_id
        self.multiplier = multiplier
        self._val = 10.0
        self._ports = (
            VariablePortDef(name="in_val", direction=PortDirection.INPUT),
            VariablePortDef(name="out_val", direction=PortDirection.OUTPUT),
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def ports(self) -> tuple[VariablePortDef, ...]:
        return self._ports

    def reset(self, seed: int | None = None) -> None:
        self._val = 10.0

    def get_outputs(self) -> dict[str, float]:
        return {"out_val": self._val}

    def step(self, t_start: float, t_step: float, inputs: dict[str, float]) -> dict[str, float]:
        in_v = inputs.get("in_val", 1.0)
        self._val = in_v * self.multiplier
        return {"out_val": self._val}


def test_cosim_envelope_fingerprint() -> None:
    """Test deterministic fingerprinting of CoSimExchangeEnvelope."""
    env1 = CoSimExchangeEnvelope(
        timestamp=1.0,
        step_index=0,
        sender_id="sub1",
        variables={"temp": 25.5, "pressure": 101.3},
    )
    env2 = CoSimExchangeEnvelope(
        timestamp=1.0,
        step_index=0,
        sender_id="sub1",
        variables={"pressure": 101.3, "temp": 25.5},  # reversed key order
    )
    assert env1.envelope_hash == env2.envelope_hash
    assert len(env1.envelope_hash) == 64


def test_cosim_master_coupling_and_connections() -> None:
    """Test CoSimMaster executing connected sub-models with ZOH and linear interpolation."""
    master = CoSimMaster(coupling_scheme=CouplingScheme.ZERO_ORDER_HOLD)

    sub_a = MockSubModel("model_a", multiplier=2.0)
    sub_b = MockSubModel("model_b", multiplier=3.0)

    master.register_submodel(sub_a)
    master.register_submodel(sub_b)

    # Route output of model_a to input of model_b
    master.connect(
        src_model="model_a",
        src_port="out_val",
        dst_model="model_b",
        dst_port="in_val",
    )

    # Step 1: external input to model_a is 5.0 -> out_a = 10.0 -> out_b = 30.0
    res1 = master.step(t_step=1.0, external_inputs={"model_a.in_val": 5.0})
    assert res1.timestamp == 1.0
    assert res1.step_index == 0
    assert res1.submodel_outputs["model_a"]["out_val"] == 10.0
    assert res1.submodel_outputs["model_b"]["out_val"] == 30.0
    assert len(res1.step_fingerprint) == 64

    # Step 2
    res2 = master.step(t_step=1.0, external_inputs={"model_a.in_val": 2.0})
    assert res2.timestamp == 2.0
    assert res2.step_index == 1
    assert res2.submodel_outputs["model_a"]["out_val"] == 4.0
    assert res2.submodel_outputs["model_b"]["out_val"] == 12.0


def test_cosim_dynamics_model_integration_with_world() -> None:
    """Test CoSimDynamicsModel adapting CoSimMaster as a top-level World DynamicsModel."""
    master = CoSimMaster()
    sub = MockSubModel("thermal", multiplier=1.5)
    master.register_submodel(sub)

    cosim_dyn = CoSimDynamicsModel(
        master=master,
        time_step=1.0,
        resource_mappings={"heat": ("thermal", "out_val")},
    )

    init_state = WorldState(
        step=0,
        resources={"heat": Resource(id="heat", current=10.0)},
        memory={"in_val": 4.0},
    )

    world = World(initial_state=init_state, dynamics=cosim_dyn)
    assert world.dynamics is not None
    trans = world.dynamics.transition(init_state, (), (), np.random.default_rng(0))

    assert trans.next_state.step == 1
    # 4.0 * 1.5 = 6.0
    assert trans.next_state.resources["heat"].current == 6.0
