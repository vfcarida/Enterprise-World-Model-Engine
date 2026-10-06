"""Co-simulation and interoperability module (T5).

Provides master scheduling, typed variable exchange envelopes, and adapters
for continuous FMUs, discrete-event simulation (SimPy), agent-based models (Mesa),
and system dynamics (PySD).
"""

from __future__ import annotations

from ewm_engine.cosim.adapters import (
    FMUSubModel,
    MesaSubModel,
    PySDSubModel,
    SimPySubModel,
    is_fmpy_available,
    is_mesa_available,
    is_pysd_available,
    is_simpy_available,
)
from ewm_engine.cosim.dynamics_adapter import (
    CoSimDynamicsModel,
    SubModelDynamicsAdapter,
)
from ewm_engine.cosim.envelope import (
    CoSimExchangeEnvelope,
    CouplingScheme,
    PortDirection,
    SubModel,
    VariablePortDef,
)
from ewm_engine.cosim.master import (
    CoSimMaster,
    CoSimStepResult,
    PortConnection,
)

__all__ = [
    "CoSimDynamicsModel",
    "CoSimExchangeEnvelope",
    "CoSimMaster",
    "CoSimStepResult",
    "CouplingScheme",
    "FMUSubModel",
    "MesaSubModel",
    "PortConnection",
    "PortDirection",
    "PySDSubModel",
    "SimPySubModel",
    "SubModel",
    "SubModelDynamicsAdapter",
    "VariablePortDef",
    "is_fmpy_available",
    "is_mesa_available",
    "is_pysd_available",
    "is_simpy_available",
]
