# Co-Simulation: Master Scheduling & Model Exchange

This guide demonstrates how to couple external domain simulations (discrete-event, system-dynamics, agent-based, and vendor FMUs) to the Enterprise World Model Engine using `ewm_engine.cosim`.

---

## 1. Architecture Overview

Co-simulation in EWM Engine is based on a zero-dependency master scheduling loop (`CoSimMaster`):
- **Deterministic Fixed-Step Master Loop:** Advances time monotonically with fixed step size $\Delta t$.
- **Typed Exchange Envelopes:** Inputs and outputs are exchanged via `CoSimExchangeEnvelope`.
- **Zero-Order-Hold (ZOH) Interpolation:** Handles sub-models operating at different natural time rates.
- **Unified Dynamics Wrapping:** `CoSimDynamicsModel` wraps the entire co-simulation ensemble into a standard EWM `DynamicsModel`, letting you attach constraints, systemic traces, and Monte Carlo engines.

---

## 2. Basic Co-Simulation Master Loop

```python
from ewm_engine.cosim import CoSimMaster, SubModel


class SimplePlantModel:
    """Discrete sub-model simulating a manufacturing unit."""

    def __init__(self, model_id: str = "plant"):
        self.model_id = model_id
        self.temperature = 25.0

    def initialize(self, start_time: float = 0.0) -> None:
        self.temperature = 25.0

    def step(self, t_current: float, t_step: float, inputs: dict) -> dict:
        cooling_power = inputs.get("cooling_power", 0.0)
        self.temperature += 2.0 * t_step - cooling_power * 0.5 * t_step
        return {"temperature": self.temperature}

    def terminate(self) -> None:
        pass

    def get_state(self) -> dict:
        return {"temp": self.temperature}

    def set_state(self, state: dict) -> None:
        self.temperature = state.get("temp", 25.0)


# Instantiate master and register sub-models
master = CoSimMaster(t_step=1.0)
master.register_submodel(
    "plant",
    SimplePlantModel(),
    required_inputs=("cooling_power",),
    provided_outputs=("temperature",),
)

# Step the coupled system
master.initialize(start_time=0.0)
envelope = master.step_master(inputs={"cooling_power": 4.0})
print("Envelope outputs:", envelope.outputs)
```

---

## 3. Adapters for Domain Engines

All external engines are quarantined behind optional extras:

| Tool | Extra | Adapter Class | Description |
| :--- | :--- | :--- | :--- |
| **FMI / FMU** | `[fmi]` | `FMUSubModel` | Couples vendor Functional Mock-up Units via FMPy |
| **SimPy** | `[simpy]` | `SimPySubModel` | Steps discrete-event process simulations |
| **Mesa** | `[mesa]` | `MesaSubModel` | Steps spatial/relational agent-based models |
| **PySD** | `[sd]` | `PySDSubModel` | Steps Vensim/XMILE system-dynamics differential equations |

### SimPy Discrete-Event Example

```python
from ewm_engine.cosim.adapters import SimPySubModel
import simpy


def env_factory():
    return simpy.Environment()


def step_runner(env, t_step, inputs):
    env.run(until=env.now + t_step)
    return {"processed_orders": env.now * 1.5}


simpy_model = SimPySubModel(
    model_id="warehouse_ops",
    env_factory=env_factory,
    step_runner=step_runner,
    required_inputs=(),
    provided_outputs=("processed_orders",),
)
```

---

## 4. CLI System-Dynamics Importer

Extract stocks and auxiliary variables from PySD models directly into EWM schemas:

```bash
ewm import-sd models/supply_chain.py --out src/models/supply_chain_ewm.json
```
