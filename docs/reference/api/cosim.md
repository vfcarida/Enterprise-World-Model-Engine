# Co-Simulation & Standards API Reference

This module provides the master scheduling loop and model exchange envelopes for co-simulation across FMI 3.0 / FMU, SimPy, Mesa, and PySD (Track T5).

---

## Master Scheduling Loop

::: ewm_engine.cosim.master
    options:
      show_root_heading: true
      show_source: false
      members:
        - CoSimMaster
        - MasterStepResult

---

## Exchange Envelopes & Protocols

::: ewm_engine.cosim.envelope
    options:
      show_root_heading: true
      show_source: false
      members:
        - SubModel
        - CoSimExchangeEnvelope
        - CouplingScheme
        - VariablePortDef

---

## Dynamics Adapters

::: ewm_engine.cosim.dynamics_adapter
    options:
      show_root_heading: true
      show_source: false
      members:
        - CoSimDynamicsModel
        - SubModelDynamicsAdapter

---

## Sub-Model Adapters

::: ewm_engine.cosim.adapters
    options:
      show_root_heading: true
      show_source: false
      members:
        - FMUSubModel
        - SimPySubModel
        - MesaSubModel
        - PySDSubModel
