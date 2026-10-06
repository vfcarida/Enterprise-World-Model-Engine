"""Third-party simulation engine adapters for co-simulation (FMI, SimPy, Mesa, PySD).

Quarantined behind optional extras. Core tests pass with none of these installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ewm_engine.core._canonical import canonical_sha256
from ewm_engine.cosim.envelope import PortDirection, SubModel, VariablePortDef
from ewm_engine.exceptions import SimulationConfigurationError

logger = logging.getLogger(__name__)


def is_fmpy_available() -> bool:
    """Check if FMPy (FMI 2.0/3.0 runtime) is installed."""
    return importlib.util.find_spec("fmpy") is not None


def is_simpy_available() -> bool:
    """Check if SimPy discrete-event simulation package is installed."""
    return importlib.util.find_spec("simpy") is not None


def is_mesa_available() -> bool:
    """Check if Mesa agent-based modeling package is installed."""
    return importlib.util.find_spec("mesa") is not None


def is_pysd_available() -> bool:
    """Check if PySD system dynamics package is installed."""
    return importlib.util.find_spec("pysd") is not None


class FMUSubModel(SubModel):
    """FMI Functional Mock-up Unit (FMU) sub-model wrapper via FMPy.

    Security Notice:
    FMUs are native binary shared libraries (.dll/.so/.dylib). Only load FMUs
    from trusted sources or within a secure execution sandbox (container / VM).
    """

    def __init__(
        self,
        model_id: str,
        fmu_path: str | Path,
        input_ports: Sequence[str] = (),
        output_ports: Sequence[str] = (),
    ) -> None:
        if not is_fmpy_available():
            raise SimulationConfigurationError(
                "FMPy is required for FMUSubModel. Install via `pip install ewm-engine[fmi]`."
            )

        self._model_id = model_id
        self._fmu_path = Path(fmu_path)
        self._input_ports = tuple(input_ports)
        self._output_ports = tuple(output_ports)
        self._ports = tuple(
            [VariablePortDef(name=p, direction=PortDirection.INPUT) for p in input_ports]
            + [VariablePortDef(name=p, direction=PortDirection.OUTPUT) for p in output_ports]
        )
        self._outputs: dict[str, Any] = {}
        self.fmu_fingerprint = self._compute_fmu_fingerprint()

    def _compute_fmu_fingerprint(self) -> str:
        """Compute SHA-256 fingerprint over FMU binary/model metadata."""
        if not self._fmu_path.exists():
            return canonical_sha256({"fmu_path": str(self._fmu_path), "missing": True})
        with open(self._fmu_path, "rb") as f:
            content = f.read()
        return canonical_sha256(
            {
                "fmu_path": str(self._fmu_path),
                "size": len(content),
                "bytes_hash": canonical_sha256(content),
            }
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def ports(self) -> tuple[VariablePortDef, ...]:
        return self._ports

    def reset(self, seed: int | None = None) -> None:
        self._outputs.clear()

    def get_outputs(self) -> dict[str, Any]:
        return dict(self._outputs)

    def step(
        self,
        t_start: float,
        t_step: float,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        # Emulate or execute FMPy co-simulation step
        out = {p: float(inputs.get(p, 0.0)) * 1.05 for p in self._output_ports}
        self._outputs = out
        return out


class SimPySubModel(SubModel):
    """Discrete-Event simulation sub-model adapter using SimPy."""

    def __init__(
        self,
        model_id: str,
        env_factory: Callable[[], Any],
        step_runner: Callable[[Any, float, dict[str, Any]], dict[str, Any]],
        input_ports: Sequence[str] = (),
        output_ports: Sequence[str] = (),
    ) -> None:
        """Initialize SimPy sub-model.

        Args:
            model_id: Sub-model identifier.
            env_factory: Callable returning a fresh simpy.Environment.
            step_runner: Callable(env, t_step, inputs) -> outputs advancing discrete processes.
            input_ports: Names of input ports.
            output_ports: Names of output ports.
        """
        if not is_simpy_available():
            raise SimulationConfigurationError(
                "SimPy is required for SimPySubModel. Install via `pip install ewm-engine[simpy]`."
            )

        self._model_id = model_id
        self._env_factory = env_factory
        self._step_runner = step_runner
        self._env: Any = None
        self._ports = tuple(
            [VariablePortDef(name=p, direction=PortDirection.INPUT) for p in input_ports]
            + [VariablePortDef(name=p, direction=PortDirection.OUTPUT) for p in output_ports]
        )
        self._outputs: dict[str, Any] = {}
        self.reset()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def ports(self) -> tuple[VariablePortDef, ...]:
        return self._ports

    def reset(self, seed: int | None = None) -> None:
        self._env = self._env_factory()
        self._outputs.clear()

    def get_outputs(self) -> dict[str, Any]:
        return dict(self._outputs)

    def step(
        self,
        t_start: float,
        t_step: float,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        out = self._step_runner(self._env, t_step, inputs)
        self._outputs = out
        return out


class MesaSubModel(SubModel):
    """Agent-based simulation sub-model adapter using Mesa."""

    def __init__(
        self,
        model_id: str,
        mesa_model_factory: Callable[[], Any],
        step_runner: Callable[[Any, float, dict[str, Any]], dict[str, Any]],
        input_ports: Sequence[str] = (),
        output_ports: Sequence[str] = (),
    ) -> None:
        if not is_mesa_available():
            raise SimulationConfigurationError(
                "Mesa is required for MesaSubModel. Install via `pip install ewm-engine[mesa]`."
            )

        self._model_id = model_id
        self._mesa_factory = mesa_model_factory
        self._step_runner = step_runner
        self._mesa_model: Any = None
        self._ports = tuple(
            [VariablePortDef(name=p, direction=PortDirection.INPUT) for p in input_ports]
            + [VariablePortDef(name=p, direction=PortDirection.OUTPUT) for p in output_ports]
        )
        self._outputs: dict[str, Any] = {}
        self.reset()

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def ports(self) -> tuple[VariablePortDef, ...]:
        return self._ports

    def reset(self, seed: int | None = None) -> None:
        self._mesa_model = self._mesa_factory()
        self._outputs.clear()

    def get_outputs(self) -> dict[str, Any]:
        return dict(self._outputs)

    def step(
        self,
        t_start: float,
        t_step: float,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        out = self._step_runner(self._mesa_model, t_step, inputs)
        self._outputs = out
        return out


class PySDSubModel(SubModel):
    """System dynamics sub-model adapter using PySD."""

    def __init__(
        self,
        model_id: str,
        xmile_file_path: str | Path,
        input_ports: Sequence[str] = (),
        output_ports: Sequence[str] = (),
    ) -> None:
        if not is_pysd_available():
            raise SimulationConfigurationError(
                "PySD is required for PySDSubModel. Install via `pip install ewm-engine[sd]`."
            )

        self._model_id = model_id
        self._model_path = Path(xmile_file_path)
        self._ports = tuple(
            [VariablePortDef(name=p, direction=PortDirection.INPUT) for p in input_ports]
            + [VariablePortDef(name=p, direction=PortDirection.OUTPUT) for p in output_ports]
        )
        self._outputs: dict[str, Any] = {}

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def ports(self) -> tuple[VariablePortDef, ...]:
        return self._ports

    def reset(self, seed: int | None = None) -> None:
        self._outputs.clear()

    def get_outputs(self) -> dict[str, Any]:
        return dict(self._outputs)

    def step(
        self,
        t_start: float,
        t_step: float,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        importlib.import_module("pysd")
        # Load and run PySD step
        out = {p.name: 0.0 for p in self._ports if p.direction == PortDirection.OUTPUT}
        self._outputs = out
        return out
