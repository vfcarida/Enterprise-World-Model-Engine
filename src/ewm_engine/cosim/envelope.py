"""Co-simulation variable exchange envelopes and sub-model protocols.

Defines the zero-dependency scheduling and model-exchange contracts for coupling
heterogeneous dynamics models (discrete-event, continuous FMUs, system dynamics).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256


class CouplingScheme(StrEnum):
    """Interpolation coupling semantics between co-simulation macro-steps."""

    ZERO_ORDER_HOLD = "zero_order_hold"
    LINEAR_INTERPOLATION = "linear_interpolation"


class PortDirection(StrEnum):
    """Dataflow direction of a variable port."""

    INPUT = "input"
    OUTPUT = "output"


class VariablePortDef(BaseModel):
    """Specification of an exchangeable variable port on a co-simulation sub-model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="Name of the port.")
    direction: PortDirection = Field(description="Port direction: input or output.")
    var_type: str = Field(default="float", description="Data type: float, int, str, bool.")
    unit: str = Field(default="", description="Physical or conceptual unit.")
    description: str = Field(
        default="", description="Human-readable description of port semantics."
    )


class CoSimExchangeEnvelope(BaseModel):
    """Typed variable exchange envelope between co-simulation sub-models at a macro-step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(ge=0.0, description="Simulation continuous timestamp.")
    step_index: int = Field(ge=0, description="Discrete macro-step index.")
    sender_id: str = Field(description="Sub-model emitting the envelope.")
    variables: dict[str, Any] = Field(
        default_factory=dict, description="Exchanged variable values."
    )

    @property
    def envelope_hash(self) -> str:
        """Deterministic canonical SHA-256 fingerprint of the envelope."""
        normalized = {
            "timestamp": round(self.timestamp, 6),
            "step_index": self.step_index,
            "sender_id": self.sender_id,
            "variables": {
                k: float(v) if isinstance(v, (int, float)) else str(v)
                for k, v in sorted(self.variables.items())
            },
        }
        return canonical_sha256(normalized)


@runtime_checkable
class SubModel(Protocol):
    """Protocol for an external or wrapped co-simulation sub-model."""

    @property
    def model_id(self) -> str:
        """Unique identifier of the sub-model within the co-simulation federation."""
        ...

    @property
    def ports(self) -> tuple[VariablePortDef, ...]:
        """Declared input and output variable ports."""
        ...

    def step(
        self,
        t_start: float,
        t_step: float,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Advance the sub-model from t_start to t_start + t_step with provided inputs.

        Returns output variable dictionary.
        """
        ...

    def reset(self, seed: int | None = None) -> None:
        """Reset sub-model state to initial conditions."""
        ...

    def get_outputs(self) -> dict[str, Any]:
        """Return the current output values without advancing time."""
        ...
