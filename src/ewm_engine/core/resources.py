"""Resource domain primitive for finite, measurable organizational quantities."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ewm_engine.core.types import EntityId, ResourceId
from ewm_engine.exceptions import ResourceBoundsError


class Resource(BaseModel):
    """Represents a finite, bounded, or measurable quantity in the enterprise.

    Models physical stock, vehicle fleets, operational budgets, bed capacities,
    workforce hours, or bandwidth.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ResourceId = Field(description="Unique identifier for the resource.")
    entity_id: EntityId | None = Field(
        default=None,
        description="Optional entity ID owning or hosting this resource.",
    )
    current: float = Field(description="Current quantity or stock of the resource.")
    min_value: float = Field(default=0.0, description="Minimum allowable lower bound.")
    max_value: float = Field(
        default=float("inf"),
        description="Maximum capacity or upper bound limit.",
    )
    unit: str = Field(
        default="", description="Measurement unit (e.g., 'units', 'liters', 'hours')."
    )
    description: str = Field(default="", description="Descriptive metadata for the resource.")

    @model_validator(mode="after")
    def validate_bounds(self) -> Resource:
        """Validate that min_value does not exceed max_value."""
        if self.min_value > self.max_value:
            raise ValueError(
                f"Resource '{self.id}' min_value ({self.min_value}) exceeds max_value ({self.max_value})."
            )
        return self

    @property
    def utilization(self) -> float:
        """Calculate fractional capacity utilization in [0.0, 1.0]."""
        if math.isinf(self.max_value):
            return 0.0
        capacity = self.max_value - self.min_value
        if capacity <= 0.0:
            return 1.0 if self.current >= self.max_value else 0.0
        clamped = max(self.min_value, min(self.current, self.max_value))
        return (clamped - self.min_value) / capacity

    @property
    def available_capacity(self) -> float:
        """Available headroom before reaching max_value."""
        if math.isinf(self.max_value):
            return float("inf")
        return max(0.0, self.max_value - self.current)

    @property
    def is_exhausted(self) -> bool:
        """True if current level is at or below min_value."""
        return self.current <= self.min_value

    @property
    def is_full(self) -> bool:
        """True if current level is at or above max_value."""
        return self.current >= self.max_value

    def with_delta(
        self, delta: float, clamp: bool = False, enforce_bounds: bool = False
    ) -> Resource:
        """Produce an updated resource instance with the specified delta.

        Args:
            delta: Quantity to add (positive) or subtract (negative).
            clamp: If True, clamps value within [min_value, max_value].
            enforce_bounds: If True, raises ResourceBoundsError on bounds violation.
        """
        new_val = self.current + delta
        if clamp:
            new_val = max(self.min_value, min(new_val, self.max_value))
        elif enforce_bounds and (new_val < self.min_value or new_val > self.max_value):
            raise ResourceBoundsError(
                f"Resource '{self.id}' delta {delta:+f} results in {new_val}, "
                f"breaching bounds [{self.min_value}, {self.max_value}]."
            )
        return self.model_copy(update={"current": new_val})

    def with_value(
        self, value: float, clamp: bool = False, enforce_bounds: bool = False
    ) -> Resource:
        """Produce an updated resource instance with an exact new value.

        Args:
            value: The new current level.
            clamp: If True, clamps value within [min_value, max_value].
            enforce_bounds: If True, raises ResourceBoundsError on bounds violation.
        """
        new_val = value
        if clamp:
            new_val = max(self.min_value, min(new_val, self.max_value))
        elif enforce_bounds and (new_val < self.min_value or new_val > self.max_value):
            raise ResourceBoundsError(
                f"Resource '{self.id}' value {new_val} breaches bounds [{self.min_value}, {self.max_value}]."
            )
        return self.model_copy(update={"current": new_val})
