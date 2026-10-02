"""Core typing definitions and aliases for EWM Engine."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

# Identifier Type Aliases
EntityId: TypeAlias = str
ResourceId: TypeAlias = str
RelationshipId: TypeAlias = str
ActionId: TypeAlias = str
EventId: TypeAlias = str
ActorId: TypeAlias = str
ConstraintId: TypeAlias = str
ScenarioId: TypeAlias = str
TrajectoryId: TypeAlias = str

# Random Number Generator Type
RandomGenerator: TypeAlias = np.random.Generator
