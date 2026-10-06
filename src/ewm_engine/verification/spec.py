"""Property specification DSL and canonical cryptographic hashing.

Conforms to Track T3: Trajectory Verification (Property-Spec DSL).
Every property specification has a deterministic canonical SHA-256 hash that
folds directly into the scenario, verification, or result provenance fingerprint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core._canonical import canonical_sha256


class PropertySpec(BaseModel):
    """Declarative specification of an invariant or temporal property over trajectories.

    Every PropertySpec has a deterministic cryptographic SHA-256 hash computed
    over its canonical representation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(
        default="1.0.0",
        description="Semantic schema version.",
    )
    property_id: str = Field(
        description="Globally unique identifier for this verification property.",
    )
    name: str = Field(
        description="Human-readable property name.",
    )
    description: str = Field(
        default="",
        description="Formal or narrative explanation of the requirement.",
    )
    property_type: Literal["oracle_graph", "stl_bounded", "rtamt_stl", "custom"] = Field(
        default="oracle_graph",
        description="Verification engine discriminator.",
    )
    definition: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured specification payload (e.g. oracle graph nodes/edges or STL formula).",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Thresholds, timing tolerance windows, or evaluation hyperparameters.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Author, domain tags, or reference documentation.",
    )

    @property
    def property_hash(self) -> str:
        """Deterministic canonical SHA-256 hash of the property specification."""
        canonical_dict = {
            "schema_version": self.schema_version,
            "property_id": self.property_id,
            "name": self.name,
            "property_type": self.property_type,
            "definition": self.definition,
            "parameters": self.parameters,
        }
        return canonical_sha256(canonical_dict)


def compute_property_hash(spec: PropertySpec) -> str:
    """Compute the canonical cryptographic hash of a PropertySpec."""
    return spec.property_hash


def fold_properties_into_fingerprint(
    base_fingerprint: str,
    properties: Sequence[PropertySpec],
) -> str:
    """Fold an ordered set of checked PropertySpecs into a simulation/verification fingerprint.

    Ensures that the exact set of verified properties is baked into cryptographic provenance.
    """
    sorted_hashes = sorted(p.property_hash for p in properties)
    payload = {
        "base_fingerprint": base_fingerprint,
        "property_hashes": sorted_hashes,
    }
    return canonical_sha256(payload)
