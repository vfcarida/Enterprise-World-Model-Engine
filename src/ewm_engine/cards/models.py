"""Pydantic Card models for Scenarios, Models, and Datasets.

Conforms to Track T9: Reproducibility Artifacts (Native Cards).
Anchored on:
- Model Cards for Model Reporting (Mitchell et al., arXiv:1810.03993)
- Datasheets for Datasets (Gebru et al., arXiv:1803.09010)
- Population calibration provenance (arXiv:2411.10109)
"""

from __future__ import annotations

import json
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BaseCard(BaseModel):
    """Base class for all canonical reproducibility cards in EWM Engine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = Field(
        default="1.0.0",
        description="Semantic schema version.",
    )
    card_type: str = Field(
        description="Card category discriminator ('model', 'scenario', or 'dataset').",
    )
    artifact_fingerprint: str = Field(
        description="Canonical cryptographic SHA-256 fingerprint of the described artifact.",
    )
    maturity: str = Field(
        default="production",
        description="Maturity tier: 'production', 'beta', or 'experimental'.",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert card to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    def to_json(self, indent: int = 2) -> str:
        """Serialize card to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Serialize card to canonical YAML."""
        return str(yaml.safe_dump(self.to_dict(), sort_keys=False))

    def to_markdown(self) -> str:
        """Render card to formatted GitHub-flavored Markdown."""
        from ewm_engine.cards.render import render_card_markdown

        return render_card_markdown(self)


class ModelCard(BaseCard):
    """Native Model Card documenting a DynamicsModel or Policy Actor.

    Fields aligned with Mitchell et al. (arXiv:1810.03993) and population
    calibration provenance standards.
    """

    card_type: Literal["model"] = "model"
    model_id: str = Field(
        description="Unique identifier of the model.",
    )
    model_name: str = Field(
        description="Human-readable model name.",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version string of the model.",
    )
    description: str = Field(
        default="",
        description="Detailed description of dynamics mechanism, architecture, or policy.",
    )
    intended_use: list[str] = Field(
        default_factory=list,
        description="Declared organizational decisions and operating regimes model was designed for.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Explicit theoretical, structural, or behavioral assumptions embedded in dynamics.",
    )
    out_of_scope: list[str] = Field(
        default_factory=list,
        description="Known scenarios, domains, or extreme regimes where model MUST NOT be applied.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Epistemic caveats, numerical approximations, or unmodeled feedback loops.",
    )
    constraints_exercised: list[str] = Field(
        default_factory=list,
        description="Identifiers of physical or operational constraints evaluated with this model.",
    )
    metrics: dict[str, float | str] = Field(
        default_factory=dict,
        description="Benchmark or calibration metrics (e.g. RMSE, coverage, calibration score).",
    )
    provenance_sources: list[str] = Field(
        default_factory=list,
        description="Source dataset hashes, training run IDs, or empirical telemetry sources.",
    )
    calibration_score: float | None = Field(
        default=None,
        description="Empirical population calibration metric score (e.g. SBC/Kolmogorov score).",
    )


class ScenarioCard(BaseCard):
    """Native Scenario Card documenting an experiment setup, stress scenario, or policy sweep."""

    card_type: Literal["scenario"] = "scenario"
    scenario_id: str = Field(
        description="Unique scenario identifier.",
    )
    description: str = Field(
        default="",
        description="Narrative explanation of the organizational experiment or policy shock.",
    )
    horizon: int = Field(
        ge=1,
        description="Discrete simulation time steps per rollout.",
    )
    samples: int = Field(
        ge=1,
        description="Monte Carlo rollout count.",
    )
    seed: int = Field(
        description="Master pseudo-random seed.",
    )
    interventions: list[str] = Field(
        default_factory=list,
        description="Interventions applied in this scenario (e.g. capacity expansions, rule shifts).",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions regarding exogenous shock rates, actor adherence, or context stability.",
    )
    out_of_scope: list[str] = Field(
        default_factory=list,
        description="Conditions under which this scenario is not representative.",
    )
    environmental_context: dict[str, Any] = Field(
        default_factory=dict,
        description="Context variables (e.g. interest rates, demand shock multipliers).",
    )
    metrics: dict[str, float | str] = Field(
        default_factory=dict,
        description="Expected or observed scenario summary metrics.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Scenario design limitations and known confounders.",
    )


class DatasetCard(BaseCard):
    """Native Dataset Card documenting a simulation trace or synthetic training dataset.

    Conforms to Datasheets for Datasets (Gebru et al., arXiv:1803.09010).
    """

    card_type: Literal["dataset"] = "dataset"
    dataset_id: str = Field(
        description="Unique dataset identifier.",
    )
    name: str = Field(
        description="Human-readable dataset name.",
    )
    version: str = Field(
        default="1.0.0",
        description="Dataset version identifier.",
    )
    description: str = Field(
        default="",
        description="Description of generated trajectories, entities, and systemic traces.",
    )
    collection_process: str = Field(
        default="Monte Carlo simulation rollout under EWM Engine",
        description="Methodology and generator configuration used to assemble dataset.",
    )
    num_trajectories: int = Field(
        default=0,
        ge=0,
        description="Total rollout trajectories contained in dataset.",
    )
    num_steps: int = Field(
        default=0,
        ge=0,
        description="Total discrete simulation steps across all trajectories.",
    )
    features: list[str] = Field(
        default_factory=list,
        description="List of recorded feature fields, resource timeseries, and state attributes.",
    )
    intended_use: list[str] = Field(
        default_factory=list,
        description="Intended downstream tasks (e.g. offline policy evaluation, dynamics learning).",
    )
    biases_and_limitations: list[str] = Field(
        default_factory=list,
        description="Sampling biases, distribution coverage limits, or simulator artifacts.",
    )
    license: str = Field(
        default="Apache-2.0",
        description="Distribution and reuse license.",
    )
    provenance_fingerprints: list[str] = Field(
        default_factory=list,
        description="Upstream world state and scenario fingerprints from which this dataset was generated.",
    )

    def to_croissant(self) -> dict[str, Any]:
        """Export dataset metadata as standard Croissant 1.1 JSON-LD."""
        from ewm_engine.cards.croissant import to_croissant_dataset

        return to_croissant_dataset(self)
