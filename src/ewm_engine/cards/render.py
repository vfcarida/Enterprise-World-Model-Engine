"""Rendering engines for native cards (Markdown, YAML, JSON).

Conforms to Track T9: Native Cards (Markdown / YAML / JSON rendering).
Generates beautiful, GitHub-flavored Markdown reports with alerts, tables,
and artifact fingerprints.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from ewm_engine.cards.models import BaseCard, DatasetCard, ModelCard, ScenarioCard


def render_card_json(card: BaseCard, indent: int = 2) -> str:
    """Serialize a card to JSON."""
    return json.dumps(card.model_dump(mode="json"), indent=indent)


def render_card_yaml(card: BaseCard) -> str:
    """Serialize a card to YAML."""
    return str(yaml.safe_dump(card.model_dump(mode="json"), sort_keys=False))


def render_card_markdown(card: BaseCard) -> str:
    """Render a card into structured GitHub-flavored Markdown."""
    lines: list[str] = []

    # Title & Badge
    if card.card_type == "model":
        model_card: ModelCard = card  # type: ignore[assignment]
        lines.append(f"# Model Card: {model_card.model_name} (`{model_card.model_id}`)")
        lines.append("")
        lines.append("> [!IMPORTANT]")
        lines.append(
            f"> **Maturity Level:** `{model_card.maturity.upper()}` | **Version:** `v{model_card.version}`"
        )
    elif card.card_type == "scenario":
        scen_card: ScenarioCard = card  # type: ignore[assignment]
        lines.append(f"# Scenario Card: `{scen_card.scenario_id}`")
        lines.append("")
        lines.append("> [!NOTE]")
        lines.append(
            f"> **Maturity Level:** `{scen_card.maturity.upper()}` | **Horizon:** `{scen_card.horizon} steps` | **Rollouts:** `{scen_card.samples}`"
        )
    elif card.card_type == "dataset":
        ds_card: DatasetCard = card  # type: ignore[assignment]
        lines.append(f"# Dataset Card: {ds_card.name} (`{ds_card.dataset_id}`)")
        lines.append("")
        lines.append("> [!NOTE]")
        lines.append(
            f"> **Trajectories:** `{ds_card.num_trajectories}` | **Total Steps:** `{ds_card.num_steps}` | **License:** `{ds_card.license}`"
        )
    else:
        lines.append(f"# Reproducibility Card: `{card.card_type}`")

    lines.append("")

    # Artifact Cryptographic Fingerprint
    lines.append("### Cryptographic Provenance")
    lines.append(f"- **Artifact Fingerprint (SHA-256):** `{card.artifact_fingerprint}`")
    lines.append(f"- **Schema Version:** `{card.schema_version}`")
    lines.append("")

    # Description
    desc = getattr(card, "description", "")
    if desc:
        lines.append("### Description")
        lines.append(desc)
        lines.append("")

    # Specific sections for ModelCard
    if card.card_type == "model":
        m: ModelCard = card  # type: ignore[assignment]

        if m.intended_use:
            lines.append("### Intended Use")
            for item in m.intended_use:
                lines.append(f"- {item}")
            lines.append("")

        if m.assumptions:
            lines.append("### Structural Assumptions")
            for item in m.assumptions:
                lines.append(f"- {item}")
            lines.append("")

        if m.out_of_scope:
            lines.append("### Out-of-Scope Regimes")
            lines.append("> [!WARNING]")
            lines.append("> The following operating conditions are outside model validity:")
            for item in m.out_of_scope:
                lines.append(f"> - {item}")
            lines.append("")

        if m.constraints_exercised:
            lines.append("### Constraints Exercised")
            for item in m.constraints_exercised:
                lines.append(f"- `{item}`")
            lines.append("")

        if m.metrics:
            lines.append("### Evaluation & Calibration Metrics")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for k, v in sorted(m.metrics.items()):
                lines.append(f"| `{k}` | **{v}** |")
            if m.calibration_score is not None:
                lines.append(f"| `population_calibration_score` | **{m.calibration_score:.4f}** |")
            lines.append("")

        if m.limitations:
            lines.append("### Limitations & Caveats")
            for item in m.limitations:
                lines.append(f"- {item}")
            lines.append("")

        if m.ethical_considerations:
            lines.append("### Ethical Considerations & Responsible AI")
            for item in m.ethical_considerations:
                lines.append(f"- {item}")
            lines.append("")

        if m.quantitative_analyses:
            lines.append("### Quantitative Analyses & Uncertainty Intervals")
            for k, v in sorted(m.quantitative_analyses.items()):
                lines.append(f"- **{k}**: `{v}`")
            lines.append("")

    # Specific sections for ScenarioCard
    elif card.card_type == "scenario":
        sc: ScenarioCard = card  # type: ignore[assignment]

        if sc.interventions:
            lines.append("### Applied Interventions")
            for item in sc.interventions:
                lines.append(f"- `{item}`")
            lines.append("")

        if sc.assumptions:
            lines.append("### Scenario Assumptions")
            for item in sc.assumptions:
                lines.append(f"- {item}")
            lines.append("")

        if sc.environmental_context:
            lines.append("### Environmental Context")
            lines.append("| Parameter | Value |")
            lines.append("|---|---|")
            for k, v in sorted(sc.environmental_context.items()):
                lines.append(f"| `{k}` | `{v}` |")
            lines.append("")

        if sc.metrics:
            lines.append("### Scenario Metrics")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            for k, v in sorted(sc.metrics.items()):
                lines.append(f"| `{k}` | **{v}** |")
            lines.append("")

        if sc.out_of_scope:
            lines.append("### Out-of-Scope Conditions")
            for item in sc.out_of_scope:
                lines.append(f"- {item}")
            lines.append("")

        if sc.limitations:
            lines.append("### Limitations")
            for item in sc.limitations:
                lines.append(f"- {item}")
            lines.append("")

    # Specific sections for DatasetCard
    elif card.card_type == "dataset":
        dc: DatasetCard = card  # type: ignore[assignment]

        lines.append("### Collection Process")
        lines.append(dc.collection_process)
        lines.append("")

        if dc.features:
            lines.append("### Recorded Features")
            for f in dc.features:
                lines.append(f"- `{f}`")
            lines.append("")

        if dc.intended_use:
            lines.append("### Intended Use")
            for item in dc.intended_use:
                lines.append(f"- {item}")
            lines.append("")

        if dc.biases_and_limitations:
            lines.append("### Biases and Known Limitations")
            for item in dc.biases_and_limitations:
                lines.append(f"- {item}")
            lines.append("")

        if dc.provenance_fingerprints:
            lines.append("### Upstream Provenance Fingerprints")
            for fp in dc.provenance_fingerprints:
                lines.append(f"- `{fp}`")
            lines.append("")

    return "\n".join(lines).strip() + "\n"
