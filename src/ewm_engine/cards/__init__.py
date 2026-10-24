"""Native reproducibility cards (ScenarioCard, ModelCard, DatasetCard) and renderers.

Conforms to Track T9 in docs/architecture/platform-tracks.md.
"""

from __future__ import annotations

from ewm_engine.cards.croissant import to_croissant_dataset, to_croissant_json
from ewm_engine.cards.models import (
    BaseCard,
    DatasetCard,
    ModelCard,
    ScenarioCard,
    validate_result_card_dict,
)
from ewm_engine.cards.render import (
    render_card_json,
    render_card_markdown,
    render_card_yaml,
)

__all__ = [
    "BaseCard",
    "DatasetCard",
    "ModelCard",
    "ScenarioCard",
    "render_card_json",
    "render_card_markdown",
    "render_card_yaml",
    "to_croissant_dataset",
    "to_croissant_json",
    "validate_result_card_dict",
]
