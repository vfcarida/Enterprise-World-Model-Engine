# Reproducibility Cards API Reference

This module defines reproducibility cards (Scenario, Model, Dataset) and Croissant ML metadata export (Track T9).

---

## Card Models

::: ewm_engine.cards.models
    options:
      show_root_heading: true
      show_source: false
      members:
        - BaseCard
        - ScenarioCard
        - ModelCard
        - DatasetCard

---

## Card Renderers

::: ewm_engine.cards.render
    options:
      show_root_heading: true
      show_source: false
      members:
        - render_card_markdown
        - render_card_json
        - render_card_yaml

---

## Croissant Metadata

::: ewm_engine.cards.croissant
    options:
      show_root_heading: true
      show_source: false
      members:
        - to_croissant_json
        - to_croissant_dataset
