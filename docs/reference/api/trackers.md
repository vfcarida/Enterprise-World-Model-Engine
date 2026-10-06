# Experiment Trackers API Reference

This module defines experiment tracker protocols, zero-dependency local JSON stores, hierarchical configuration loaders, and MLflow/W&B adapters (Track T9).

---

## Tracker Protocol & Local JSON

::: ewm_engine.trackers.protocol
    options:
      show_root_heading: true
      show_source: false
      members:
        - TrackerBackend

::: ewm_engine.trackers.local_json
    options:
      show_root_heading: true
      show_source: false
      members:
        - LocalJsonTracker

---

## Configuration Loaders

::: ewm_engine.trackers.omegaconf_adapter
    options:
      show_root_heading: true
      show_source: false
      members:
        - OmegaConfConfigLoader

---

## Enterprise Tracker Adapters

::: ewm_engine.trackers.adapters
    options:
      show_root_heading: true
      show_source: false
      members:
        - MLflowTracker
        - WandbTracker
        - DvcCliTracker
