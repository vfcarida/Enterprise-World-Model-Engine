# Serialization, WSL v2 & WMIF v1.0 API Reference

This module provides canonical JSON and safe YAML serialization, the declarative World Specification Language (WSL v2) compiler and Language Server Protocol (LSP) diagnostics, and the World Model Interchange Format (WMIF v1.0) conformance suite.

---

## World Specification Language (WSL v2)

::: ewm_engine.serialization.wsl
    options:
      show_root_heading: true
      show_source: false
      members:
        - WSLDocument
        - WSLCompiler
        - parse_wsl_yaml
        - parse_wsl_json
        - compile_wsl_to_world
        - export_world_to_wsl

::: ewm_engine.serialization.wsl_parser
    options:
      show_root_heading: true
      show_source: false
      members:
        - parse_wsl_authored

---

## World Model Interchange Format (WMIF v1.0)

::: ewm_engine.serialization.wmif
    options:
      show_root_heading: true
      show_source: false
      members:
        - WMIFPackage
        - ConformanceReport
        - export_wmif_package
        - import_wmif_package
        - check_wmif_conformance

---

## Canonical JSON & Safe YAML

::: ewm_engine.serialization.json
    options:
      show_root_heading: true
      show_source: false
      members:
        - dumps_canonical_json
        - loads_safe_json

::: ewm_engine.serialization.yaml
    options:
      show_root_heading: true
      show_source: false
      members:
        - StrictSafeLoader
        - safe_load_yaml
        - safe_dump_yaml
