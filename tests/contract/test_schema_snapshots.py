"""Contract test (AC-003): Committed JSON Schemas match current models with no drift."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_schemas import (
    SCHEMA_REGISTRY,
    SCHEMAS_DIR,
    build_json_schema,
    generate_all,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@pytest.mark.contract
def test_all_expected_schemas_exist_on_disk() -> None:
    """AC-003: All declared public schemas must exist in schemas/ directory."""
    assert SCHEMAS_DIR.exists(), f"Schemas directory missing: {SCHEMAS_DIR}"

    for filename in SCHEMA_REGISTRY:
        schema_path = SCHEMAS_DIR / filename
        assert schema_path.exists(), f"Committed schema file missing: {filename}"
        assert schema_path.stat().st_size > 0, f"Schema file is empty: {filename}"


@pytest.mark.contract
def test_schema_snapshots_match_generated_models_without_drift() -> None:
    """AC-003: Committed schema files must strictly match freshly generated schemas."""
    expected_schemas = generate_all()

    for filename, expected_json in expected_schemas.items():
        file_path = SCHEMAS_DIR / filename
        assert file_path.exists(), f"Schema file {filename} does not exist."
        actual_json = file_path.read_text(encoding="utf-8")

        assert actual_json == expected_json, (
            f"Schema drift detected in {filename}! "
            f"Run 'uv run python scripts/generate_schemas.py' to update committed schemas."
        )


@pytest.mark.contract
def test_schema_metadata_conformance() -> None:
    """AC-003: Every schema must specify Draft 2020-12, stable $id, and const schema_version."""
    for filename, model_cls in SCHEMA_REGISTRY.items():
        schema = build_json_schema(model_cls, filename)

        # Draft 2020-12
        assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema", (
            f"{filename} must declare Draft 2020-12 $schema."
        )

        # Stable $id
        expected_id = f"https://ewm-engine.org/schemas/v1/{filename}"
        assert schema.get("$id") == expected_id, (
            f"{filename} $id must be {expected_id}, got {schema.get('$id')}"
        )

        # const schema_version
        assert "properties" in schema, f"{filename} missing properties."
        assert "schema_version" in schema["properties"], (
            f"{filename} missing schema_version property."
        )
        assert schema["properties"]["schema_version"].get("const") == "1.0.0", (
            f"{filename} schema_version must have const '1.0.0'."
        )
