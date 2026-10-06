#!/usr/bin/env python3
"""Script to generate versioned Draft 2020-12 JSON Schemas for EWM Engine models.

Usage:
    uv run python scripts/generate_schemas.py          # Generate/update schemas in schemas/
    uv run python scripts/generate_schemas.py --check  # Verify committed schemas match code without drift
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ewm_engine.cards.models import DatasetCard, ModelCard, ScenarioCard
from ewm_engine.core.actions import Action
from ewm_engine.core.spec import WorldSpec
from ewm_engine.core.state import WorldState
from ewm_engine.durability.events import TransitionEvent
from ewm_engine.experimentation.params import ParameterSpace
from ewm_engine.provenance.metadata import Provenance
from ewm_engine.reporting.model import ReportModel
from ewm_engine.simulation.scenario import Scenario
from ewm_engine.simulation.trajectory import Trajectory
from ewm_engine.verification.spec import PropertySpec

ROOT_DIR = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = ROOT_DIR / "schemas"

SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "action.schema.json": Action,
    "world-state.schema.json": WorldState,
    "world.schema.json": WorldSpec,
    "scenario.schema.json": Scenario,
    "trajectory.schema.json": Trajectory,
    "provenance.schema.json": Provenance,
    "transition-event.schema.json": TransitionEvent,
    "scenario-card.schema.json": ScenarioCard,
    "model-card.schema.json": ModelCard,
    "dataset-card.schema.json": DatasetCard,
    "property-spec.schema.json": PropertySpec,
    "parameter-space.schema.json": ParameterSpace,
    "report-model.schema.json": ReportModel,
}


def build_json_schema(model_cls: type[BaseModel], filename: str) -> dict[str, Any]:
    """Generate Draft 2020-12 schema with stable $id and const schema_version."""
    schema = model_cls.model_json_schema()

    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://ewm-engine.org/schemas/v1/{filename}"

    # Ensure schema_version is enforced with const
    if "properties" in schema and "schema_version" in schema["properties"]:
        schema["properties"]["schema_version"]["const"] = "1.0.0"
        # Also ensure schema_version is in required if required exists
        if "required" in schema:
            if "schema_version" not in schema["required"]:
                schema["required"].append("schema_version")
                schema["required"].sort()

    return schema


def format_schema(schema_dict: dict[str, Any]) -> str:
    """Format schema dict into deterministic, canonical formatted JSON with trailing newline."""
    return json.dumps(schema_dict, indent=2, sort_keys=True) + "\n"


def generate_all() -> dict[str, str]:
    """Generate canonical JSON strings for all registered schemas."""
    return {
        filename: format_schema(build_json_schema(model_cls, filename))
        for filename, model_cls in SCHEMA_REGISTRY.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and verify EWM Engine JSON Schemas.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that committed schemas match current Pydantic models (fail if drift detected).",
    )
    args = parser.parse_args()

    generated = generate_all()

    if args.check:
        drift_detected = False
        for filename, expected_content in generated.items():
            file_path = SCHEMAS_DIR / filename
            if not file_path.exists():
                print(f"ERROR: Schema file missing: {file_path}", file=sys.stderr)
                drift_detected = True
                continue

            actual_content = file_path.read_text(encoding="utf-8")
            if actual_content != expected_content:
                print(f"ERROR: Schema drift detected in {filename}:", file=sys.stderr)
                diff = difflib.unified_diff(
                    actual_content.splitlines(keepends=True),
                    expected_content.splitlines(keepends=True),
                    fromfile=f"committed/{filename}",
                    tofile=f"generated/{filename}",
                )
                sys.stderr.writelines(diff)
                drift_detected = True

        if drift_detected:
            print(
                "\nSchema check FAILED. Run 'uv run python scripts/generate_schemas.py' to regenerate.",
                file=sys.stderr,
            )
            return 1

        print(f"Schema check PASSED: all {len(generated)} schemas match generated models.")
        return 0

    # Write files
    SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, content in generated.items():
        file_path = SCHEMAS_DIR / filename
        file_path.write_text(content, encoding="utf-8")
        print(f"Wrote schema: {file_path.relative_to(ROOT_DIR)}")

    print(
        f"Successfully generated {len(generated)} JSON Schemas in {SCHEMAS_DIR.relative_to(ROOT_DIR)}/."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
