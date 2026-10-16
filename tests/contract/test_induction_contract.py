"""Contract tests for WorldSpec induction schemas and InductionReport serialization."""

from __future__ import annotations

import json

import pytest

from ewm_engine.core.spec import WorldSpec
from ewm_engine.induction.connectors import (
    DataFrameSchemaMapping,
    induce_from_dataframe,
)
from ewm_engine.induction.report import InductionReport
from ewm_engine.induction.schema import induce_schema


@pytest.mark.contract
def test_induced_world_spec_contract_compliance() -> None:
    """Verify that induced WorldSpec instances strictly validate against WorldSpec Pydantic schema."""
    data = [
        {"entity_id": "machine_alpha", "timestamp": 1, "temperature": 75.0, "rpm": 1200},
        {"entity_id": "machine_beta", "timestamp": 2, "temperature": 82.5, "rpm": 1400},
    ]

    spec, _report = induce_from_dataframe(
        df=data,
        schema=DataFrameSchemaMapping(
            entity_id_col="entity_id",
            timestamp_col="timestamp",
            resource_cols=["temperature", "rpm"],
        ),
        name="ContractValidatedWorld",
    )

    # Must be valid Pydantic WorldSpec
    assert isinstance(spec, WorldSpec)
    dumped = spec.model_dump(mode="json")
    reloaded = WorldSpec.model_validate(dumped)
    assert len(reloaded.entities) == len(spec.entities)
    assert len(reloaded.resources) == len(spec.resources)


@pytest.mark.contract
def test_induction_report_contract_and_roundtrip() -> None:
    """Verify that InductionReport serializes to valid JSON, computes fingerprint, and round-trips."""
    ddl = "CREATE TABLE assets (id TEXT PRIMARY KEY, value REAL CHECK (value >= 0));"
    _spec, report = induce_schema(ddl=ddl, name="ContractAssetWorld", seed=123)

    assert isinstance(report, InductionReport)
    assert report.sha256_fingerprint != ""
    assert len(report.sha256_fingerprint) == 64

    # JSON round-trip
    rep_json = report.to_json()
    rep_data = json.loads(rep_json)
    reloaded_report = InductionReport.model_validate(rep_data)

    assert reloaded_report.sha256_fingerprint == report.sha256_fingerprint
    assert reloaded_report.entities_count == report.entities_count
    assert reloaded_report.resources_count == report.resources_count
