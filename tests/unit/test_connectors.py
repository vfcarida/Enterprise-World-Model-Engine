"""Unit tests for SAP, Salesforce, and ServiceNow read-only bitemporal data connectors."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from ewm_engine.connectors import (
    ConnectorAuthConfig,
    ConnectorAuthMode,
    SalesforceConnector,
    SAPConnector,
    ServiceNowConnector,
)
from ewm_engine.core.bitemporal import BitemporalLeakageError


def test_connector_auth_config_enforces_readonly_airgap() -> None:
    with pytest.raises(ValueError, match="read_only_enforced=True"):
        ConnectorAuthConfig(read_only_enforced=False)

    with pytest.raises(ValueError, match="OAUTH2_CLIENT_CREDENTIALS"):
        ConnectorAuthConfig(
            auth_mode=ConnectorAuthMode.OAUTH2_CLIENT_CREDENTIALS,
            client_id="sap_client",
        )

    with pytest.raises(ValueError, match="BEARER_TOKEN"):
        ConnectorAuthConfig(auth_mode=ConnectorAuthMode.BEARER_TOKEN)

    cfg = ConnectorAuthConfig(
        auth_mode=ConnectorAuthMode.OAUTH2_CLIENT_CREDENTIALS,
        client_id="sap_client",
        client_secret=SecretStr("secret-token"),
        pii_salt="pilot-salt-01",
    )
    assert cfg.pseudonymize_id("ACCT-9981", prefix="acct") == cfg.pseudonymize_id(
        "ACCT-9981", prefix="acct"
    )
    assert "ACCT-9981" not in cfg.pseudonymize_id("ACCT-9981", prefix="acct")


def test_sap_connector_bitemporal_mapping_and_leakage_guard() -> None:
    connector = SAPConnector()
    records = [
        {
            "MaterialDocument": "490001",
            "Plant": "DC_EAST",
            "SupplyingPlant": "FACTORY_01",
            "Material": "SKU_COLD_01",
            "AvailableStock": 420.0,
            "MaxCapacity": 1000.0,
            "ReorderPoint": 150.0,
            "PostingTimestamp": 1.0,
            "EntryTimestamp": 2.0,
        },
        {
            "MaterialDocument": "490002",
            "Plant": "DC_EAST",
            "SupplyingPlant": "FACTORY_01",
            "Material": "SKU_COLD_01",
            "AvailableStock": 280.0,
            "MaxCapacity": 1000.0,
            "ReorderPoint": 150.0,
            "PostingTimestamp": 3.0,
            "EntryTimestamp": 6.0,
        },
    ]

    # At decision_time=4.0, second record has EntryTimestamp (knowledge_time)=6.0 > 4.0 and must be withheld
    state_t4, ledger, result_t4 = connector.map_to_world_state(records, decision_time=4.0)
    assert result_t4.records_ingested == 2
    assert result_t4.entities_mapped == 2
    assert result_t4.resources_mapped == 1
    assert result_t4.relationships_mapped == 1
    assert state_t4.resources["stock_dc_east_sku_cold_01"].current == 420.0
    assert ledger.as_of(4.0).withheld_future_knowledge_count == 1

    # At decision_time=6.0, second record becomes visible
    state_t6, _, _ = connector.map_to_world_state(records, decision_time=6.0)
    assert state_t6.resources["stock_dc_east_sku_cold_01"].current == 280.0


def test_salesforce_connector_queue_mapping_and_pseudonymization() -> None:
    cfg = ConnectorAuthConfig(
        auth_mode=ConnectorAuthMode.BEARER_TOKEN,
        bearer_token=SecretStr("sf-readonly-token"),
        pii_salt="sf-salt",
    )
    connector = SalesforceConnector(auth_config=cfg)
    records = [
        {
            "CaseId": "50030001",
            "AccountId": "001_RAW_CUSTOMER_PII",
            "QueueName": "Fraud_Decline_Escalations",
            "OpenCases": 145.0,
            "AgentCapacity": 200.0,
            "SLAViolationRate": 0.08,
            "CreatedDateStep": 2.0,
            "SystemModstampStep": 2.5,
        }
    ]
    state, _ledger, res = connector.map_to_world_state(records, decision_time=3.0)
    assert res.records_ingested == 1
    assert res.entities_mapped == 2
    assert res.resources_mapped == 1
    assert res.relationships_mapped == 1
    assert "001_RAW_CUSTOMER_PII" not in state.entities
    assert state.resources["queue_load_fraud_decline_escalations"].current == 145.0


def test_servicenow_connector_cmdb_dependency_and_acausal_rejection() -> None:
    connector = ServiceNowConnector()
    records = [
        {
            "IncidentNumber": "INC00101",
            "ConfigurationItem": "checkout_api",
            "DependsOnCI": "ledger_db",
            "ActiveP1Incidents": 3.0,
            "ErrorBudgetCapacity": 10.0,
            "OpenedAtStep": 1.0,
            "SysUpdatedAtStep": 1.5,
        }
    ]
    state, _ledger, res = connector.map_to_world_state(records, decision_time=2.0)
    assert res.entities_mapped == 2
    assert res.relationships_mapped == 1
    assert state.resources["incident_load_checkout_api"].current == 3.0

    # Acausal record (SysUpdatedAtStep < OpenedAtStep) must fail closed with BitemporalLeakageError
    bad_records = [
        {
            "IncidentNumber": "INC00102",
            "ConfigurationItem": "checkout_api",
            "ActiveP1Incidents": 5.0,
            "OpenedAtStep": 5.0,
            "SysUpdatedAtStep": 3.0,
        }
    ]
    with pytest.raises(BitemporalLeakageError):
        connector.map_to_world_state(bad_records, decision_time=6.0)
