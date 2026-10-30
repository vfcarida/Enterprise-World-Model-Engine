"""Integration tests for Enterprise Data Connectors (SAP, Salesforce, ServiceNow)."""

from __future__ import annotations

from tests.unit.test_connectors import (
    test_connector_auth_config_enforces_readonly_airgap,
    test_salesforce_connector_queue_mapping_and_pseudonymization,
    test_sap_connector_bitemporal_mapping_and_leakage_guard,
    test_servicenow_connector_cmdb_dependency_and_acausal_rejection,
)

__all__ = [
    "test_connector_auth_config_enforces_readonly_airgap",
    "test_salesforce_connector_queue_mapping_and_pseudonymization",
    "test_sap_connector_bitemporal_mapping_and_leakage_guard",
    "test_servicenow_connector_cmdb_dependency_and_acausal_rejection",
]
