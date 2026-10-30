"""Integration tests for Enterprise Data Connectors (SAP, Salesforce, ServiceNow)."""

from __future__ import annotations

from tests.unit.test_connectors import (
    test_auth_config_resolves_from_environment,
    test_salesforce_connector_maps_cases_and_sla_breaches,
    test_sap_connector_maps_inventory_and_purchase_orders,
    test_servicenow_connector_maps_incidents_and_change_freeze,
)

__all__ = [
    "test_auth_config_resolves_from_environment",
    "test_salesforce_connector_maps_cases_and_sla_breaches",
    "test_sap_connector_maps_inventory_and_purchase_orders",
    "test_servicenow_connector_maps_incidents_and_change_freeze",
]
