"""Read-only enterprise data connectors (SAP, Salesforce, ServiceNow) with bitemporal schema mapping."""

from __future__ import annotations

from ewm_engine.connectors.base import (
    ConnectorAuthConfig,
    ConnectorAuthMode,
    ConnectorIngestionResult,
    EnterpriseDataConnector,
)
from ewm_engine.connectors.salesforce import SalesforceConnector
from ewm_engine.connectors.sap import SAPConnector
from ewm_engine.connectors.servicenow import ServiceNowConnector

__all__ = [
    "ConnectorAuthConfig",
    "ConnectorAuthMode",
    "ConnectorIngestionResult",
    "EnterpriseDataConnector",
    "SAPConnector",
    "SalesforceConnector",
    "ServiceNowConnector",
]
