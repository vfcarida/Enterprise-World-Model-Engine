# Enterprise Data Connectors Tutorial (SAP, Salesforce, ServiceNow)

**Module**: `ewm_engine.connectors`  
**Integration Tests**: [`tests/integration/test_connectors.py`](../../tests/integration/test_connectors.py)  
**Pilot Protocol**: [`docs/pilots/pilot_program_protocol.md`](../pilots/pilot_program_protocol.md)  

---

## 1. Overview

`ewm_engine.connectors` provides read-only, bitemporally disciplined ingestion adapters that map enterprise operational records into canonical `WorldState` snapshots and `BitemporalLedger` streams:

- **`SAPConnector` ([`src/ewm_engine/connectors/sap.py`](../../src/ewm_engine/connectors/sap.py))**: OAuth2 / Bearer / Offline snapshot authentication; maps SAP S/4HANA Material Management (`MM-IM`) stock movements and multi-echelon plant dependencies (`Plant`, `SupplyingPlant`, `AvailableStock`, `PostingTimestamp` $t_e$, `EntryTimestamp` $t_k$).
- **`SalesforceConnector` ([`src/ewm_engine/connectors/salesforce.py`](../../src/ewm_engine/connectors/salesforce.py))**: OAuth2 / Bearer authentication; maps Salesforce Service Cloud support queues, opportunities/accounts, and customer escalations (`CaseId`, `AccountId`, `QueueName`, `OpenCases`, `CreatedDateStep` $t_e$, `SystemModstampStep` $t_k$).
- **`ServiceNowConnector` ([`src/ewm_engine/connectors/servicenow.py`](../../src/ewm_engine/connectors/servicenow.py))**: Read-only Basic Auth (`BASIC_READONLY`) / OAuth2; maps ServiceNow ITSM incidents/problems and CMDB Configuration Item (`CI`) dependency graphs (`IncidentNumber`, `ConfigurationItem`, `DependsOnCI`, `ActiveP1Incidents`, `OpenedAtStep` $t_e$, `SysUpdatedAtStep` $t_k$).

---

## 2. Read-Only Airgap, PII Pseudonymization & Leakage Guards

All three connectors enforce mandatory pilot safety invariants via `ConnectorAuthConfig`:
1. **`read_only_enforced=True`**: Attempting to instantiate a connector with `read_only_enforced=False` raises a `ValueError`.
2. **Salted SHA-256 Entity Pseudonymization**: Raw customer, account, plant, or CI identifiers are hashed via `ConnectorAuthConfig.pseudonymize_id()` before entering `WorldState.entities`.
3. **Point-in-Time Leakage Prevention**: Records with `knowledge_time > decision_time` are automatically withheld by `BitemporalLedger.as_of(decision_time)`, and acausal records (`knowledge_time < event_time`) raise `BitemporalLeakageError`.

---

## 3. End-to-End Walkthrough

```python
from pydantic import SecretStr
from ewm_engine.connectors import (
    ConnectorAuthConfig,
    ConnectorAuthMode,
    SAPConnector,
    SalesforceConnector,
    ServiceNowConnector,
)

# 1. SAP S/4HANA Inventory Ingestion with Bitemporal As-Of Filtering
sap = SAPConnector()
state_sap, ledger_sap, res_sap = sap.map_to_world_state(
    [
        {
            "MaterialDocument": "490001",
            "Plant": "DC_EAST",
            "SupplyingPlant": "FACTORY_01",
            "Material": "SKU_01",
            "AvailableStock": 420.0,
            "MaxCapacity": 1000.0,
            "PostingTimestamp": 1.0,
            "EntryTimestamp": 2.0,
        }
    ],
    decision_time=3.0,
)
assert state_sap.resources["stock_dc_east_sku_01"].current == 420.0

# 2. Salesforce Queue & Account Ingestion with Salted PII Pseudonymization
sf = SalesforceConnector(
    auth_config=ConnectorAuthConfig(
        auth_mode=ConnectorAuthMode.BEARER_TOKEN,
        bearer_token=SecretStr("sf-readonly-token"),
        pii_salt="pilot-salt",
    )
)
state_sf, _, res_sf = sf.map_to_world_state(
    [
        {
            "CaseId": "50030001",
            "AccountId": "001_RAW_PII",
            "QueueName": "Fraud_Decline_Escalations",
            "OpenCases": 145.0,
            "AgentCapacity": 200.0,
            "CreatedDateStep": 2.0,
            "SystemModstampStep": 2.5,
        }
    ],
    decision_time=3.0,
)
assert "001_RAW_PII" not in state_sf.entities

# 3. ServiceNow Incident & CMDB Dependency Mapping
snow = ServiceNowConnector(
    auth_config=ConnectorAuthConfig(auth_mode=ConnectorAuthMode.BASIC_READONLY)
)
state_snow, _, res_snow = snow.map_to_world_state(
    [
        {
            "IncidentNumber": "INC00101",
            "ConfigurationItem": "checkout_api",
            "DependsOnCI": "ledger_db",
            "ActiveP1Incidents": 3.0,
            "ErrorBudgetCapacity": 10.0,
            "OpenedAtStep": 1.0,
            "SysUpdatedAtStep": 1.5,
        }
    ],
    decision_time=2.0,
)
assert res_snow.relationships_mapped == 1
```
