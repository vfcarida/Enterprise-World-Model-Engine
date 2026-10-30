# Enterprise Data Connectors Guide (SAP, Salesforce, ServiceNow)

**Module**: `ewm_engine.connectors`  
**Related Protocol**: [Pilot Program Protocol](../pilots/pilot_program_protocol.md)  

---

## 1. Overview

`ewm_engine.connectors` provides read-only, bitemporally disciplined ingestion adapters that map enterprise operational records into canonical `WorldState` snapshots and `BitemporalLedger` streams:

- **`SAPConnector` (`ewm_engine.connectors.sap`)**: Maps SAP S/4HANA Material Management (`MM-IM`) stock movements and multi-echelon plant dependencies (`Plant`, `SupplyingPlant`, `AvailableStock`, `PostingTimestamp` $t_e$, `EntryTimestamp` $t_k$).
- **`SalesforceConnector` (`ewm_engine.connectors.salesforce`)**: Maps Salesforce Service Cloud support queues and customer escalations (`CaseId`, `AccountId`, `QueueName`, `OpenCases`, `CreatedDateStep` $t_e$, `SystemModstampStep` $t_k$).
- **`ServiceNowConnector` (`ewm_engine.connectors.servicenow`)**: Maps ServiceNow ITSM incidents and CMDB Configuration Item (`CI`) dependency graphs (`IncidentNumber`, `ConfigurationItem`, `DependsOnCI`, `ActiveP1Incidents`, `OpenedAtStep` $t_e$, `SysUpdatedAtStep` $t_k$).

---

## 2. Read-Only Airgap & PII Pseudonymization

All connectors enforce two mandatory pilot safety invariants via `ConnectorAuthConfig`:
1. **`read_only_enforced=True`**: Attempting to instantiate a connector with `read_only_enforced=False` raises a `ValueError`.
2. **Salted SHA-256 Entity Pseudonymization**: Raw customer, account, plant, or CI identifiers are hashed via `ConnectorAuthConfig.pseudonymize_id()` before entering `WorldState.entities`.
3. **Point-in-Time Leakage Prevention**: Records with `knowledge_time > decision_time` are automatically withheld by `BitemporalLedger.as_of(decision_time)`, and acausal records (`knowledge_time < event_time`) raise `BitemporalLeakageError`.

---

## 3. Minimal Example

```python
from ewm_engine.connectors import SAPConnector

connector = SAPConnector()
state, ledger, summary = connector.map_to_world_state(
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
assert summary.records_ingested == 1
assert state.resources["stock_dc_east_sku_01"].current == 420.0
```
