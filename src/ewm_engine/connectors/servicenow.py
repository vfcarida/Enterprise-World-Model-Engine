"""ServiceNow ITSM & Change-Management read-only bitemporal connector."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ewm_engine.connectors.base import (
    ConnectorAuthConfig,
    ConnectorIngestionResult,
    EnterpriseDataConnector,
)
from ewm_engine.core.bitemporal import BitemporalLedger
from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState


class ServiceNowConnector(EnterpriseDataConnector):
    """Maps ServiceNow Incident and CMDB Configuration Item (CI) records into WorldState and BitemporalLedger.

    Expected ServiceNow record schema fields:
        - ``IncidentNumber`` (str): Unique incident or change ticket ID (e.g. ``INC001001``)
        - ``ConfigurationItem`` (str): Affected CMDB service/CI name
        - ``DependsOnCI`` (str, optional): Upstream CMDB dependency CI name
        - ``ActiveP1Incidents`` (float): Count of active severity-1 incidents on the CI
        - ``ErrorBudgetCapacity`` (float, optional): Max allowable incident budget before freeze
        - ``OpenedAtStep`` (float): Event time (t_e) when incident occurred
        - ``SysUpdatedAtStep`` (float): Knowledge time (t_k >= t_e) when logged in ServiceNow
    """

    def __init__(self, auth_config: ConnectorAuthConfig | None = None) -> None:
        super().__init__(auth_config)

    def ingest_to_ledger(
        self,
        raw_records: Sequence[Mapping[str, Any]],
        ledger: BitemporalLedger,
    ) -> int:
        if not self.is_authenticated:
            self.authenticate()

        ingested = 0
        for idx, rec in enumerate(raw_records):
            inc_id = str(rec.get("IncidentNumber", f"INC{idx:06d}"))
            ci_name = str(rec.get("ConfigurationItem", "ci_core_service"))
            dep_ci = str(rec.get("DependsOnCI", ""))
            event_time = float(rec.get("OpenedAtStep", float(idx)))
            knowledge_time = float(rec.get("SysUpdatedAtStep", event_time))
            active_p1 = float(rec.get("ActiveP1Incidents", 0.0))
            budget_cap = float(rec.get("ErrorBudgetCapacity", 50.0))

            bt_rec = self._build_bitemporal_record(
                record_id=f"snow:{inc_id}:{ci_name}",
                stream=f"snow_cmdb:{ci_name}",
                event_time=event_time,
                knowledge_time=knowledge_time,
                payload={
                    "ci_name": ci_name,
                    "depends_on_ci": dep_ci,
                    "active_p1_incidents": active_p1,
                    "error_budget_capacity": budget_cap,
                },
            )
            ledger.append(bt_rec)
            ingested += 1
        return ingested

    def map_to_world_state(
        self,
        raw_records: Sequence[Mapping[str, Any]],
        *,
        decision_time: float,
    ) -> tuple[WorldState, BitemporalLedger, ConnectorIngestionResult]:
        ledger = BitemporalLedger()
        ingested = self.ingest_to_ledger(raw_records, ledger)
        pit = ledger.as_of(decision_time)

        entities: dict[str, Entity] = {}
        resources: dict[str, Resource] = {}
        relationships: list[Relationship] = []
        seen_edges: set[tuple[str, str]] = set()

        for rec in pit.effective_records():
            ci_name = str(rec.payload["ci_name"])
            ci_id = self.auth_config.pseudonymize_id(ci_name, prefix="snow_ci")
            res_id = f"incident_load_{ci_name.lower()}"

            entities[ci_id] = Entity(
                id=ci_id,
                type="servicenow_cmdb_ci",
                attributes={
                    "ci_name": ci_name,
                    "active_p1_incidents": float(rec.payload["active_p1_incidents"]),
                },
            )
            resources[res_id] = Resource(
                id=res_id,
                current=max(0.0, float(rec.payload["active_p1_incidents"])),
                min_value=0.0,
                max_value=max(1.0, float(rec.payload["error_budget_capacity"])),
                unit="incidents",
            )
            dep_ci = str(rec.payload.get("depends_on_ci", ""))
            if dep_ci:
                dep_id = self.auth_config.pseudonymize_id(dep_ci, prefix="snow_ci")
                if dep_id not in entities:
                    entities[dep_id] = Entity(
                        id=dep_id,
                        type="servicenow_cmdb_ci",
                        attributes={"ci_name": dep_ci},
                    )
                edge_key = (dep_id, ci_id)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    relationships.append(
                        Relationship(
                            source=dep_id,
                            target=ci_id,
                            type="cmdb_depends_on",
                            attributes={"weight": 1.0},
                        )
                    )

        state = WorldState(
            entities=entities,
            relationships=relationships,
            resources=resources,
            context={"connector": "servicenow_itsm_cmdb", "decision_time": decision_time},
            timestamp=decision_time,
        )
        result = ConnectorIngestionResult(
            connector_name="ServiceNowConnector",
            records_ingested=ingested,
            entities_mapped=len(entities),
            resources_mapped=len(resources),
            relationships_mapped=len(relationships),
            state_fingerprint=state.fingerprint,
            decision_time=decision_time,
        )
        return state, ledger, result
