"""Salesforce Service Cloud & CRM Pipeline read-only bitemporal connector."""

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


class SalesforceConnector(EnterpriseDataConnector):
    """Maps Salesforce Case Queue & Account/Opportunity records into WorldState and BitemporalLedger.

    Expected Salesforce record schema fields:
        - ``CaseId`` (str): Unique Case or Opportunity record ID
        - ``AccountId`` (str): Customer Account identifier (pseudonymized on ingest)
        - ``QueueName`` (str): Support queue or routing tier identifier
        - ``OpenCases`` (float): Active open cases in queue
        - ``AgentCapacity`` (float, optional): Maximum concurrent queue capacity
        - ``SLAViolationRate`` (float, optional): Fraction of cases breaching SLA
        - ``CreatedDateStep`` (float): Event time (t_e) when case transition occurred
        - ``SystemModstampStep`` (float): Knowledge time (t_k >= t_e) when synced to telemetry
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
            case_id = str(rec.get("CaseId", f"sf_case_{idx}"))
            acct_raw = str(rec.get("AccountId", f"acct_{idx}"))
            acct_pseudo = self.auth_config.pseudonymize_id(acct_raw, prefix="sf_acct")
            queue_name = str(rec.get("QueueName", "tier1_support"))
            event_time = float(rec.get("CreatedDateStep", float(idx)))
            knowledge_time = float(rec.get("SystemModstampStep", event_time))
            open_cases = float(rec.get("OpenCases", 0.0))
            agent_cap = float(rec.get("AgentCapacity", 500.0))
            sla_rate = float(rec.get("SLAViolationRate", 0.0))

            bt_rec = self._build_bitemporal_record(
                record_id=f"sf:{case_id}:{queue_name}",
                stream=f"sf_queue:{queue_name}",
                event_time=event_time,
                knowledge_time=knowledge_time,
                payload={
                    "account_id": acct_pseudo,
                    "queue_name": queue_name,
                    "open_cases": open_cases,
                    "agent_capacity": agent_cap,
                    "sla_violation_rate": sla_rate,
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
            acct_id = str(rec.payload["account_id"])
            queue_name = str(rec.payload["queue_name"])
            queue_ent_id = f"sf_queue_{queue_name.lower()}"
            res_id = f"queue_load_{queue_name.lower()}"

            entities[acct_id] = Entity(
                id=acct_id,
                type="salesforce_account",
                attributes={"routed_queue": queue_name},
            )
            entities[queue_ent_id] = Entity(
                id=queue_ent_id,
                type="salesforce_service_queue",
                attributes={
                    "queue_name": queue_name,
                    "sla_violation_rate": float(rec.payload["sla_violation_rate"]),
                },
            )
            resources[res_id] = Resource(
                id=res_id,
                current=max(0.0, float(rec.payload["open_cases"])),
                min_value=0.0,
                max_value=max(1.0, float(rec.payload["agent_capacity"])),
                unit="cases",
            )
            edge_key = (acct_id, queue_ent_id)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                relationships.append(
                    Relationship(
                        source=acct_id,
                        target=queue_ent_id,
                        type="routes_cases_to",
                        attributes={"weight": 1.0},
                    )
                )

        state = WorldState(
            entities=entities,
            relationships=relationships,
            resources=resources,
            context={"connector": "salesforce_service_cloud", "decision_time": decision_time},
            timestamp=decision_time,
        )
        result = ConnectorIngestionResult(
            connector_name="SalesforceConnector",
            records_ingested=ingested,
            entities_mapped=len(entities),
            resources_mapped=len(resources),
            relationships_mapped=len(relationships),
            state_fingerprint=state.fingerprint,
            decision_time=decision_time,
        )
        return state, ledger, result
