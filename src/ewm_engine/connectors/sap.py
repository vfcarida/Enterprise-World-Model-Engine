"""SAP S/4HANA OData (MM-IM / Supply Chain & Inventory) read-only bitemporal connector."""

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


class SAPConnector(EnterpriseDataConnector):
    """Maps SAP S/4HANA material document and plant stock snapshots into WorldState and BitemporalLedger.

    Expected SAP record schema fields:
        - ``MaterialDocument`` (str): Unique document identifier
        - ``Plant`` (str): Warehouse/plant echelon code
        - ``SupplyingPlant`` (str, optional): Upstream echelon plant code
        - ``Material`` (str): Material SKU identifier
        - ``AvailableStock`` (float): Unrestricted-use stock quantity
        - ``MaxCapacity`` (float, optional): Warehouse bin capacity ceiling (default 10000.0)
        - ``ReorderPoint`` (float, optional): Safety-stock reorder threshold
        - ``PostingTimestamp`` (float): Event time (t_e) when goods movement occurred
        - ``EntryTimestamp`` (float): Knowledge time (t_k >= t_e) when posted in SAP ledger
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
            doc_id = str(rec.get("MaterialDocument", f"sap_doc_{idx}"))
            plant = str(rec.get("Plant", "PLANT_01"))
            sku = str(rec.get("Material", "SKU_01"))
            event_time = float(rec.get("PostingTimestamp", float(idx)))
            knowledge_time = float(rec.get("EntryTimestamp", event_time))
            stock = float(rec.get("AvailableStock", 0.0))
            max_cap = float(rec.get("MaxCapacity", 10000.0))
            reorder = float(rec.get("ReorderPoint", 100.0))

            bt_rec = self._build_bitemporal_record(
                record_id=f"sap:{doc_id}:{plant}:{sku}",
                stream=f"sap_inventory:{plant}:{sku}",
                event_time=event_time,
                knowledge_time=knowledge_time,
                payload={
                    "plant": plant,
                    "material": sku,
                    "available_stock": stock,
                    "max_capacity": max_cap,
                    "reorder_point": reorder,
                    "supplying_plant": str(rec.get("SupplyingPlant", "")),
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
            plant_code = str(rec.payload["plant"])
            sku_code = str(rec.payload["material"])
            plant_id = self.auth_config.pseudonymize_id(plant_code, prefix="sap_plant")
            res_id = f"stock_{plant_code.lower()}_{sku_code.lower()}"

            entities[plant_id] = Entity(
                id=plant_id,
                type="sap_plant",
                attributes={
                    "plant_code": plant_code,
                    "latest_sku": sku_code,
                    "reorder_point": float(rec.payload["reorder_point"]),
                },
            )
            resources[res_id] = Resource(
                id=res_id,
                current=max(0.0, float(rec.payload["available_stock"])),
                min_value=0.0,
                max_value=max(1.0, float(rec.payload["max_capacity"])),
                unit="units",
            )
            sup_code = str(rec.payload.get("supplying_plant", ""))
            if sup_code:
                sup_id = self.auth_config.pseudonymize_id(sup_code, prefix="sap_plant")
                if sup_id not in entities:
                    entities[sup_id] = Entity(
                        id=sup_id,
                        type="sap_plant",
                        attributes={"plant_code": sup_code},
                    )
                edge_key = (sup_id, plant_id)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    relationships.append(
                        Relationship(
                            source=sup_id,
                            target=plant_id,
                            type="supplies_echelon",
                            attributes={"weight": 1.0},
                        )
                    )

        state = WorldState(
            entities=entities,
            relationships=relationships,
            resources=resources,
            context={"connector": "sap_s4hana_mm", "decision_time": decision_time},
            timestamp=decision_time,
        )
        result = ConnectorIngestionResult(
            connector_name="SAPConnector",
            records_ingested=ingested,
            entities_mapped=len(entities),
            resources_mapped=len(resources),
            relationships_mapped=len(relationships),
            state_fingerprint=state.fingerprint,
            decision_time=decision_time,
        )
        return state, ledger, result
