"""Heterogeneous temporal relational graph view over WorldState (v2.0 candidate).

Formalizes enterprise state as a multi-relational, typed graph:
G_t = (V_t, E_t, R, T, X_v, X_e)
where:
- V_t is the set of heterogeneous entity nodes partitioned by Entity.type.
- E_t is the set of multi-relational edges partitioned by (src_type, rel_type, dst_type).
- X_v is the node feature space combining entity attributes and attached resources.
- X_e is the edge feature space combining relationship attributes and weights.
- Temporal validity intervals [t_start, t_end) define active windows.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.entities import Entity, Relationship
from ewm_engine.core.resources import Resource
from ewm_engine.core.state import WorldState
from ewm_engine.core.types import EntityId, ResourceId


class GraphNode(BaseModel):
    """A heterogeneous node in the graph view, derived from an Entity and its attached Resources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: EntityId = Field(description="Unique entity identifier.")
    type: str = Field(description="Categorical entity classification.")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Immutable node attributes.",
    )
    resources: dict[ResourceId, float] = Field(
        default_factory=dict,
        description="Current values of measurable resources attached to this entity.",
    )
    tags: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Categorical entity tags.",
    )
    temporal_valid_from: float | None = Field(
        default=None,
        description="Optional simulation timestamp when this node becomes active.",
    )
    temporal_valid_until: float | None = Field(
        default=None,
        description="Optional simulation timestamp when this node expires.",
    )

    def is_active_at(self, timestamp: float) -> bool:
        """Check if this node is active at the given simulation timestamp."""
        if self.temporal_valid_from is not None and timestamp < self.temporal_valid_from:
            return False
        if self.temporal_valid_until is not None and timestamp >= self.temporal_valid_until:
            return False
        return True


class GraphEdge(BaseModel):
    """A directed, typed relational edge between two heterogeneous entity nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: EntityId = Field(description="Source entity identifier.")
    target: EntityId = Field(description="Target entity identifier.")
    type: str = Field(description="Relational edge type.")
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Immutable edge attributes.",
    )
    weight: float = Field(
        default=1.0,
        description="Scalar edge weight or capacity.",
    )
    temporal_valid_from: float | None = Field(
        default=None,
        description="Optional timestamp when this edge becomes active.",
    )
    temporal_valid_until: float | None = Field(
        default=None,
        description="Optional timestamp when this edge expires.",
    )

    def is_active_at(self, timestamp: float) -> bool:
        """Check if this edge is active at the given simulation timestamp."""
        if self.temporal_valid_from is not None and timestamp < self.temporal_valid_from:
            return False
        if self.temporal_valid_until is not None and timestamp >= self.temporal_valid_until:
            return False
        return True


class HeterogeneousGraphView(BaseModel):
    """Additive, multi-relational graph projection over a WorldState.

    Enables relational message passing, GNN dynamics, and network topology analysis
    without altering the underlying immutable WorldState contract.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(default=0, description="Simulation discrete step index.")
    timestamp: float = Field(default=0.0, description="Continuous simulation time.")
    schema_version: str = Field(default="2.0.0", description="Graph representation schema version.")
    nodes_by_type: dict[str, dict[EntityId, GraphNode]] = Field(
        default_factory=dict,
        description="Entity nodes partitioned by categorical node type.",
    )
    edges_by_type: dict[str, list[GraphEdge]] = Field(
        default_factory=dict,
        description="Directed relational edges keyed by canonical relation key: 'src_type__rel_type__dst_type'.",
    )
    global_attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Global state attributes (memory, active rules, context).",
    )

    @classmethod
    def from_world_state(cls, state: WorldState) -> HeterogeneousGraphView:
        """Construct an additive heterogeneous graph view from a canonical WorldState."""
        # 1. Map resources to entities
        resources_by_entity: dict[EntityId, dict[ResourceId, float]] = {}
        for r_id, res in state.resources.items():
            if res.entity_id:
                resources_by_entity.setdefault(res.entity_id, {})[r_id] = float(res.current)

        # 2. Partition nodes by Entity.type
        nodes_by_type: dict[str, dict[EntityId, GraphNode]] = {}
        entity_to_type: dict[EntityId, str] = {}

        for e_id, entity in state.entities.items():
            e_type = entity.type
            entity_to_type[e_id] = e_type

            # Check for temporal metadata in entity attributes
            t_from = (
                float(entity.attributes["temporal_valid_from"])
                if "temporal_valid_from" in entity.attributes
                else None
            )
            t_until = (
                float(entity.attributes["temporal_valid_until"])
                if "temporal_valid_until" in entity.attributes
                else None
            )

            attached_res = resources_by_entity.get(e_id, {})
            node = GraphNode(
                id=e_id,
                type=e_type,
                attributes=entity.attributes,
                resources=attached_res,
                tags=entity.tags,
                temporal_valid_from=t_from,
                temporal_valid_until=t_until,
            )
            nodes_by_type.setdefault(e_type, {})[e_id] = node

        # 3. Partition edges by typed canonical key: 'src_type__rel_type__dst_type'
        edges_by_type: dict[str, list[GraphEdge]] = {}
        for rel in state.relationships:
            src_type = entity_to_type.get(rel.source, "unknown")
            dst_type = entity_to_type.get(rel.target, "unknown")
            edge_key = f"{src_type}__{rel.type}__{dst_type}"

            weight = float(rel.attributes.get("weight", 1.0))
            t_from = (
                float(rel.attributes["temporal_valid_from"])
                if "temporal_valid_from" in rel.attributes
                else None
            )
            t_until = (
                float(rel.attributes["temporal_valid_until"])
                if "temporal_valid_until" in rel.attributes
                else None
            )

            edge = GraphEdge(
                source=rel.source,
                target=rel.target,
                type=rel.type,
                attributes=rel.attributes,
                weight=weight,
                temporal_valid_from=t_from,
                temporal_valid_until=t_until,
            )
            edges_by_type.setdefault(edge_key, []).append(edge)

        global_attrs = {
            "memory": state.memory,
            "active_rules": state.active_rules,
            "context": state.context,
        }

        return cls(
            step=state.step,
            timestamp=state.timestamp,
            schema_version="2.0.0",
            nodes_by_type=nodes_by_type,
            edges_by_type=edges_by_type,
            global_attributes=global_attrs,
        )

    def to_world_state(self, base_state: WorldState | None = None) -> WorldState:
        """Reconstruct a canonical WorldState from this heterogeneous graph view.

        If base_state is provided, unattached resources and non-graph metadata are preserved.
        """
        entities: list[Entity] = []
        resources_dict: dict[ResourceId, Resource] = (
            copy.deepcopy(base_state.resources) if base_state is not None else {}
        )

        for _n_type, nodes in self.nodes_by_type.items():
            for _n_id, node in nodes.items():
                e = Entity(
                    id=node.id,
                    type=node.type,
                    attributes=node.attributes,
                    tags=node.tags,
                )
                entities.append(e)

                # Update or register attached resources
                for r_id, current_val in node.resources.items():
                    if r_id in resources_dict:
                        existing = resources_dict[r_id]
                        resources_dict[r_id] = existing.model_copy(
                            update={"current": current_val, "entity_id": node.id}
                        )
                    else:
                        resources_dict[r_id] = Resource(
                            id=r_id,
                            current=current_val,
                            min_value=0.0,
                            max_value=max(1000.0, current_val * 2.0),
                            entity_id=node.id,
                        )

        relationships: list[Relationship] = []
        for _rel_key, edges in self.edges_by_type.items():
            for edge in edges:
                r = Relationship(
                    source=edge.source,
                    target=edge.target,
                    type=edge.type,
                    attributes=edge.attributes,
                )
                relationships.append(r)

        memory = self.global_attributes.get("memory", base_state.memory if base_state else {}) or {}
        active_rules = (
            self.global_attributes.get(
                "active_rules", base_state.active_rules if base_state else {}
            )
            or {}
        )
        context = (
            self.global_attributes.get("context", base_state.context if base_state else {}) or {}
        )

        return WorldState(
            step=self.step,
            timestamp=self.timestamp,
            schema_version="2.0.0",
            entities=entities,
            relationships=relationships,
            resources=list(resources_dict.values()),
            memory=memory,
            active_rules=active_rules,
            context=context,
        )

    @property
    def node_types(self) -> list[str]:
        """List all categorical node types present in the graph."""
        return sorted(self.nodes_by_type.keys())

    @property
    def edge_types(self) -> list[str]:
        """List all canonical edge type keys: 'src_type__rel_type__dst_type'."""
        return sorted(self.edges_by_type.keys())

    def num_nodes(self, node_type: str | None = None) -> int:
        """Total count of nodes in the graph or within a specific node type."""
        if node_type is not None:
            return len(self.nodes_by_type.get(node_type, {}))
        return sum(len(nodes) for nodes in self.nodes_by_type.values())

    def num_edges(self, edge_type: str | None = None) -> int:
        """Total count of relational edges or within a specific canonical edge type."""
        if edge_type is not None:
            return len(self.edges_by_type.get(edge_type, []))
        return sum(len(edges) for edges in self.edges_by_type.values())

    def get_node_features(
        self,
        node_type: str,
        resource_keys: Sequence[str] | None = None,
        attribute_keys: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Extract a 2D float NumPy feature matrix for a specific node type: (N, D)."""
        nodes = self.nodes_by_type.get(node_type, {})
        if not nodes:
            return np.empty((0, 0), dtype=np.float32)

        sorted_nodes = [nodes[n_id] for n_id in sorted(nodes.keys())]

        # Determine feature keys
        if resource_keys is None:
            all_r_keys: set[str] = set()
            for n in sorted_nodes:
                all_r_keys.update(n.resources.keys())
            r_keys = sorted(all_r_keys)
        else:
            r_keys = list(resource_keys)

        attr_keys = list(attribute_keys or [])

        rows = []
        for n in sorted_nodes:
            r_vals = [float(n.resources.get(k, 0.0)) for k in r_keys]
            a_vals = [
                float(n.attributes[k]) if isinstance(n.attributes.get(k), (int, float)) else 0.0
                for k in attr_keys
            ]
            rows.append(r_vals + a_vals)

        return np.array(rows, dtype=np.float32)

    def get_edge_index(self, edge_type: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract 1D integer index arrays (source_indices, target_indices) for an edge type.

        Indices map to the sorted node order within source and destination node types.
        """
        edges = self.edges_by_type.get(edge_type, [])
        if not edges:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

        parts = edge_type.split("__")
        if len(parts) == 3:
            src_type, _rel_type, dst_type = parts
        else:
            src_type, dst_type = "unknown", "unknown"

        src_nodes = sorted(self.nodes_by_type.get(src_type, {}).keys())
        dst_nodes = sorted(self.nodes_by_type.get(dst_type, {}).keys())

        src_map = {n_id: idx for idx, n_id in enumerate(src_nodes)}
        dst_map = {n_id: idx for idx, n_id in enumerate(dst_nodes)}

        src_indices = []
        dst_indices = []
        for e in edges:
            if e.source in src_map and e.target in dst_map:
                src_indices.append(src_map[e.source])
                dst_indices.append(dst_map[e.target])

        return np.array(src_indices, dtype=np.int64), np.array(dst_indices, dtype=np.int64)

    def get_edge_features(
        self,
        edge_type: str,
        attribute_keys: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Extract a 2D float NumPy edge feature matrix (E, D) including weight and scalar attributes."""
        edges = self.edges_by_type.get(edge_type, [])
        if not edges:
            return np.empty((0, 0), dtype=np.float32)

        attr_keys = list(attribute_keys or [])
        rows = []
        for e in edges:
            vals = [float(e.weight)]
            for k in attr_keys:
                v = e.attributes.get(k, 0.0)
                vals.append(float(v) if isinstance(v, (int, float)) else 0.0)
            rows.append(vals)

        return np.array(rows, dtype=np.float32)

    def filter_temporal(self, current_time: float) -> HeterogeneousGraphView:
        """Create a new HeterogeneousGraphView containing only nodes and edges active at current_time."""
        active_nodes_by_type: dict[str, dict[EntityId, GraphNode]] = {}
        active_entity_ids: set[EntityId] = set()

        for n_type, nodes in self.nodes_by_type.items():
            active_nodes = {
                n_id: node for n_id, node in nodes.items() if node.is_active_at(current_time)
            }
            if active_nodes:
                active_nodes_by_type[n_type] = active_nodes
                active_entity_ids.update(active_nodes.keys())

        active_edges_by_type: dict[str, list[GraphEdge]] = {}
        for edge_key, edges in self.edges_by_type.items():
            active_edges = [
                e
                for e in edges
                if e.is_active_at(current_time)
                and e.source in active_entity_ids
                and e.target in active_entity_ids
            ]
            if active_edges:
                active_edges_by_type[edge_key] = active_edges

        return self.model_copy(
            update={
                "timestamp": current_time,
                "nodes_by_type": active_nodes_by_type,
                "edges_by_type": active_edges_by_type,
            }
        )
