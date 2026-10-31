"""In-memory continuous graph registry for high-frequency topology mutations.

Provides O(1) node and edge additions/removals and fast vectorized snapshot
generation (<10ms for 1,000-node graphs) for EvolveGCN temporal training.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

_LAST_UTC_MS: int = -1
_LAST_UTC_DT: datetime = datetime(2026, 1, 1)


def _utcnow_naive() -> datetime:
    """Return current UTC timestamp as naive datetime."""
    global _LAST_UTC_MS, _LAST_UTC_DT
    now_ms = int(time.time() * 1000)
    if now_ms != _LAST_UTC_MS:
        _LAST_UTC_MS = now_ms
        _LAST_UTC_DT = datetime.now(UTC).replace(tzinfo=None)
    return _LAST_UTC_DT


@dataclass(slots=True)
class Node:
    """Represents a graph node (financial entity or microservice)."""

    node_id: str
    node_type: str  # 'equity', 'sector', 'service', 'pod'
    features: dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=_utcnow_naive)


@dataclass(slots=True)
class Edge:
    """Represents a directed edge (supply-chain link, API call)."""

    source_id: str
    target_id: str
    edge_type: str  # 'supply_chain', 'sector_correlation', 'api_call', 'dependency'
    weight: float = 1.0
    last_updated: datetime = field(default_factory=_utcnow_naive)


class ContinuousGraphRegistry:
    """In-memory registry for dynamic graph topology with O(1) edge updates."""

    def __init__(self, max_snapshot_history: int = 1000) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str], Edge] = {}  # Keyed by (source_id, target_id)
        self.adjacency_list: dict[str, set[str]] = {}  # Outgoing neighbors
        self._reverse_adjacency_list: dict[str, set[str]] = {}  # Incoming neighbors

        # Temporal snapshot history: (timestamp, node_features, adjacency_matrix)
        self.snapshot_history: list[tuple[datetime, np.ndarray, np.ndarray]] = []
        self.max_snapshot_history: int = max_snapshot_history

    def add_node(self, node: Node) -> None:
        """Add or update a node. O(1) operation."""
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency_list:
            self.adjacency_list[node.node_id] = set()
        if node.node_id not in self._reverse_adjacency_list:
            self._reverse_adjacency_list[node.node_id] = set()

    def add_edge(self, edge: Edge) -> None:
        """Add or update an edge. O(1) operation."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError(
                f"Cannot add edge: nodes {edge.source_id} or {edge.target_id} not found"
            )

        self.edges[(edge.source_id, edge.target_id)] = edge
        self.adjacency_list[edge.source_id].add(edge.target_id)
        self._reverse_adjacency_list[edge.target_id].add(edge.source_id)
        edge.last_updated = _utcnow_naive()

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove an edge. O(1) operation."""
        key = (source_id, target_id)
        if key in self.edges:
            del self.edges[key]
            if source_id in self.adjacency_list:
                self.adjacency_list[source_id].discard(target_id)
            if target_id in self._reverse_adjacency_list:
                self._reverse_adjacency_list[target_id].discard(source_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its incident edges. O(degree) operation."""
        if node_id not in self.nodes:
            return

        # Remove all outgoing edges in O(out_degree)
        for target_id in list(self.adjacency_list.get(node_id, ())):
            self.remove_edge(node_id, target_id)

        # Remove all incoming edges in O(in_degree)
        for source_id in list(self._reverse_adjacency_list.get(node_id, ())):
            self.remove_edge(source_id, node_id)

        del self.nodes[node_id]
        self.adjacency_list.pop(node_id, None)
        self._reverse_adjacency_list.pop(node_id, None)

    def take_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate snapshot for EvolveGCN training.

        Returns:
            Tuple of:
                node_features: (N, F) float32 array where N=num_nodes, F=num_features
                adjacency_matrix: (N, N) float32 array containing edge weights
        """
        node_ids = sorted(self.nodes.keys())
        num_nodes = len(node_ids)
        node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        feature_keys = sorted(self.nodes[node_ids[0]].features.keys()) if num_nodes > 0 else []
        num_features = len(feature_keys)

        if num_features > 0 and num_nodes > 0:
            node_features = np.array(
                [[self.nodes[nid].features.get(k, 0.0) for k in feature_keys] for nid in node_ids],
                dtype=np.float32,
            )
        else:
            node_features = np.zeros((num_nodes, num_features), dtype=np.float32)

        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        if self.edges:
            src_indices = [node_id_to_idx[s] for (s, _) in self.edges]
            tgt_indices = [node_id_to_idx[t] for (_, t) in self.edges]
            weights = [e.weight for e in self.edges.values()]
            adjacency_matrix[src_indices, tgt_indices] = np.asarray(weights, dtype=np.float32)

        timestamp = _utcnow_naive()
        self.snapshot_history.append((timestamp, node_features, adjacency_matrix))
        if len(self.snapshot_history) > self.max_snapshot_history:
            self.snapshot_history.pop(0)

        return node_features, adjacency_matrix
