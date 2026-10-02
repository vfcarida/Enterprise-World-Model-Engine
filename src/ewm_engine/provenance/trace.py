"""Systemic trace representations capturing dependency propagation in simulations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ewm_engine.provenance.evidence import EvidenceLevel


class TraceNode(BaseModel):
    """An event, state modification, intervention, or violation node in the systemic trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Unique node identifier in trace DAG.")
    step: int = Field(description="Simulation step at which this occurred.")
    category: str = Field(
        description="Node type: 'intervention', 'event', 'action', 'state_change', 'violation', 'metric'."
    )
    label: str = Field(description="Concise descriptive text.")
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.STRUCTURAL,
        description="Epistemic basis for this node's validity.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed quantitative attributes.",
    )


class TraceEdge(BaseModel):
    """A directed dependency or simulated influence connecting two trace nodes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(description="Origin node ID.")
    target: str = Field(description="Destination node ID.")
    step: int = Field(default=0, description="Simulation step at which this edge occurred.")
    relation: str = Field(default="influences", description="Nature of relationship.")
    evidence_level: EvidenceLevel = Field(
        default=EvidenceLevel.ASSUMED,
        description="Epistemic standing of this dependency link.",
    )

    @field_validator("relation")
    @classmethod
    def _reject_causes_relation(cls, v: str) -> str:
        if v.strip().lower() == "causes":
            raise ValueError(
                "TraceEdge relation 'causes' is forbidden. EWM Engine systemic traces capture "
                "mechanistic dependencies, not unverified causal claims. Use descriptive structural "
                "relations (e.g. 'influences', 'drives', 'perturbs', 'conditions', 'rejects', 'leads_to_violation')."
            )
        return v


class SystemicTrace(BaseModel):
    """Directed acyclic dependency graph tracking how interventions and shocks propagate.

    Explicitly models simulated mechanisms without overclaiming real-world causality.
    """

    model_config = ConfigDict(extra="forbid")

    nodes: dict[str, TraceNode] = Field(default_factory=dict)
    edges: list[TraceEdge] = Field(default_factory=list)

    def add_node(
        self,
        node_id: str,
        step: int,
        category: str,
        label: str,
        evidence_level: EvidenceLevel = EvidenceLevel.STRUCTURAL,
        details: dict[str, Any] | None = None,
    ) -> TraceNode:
        """Record a node in the systemic trace."""
        node = TraceNode(
            id=node_id,
            step=step,
            category=category,
            label=label,
            evidence_level=evidence_level,
            details=details or {},
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        step: int | None = None,
        relation: str = "influences",
        evidence_level: EvidenceLevel = EvidenceLevel.ASSUMED,
    ) -> TraceEdge:
        """Record a directed dependency edge in the systemic trace.

        Validates that both source and target reference existing nodes in the trace.
        """
        if source not in self.nodes:
            raise ValueError(
                f"Cannot add trace edge: source node '{source}' does not exist in systemic trace."
            )
        if target not in self.nodes:
            raise ValueError(
                f"Cannot add trace edge: target node '{target}' does not exist in systemic trace."
            )

        edge_step = step if step is not None else self.nodes[target].step
        edge = TraceEdge(
            source=source,
            target=target,
            step=edge_step,
            relation=relation,
            evidence_level=evidence_level,
        )
        self.edges.append(edge)
        return edge

    def to_mermaid(self) -> str:
        """Export trace as a GitHub/Markdown-compatible Mermaid flowchart."""
        lines = ["flowchart TD"]
        for node in self.nodes.values():
            # Clean label for mermaid syntax
            sanitized = node.label.replace('"', "'")
            lines.append(f'    {node.id}["{sanitized} ({node.evidence_level.value})"]')

        for edge in self.edges:
            lines.append(f'    {edge.source} -->|"{edge.relation}"| {edge.target}')

        return "\n".join(lines)

    def to_networkx(self) -> Any:
        """Export systemic trace to a NetworkX DiGraph if networkx is available."""
        try:
            import networkx as nx
        except ImportError as err:
            raise ImportError(
                "NetworkX is required to export trace as a graph object. "
                "Install it with `pip install networkx` or `pip install ewm-engine[graphs]`."
            ) from err

        graph = nx.DiGraph()
        for node in self.nodes.values():
            graph.add_node(
                node.id,
                step=node.step,
                category=node.category,
                label=node.label,
                evidence_level=node.evidence_level.value,
                **node.details,
            )
        for edge in self.edges:
            graph.add_edge(
                edge.source,
                edge.target,
                step=edge.step,
                relation=edge.relation,
                evidence_level=edge.evidence_level.value,
            )
        return graph
