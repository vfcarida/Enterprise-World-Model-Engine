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
        description="Node type: 'intervention', 'event', 'action', 'state_change', 'violation', 'metric', 'exogenous_noise'."
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


class ExogenousNoiseRecord(BaseModel):
    """An addressable exogenous noise draw (U_i) in an SCM trajectory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(description="Unique addressable identifier of the exogenous draw.")
    step: int = Field(description="Discrete simulation step at which this exogenous noise was drawn.")
    mechanism: str = Field(description="Identifier or class name of the generating mechanism.")
    variable: str = Field(description="Name of the endogenous target or parameter influenced by this noise.")
    value: Any = Field(description="Exact realized numerical or structured draw value.")
    is_invertible: bool = Field(default=True, description="Whether this noise draw is algebraically invertible from outcomes.")
    seed_state: int | None = Field(default=None, description="Optional discrete integer state of the seed stream.")
    fingerprint: str = Field(default="", description="Canonical SHA-256 fingerprint of the draw.")


class TraceDiff(BaseModel):
    """Factual vs counterfactual divergence audit across systemic traces."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    invariant_nodes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Node IDs identical across factual and counterfactual traces.",
    )
    divergent_nodes: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Node IDs with divergent values or details: {node_id: {'factual': ..., 'counterfactual': ...}}.",
    )
    factual_only_nodes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Node IDs present only in the factual trace.",
    )
    counterfactual_only_nodes: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Node IDs present only in the counterfactual trace.",
    )
    coupled_noise_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Exogenous noise variable IDs coupled bit-for-bit across both branches.",
    )
    divergent_noise_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Exogenous noise variable IDs that diverged across branches.",
    )
    is_noise_coupled: bool = Field(
        default=True,
        description="True if all shared exogenous noise draws were bit-identical.",
    )

    def to_mermaid(self) -> str:
        """Export counterfactual trace divergence as a Mermaid flowchart with visual styling."""
        lines = ["flowchart TD"]
        for n_id in self.invariant_nodes:
            sanitized = n_id.replace('"', "'")
            lines.append(f'    {n_id}["{sanitized} (Invariant)"]:::invariant')
        for n_id in self.divergent_nodes:
            sanitized = n_id.replace('"', "'")
            lines.append(f'    {n_id}["{sanitized} (Diverged)"]:::divergent')
        for n_id in self.counterfactual_only_nodes:
            sanitized = n_id.replace('"', "'")
            lines.append(f'    {n_id}["{sanitized} (Counterfactual Only)"]:::cf_only')

        lines.append("    classDef invariant fill:#e2e8f0,stroke:#64748b,stroke-width:1px;")
        lines.append("    classDef divergent fill:#fee2e2,stroke:#ef4444,stroke-width:2px,stroke-dasharray: 5 5;")
        lines.append("    classDef cf_only fill:#dbeafe,stroke:#3b82f6,stroke-width:2px;")
        return "\n".join(lines)


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
    exogenous_noise: dict[str, ExogenousNoiseRecord] = Field(
        default_factory=dict,
        description="Addressable exogenous noise draws (U) indexed by noise id for exact SCM abduction and coupling.",
    )

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

    def to_html(self, title: str = "Systemic Trace Dependency Graph") -> str:
        """Export trace as a self-contained, offline-ready interactive HTML visualizer.

        Args:
            title: Title displayed in the visualizer header.

        Returns:
            Complete HTML5 document string with embedded SVG, dark theme, and node inspection.
        """
        from ewm_engine.provenance.html_visualizer import render_trace_html

        return render_trace_html(self, title=title)

    def record_exogenous_draw(
        self,
        noise_id: str,
        step: int,
        mechanism: str,
        variable: str,
        value: Any,
        is_invertible: bool = True,
        seed_state: int | None = None,
        add_to_nodes: bool = False,
    ) -> ExogenousNoiseRecord:
        """Record an addressable exogenous noise draw into this systemic trace."""
        from ewm_engine.core._canonical import canonical_sha256

        fp = canonical_sha256(
            {
                "id": noise_id,
                "step": step,
                "mechanism": mechanism,
                "variable": variable,
                "value": value,
            }
        )
        rec = ExogenousNoiseRecord(
            id=noise_id,
            step=step,
            mechanism=mechanism,
            variable=variable,
            value=value,
            is_invertible=is_invertible,
            seed_state=seed_state,
            fingerprint=fp,
        )
        self.exogenous_noise[noise_id] = rec
        if add_to_nodes and noise_id not in self.nodes:
            self.add_node(
                node_id=noise_id,
                step=step,
                category="exogenous_noise",
                label=f"Exogenous Noise: {noise_id}",
                evidence_level=EvidenceLevel.STRUCTURAL,
                details={
                    "value": value,
                    "mechanism": mechanism,
                    "variable": variable,
                    "fingerprint": fp,
                },
            )
        return rec

    @property
    def exogenous_noise_fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint over all sorted addressable exogenous noise draws."""
        from ewm_engine.core._canonical import canonical_sha256

        sorted_records = [
            {"id": k, "fingerprint": v.fingerprint, "step": v.step, "value": v.value}
            for k, v in sorted(self.exogenous_noise.items())
        ]
        return canonical_sha256(sorted_records)

    def diff(self, other: SystemicTrace) -> TraceDiff:
        """Compute divergence between this trace (factual) and another trace (counterfactual)."""
        from ewm_engine.core._canonical import normalize_canonical_value

        invariant_nodes: list[str] = []
        divergent_nodes: dict[str, dict[str, Any]] = {}
        factual_only_nodes: list[str] = []
        cf_only_nodes: list[str] = []

        all_node_ids = sorted(set(self.nodes.keys()) | set(other.nodes.keys()))
        for nid in all_node_ids:
            in_self = nid in self.nodes
            in_other = nid in other.nodes
            if in_self and in_other:
                d1 = normalize_canonical_value(self.nodes[nid].details)
                d2 = normalize_canonical_value(other.nodes[nid].details)
                if d1 == d2:
                    invariant_nodes.append(nid)
                else:
                    divergent_nodes[nid] = {
                        "factual": self.nodes[nid].details,
                        "counterfactual": other.nodes[nid].details,
                    }
            elif in_self:
                factual_only_nodes.append(nid)
            else:
                cf_only_nodes.append(nid)

        coupled_noise_ids: list[str] = []
        divergent_noise_ids: list[str] = []
        all_noise_ids = sorted(
            set(self.exogenous_noise.keys()) | set(other.exogenous_noise.keys())
        )
        for uid in all_noise_ids:
            if uid in self.exogenous_noise and uid in other.exogenous_noise:
                u1 = self.exogenous_noise[uid]
                u2 = other.exogenous_noise[uid]
                if (
                    u1.fingerprint == u2.fingerprint
                    and normalize_canonical_value(u1.value) == normalize_canonical_value(u2.value)
                ):
                    coupled_noise_ids.append(uid)
                else:
                    divergent_noise_ids.append(uid)
            else:
                divergent_noise_ids.append(uid)

        is_coupled = (
            len(divergent_noise_ids) == 0
            and len(coupled_noise_ids) == len(self.exogenous_noise)
            and len(self.exogenous_noise) > 0
        ) if self.exogenous_noise else (len(other.exogenous_noise) == 0)

        return TraceDiff(
            invariant_nodes=tuple(invariant_nodes),
            divergent_nodes=divergent_nodes,
            factual_only_nodes=tuple(factual_only_nodes),
            counterfactual_only_nodes=tuple(cf_only_nodes),
            coupled_noise_ids=tuple(coupled_noise_ids),
            divergent_noise_ids=tuple(divergent_noise_ids),
            is_noise_coupled=is_coupled,
        )
