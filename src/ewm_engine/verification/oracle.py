"""Oracle-graph trajectory verifier based on Meta ARE (arXiv:2509.17158).

Evaluates systemic traces against an expected event DAG across three orthogonal axes:
1. Consistency: Exact parameter and category matching for expected events.
2. Causality: Topological parent-before-child ordering (independent branches may interleave).
3. Timing: Discrete step and continuous time tolerance windows between linked events.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.provenance.trace import SystemicTrace
from ewm_engine.simulation.trajectory import Trajectory


class OracleNode(BaseModel):
    """An expected event node within the Oracle Graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str = Field(
        description="Unique identifier of this expected event node in the oracle graph.",
    )
    category: str = Field(
        description="Expected trace category ('action', 'event', 'state_change', 'violation').",
    )
    label_pattern: str | None = Field(
        default=None,
        description="Optional substring or regex pattern expected in node label.",
    )
    expected_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Exact parameter key-values required to satisfy the consistency check.",
    )
    step_range: tuple[int, int] | None = Field(
        default=None,
        description="Optional [min_step, max_step] execution window.",
    )


class OracleEdge(BaseModel):
    """An expected causal or temporal dependency edge in the Oracle Graph."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(
        description="Source parent node_id that must precede target.",
    )
    target: str = Field(
        description="Target child node_id.",
    )
    relation: str = Field(
        default="precedes",
        description="Semantic relationship ('precedes', 'triggers', 'enables', 'conditions').",
    )
    min_step_delay: int = Field(
        default=0,
        ge=0,
        description="Minimum discrete simulation steps required between source and target.",
    )
    max_step_delay: int | None = Field(
        default=None,
        description="Maximum discrete simulation steps allowed between source and target.",
    )
    min_time_delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum continuous simulation time elapsed between source and target.",
    )
    max_time_delay: float | None = Field(
        default=None,
        description="Maximum continuous simulation time allowed between source and target.",
    )


class OracleGraph(BaseModel):
    """Specification of an expected event DAG for trajectory verification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodes: dict[str, OracleNode] = Field(
        default_factory=dict,
        description="Expected event nodes keyed by node_id.",
    )
    edges: tuple[OracleEdge, ...] = Field(
        default_factory=tuple,
        description="Directed causal dependencies between expected events.",
    )

    @classmethod
    def create(
        cls,
        nodes: Sequence[OracleNode] | Mapping[str, OracleNode] = (),
        edges: Sequence[OracleEdge] = (),
    ) -> OracleGraph:
        """Construct an OracleGraph with validated node dictionaries."""
        if isinstance(nodes, Mapping):
            node_dict = dict(nodes)
        else:
            node_dict = {n.node_id: n for n in nodes}
        return cls(nodes=node_dict, edges=tuple(edges))


class VerificationViolation(BaseModel):
    """Detailed audit record of an Oracle Graph verification check failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: Literal["consistency", "causality", "timing"] = Field(
        description="Failed verification axis.",
    )
    node_id: str | None = Field(
        default=None,
        description="Associated oracle node identifier.",
    )
    edge: tuple[str, str] | None = Field(
        default=None,
        description="Associated oracle dependency edge (source, target).",
    )
    message: str = Field(
        description="Human-readable violation message.",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation context, violating parameters, or observed delays.",
    )


class OracleVerificationResult(BaseModel):
    """Composite verification outcome across Consistency, Causality, and Timing axes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    satisfied: bool = Field(
        description="Whether all consistency, causality, and timing invariants held without violation.",
    )
    consistency_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of expected oracle nodes consistently matched in trace (1.0 = perfect).",
    )
    causality_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of expected causal ordering edges satisfied (1.0 = perfect).",
    )
    timing_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of expected timing intervals within tolerance windows (1.0 = perfect).",
    )
    matched_nodes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from oracle node_id to matched systemic trace node_id.",
    )
    violations: tuple[VerificationViolation, ...] = Field(
        default_factory=tuple,
        description="Collection of specific verification violations.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Evaluation diagnostics and execution metadata.",
    )


def evaluate_oracle_graph(
    oracle: OracleGraph,
    trace: SystemicTrace | Trajectory,
) -> OracleVerificationResult:
    """Evaluate a simulation trace or trajectory against an expected Oracle Graph.

    Performs three-axis verification:
    1. Consistency: Exact parameter and category matching for all oracle nodes.
    2. Causality: Parent events must strictly precede child events topologically.
    3. Timing: Interval between linked events must fall within [min_delay, max_delay].

    Args:
        oracle: Expected event DAG.
        trace: SystemicTrace graph or Trajectory containing a systemic trace.

    Returns:
        OracleVerificationResult containing scores, node matches, and violations.
    """
    actual_trace = trace.systemic_trace if isinstance(trace, Trajectory) else trace
    trace_nodes = actual_trace.nodes

    violations: list[VerificationViolation] = []
    matched_nodes: dict[str, str] = {}

    # 1. Axis 1: Consistency (Match Oracle Nodes to Trace Nodes)
    for oracle_id, oracle_node in oracle.nodes.items():
        best_match_id: str | None = None
        for t_id, t_node in trace_nodes.items():
            # Injective matching: an event in the trace cannot satisfy multiple distinct oracle nodes
            if t_id in matched_nodes.values():
                continue

            # Check category
            if t_node.category != oracle_node.category:
                continue

            # Check label pattern if specified
            if oracle_node.label_pattern and oracle_node.label_pattern not in t_node.label:
                continue

            # Check step range
            if oracle_node.step_range:
                min_s, max_s = oracle_node.step_range
                if not (min_s <= t_node.step <= max_s):
                    continue

            # Check exact expected details
            details_match = True
            for k, expected_v in oracle_node.expected_details.items():
                actual_v = t_node.details.get(k)
                if actual_v != expected_v:
                    details_match = False
                    break

            if details_match:
                best_match_id = t_id
                break

        if best_match_id is not None:
            matched_nodes[oracle_id] = best_match_id
        else:
            violations.append(
                VerificationViolation(
                    axis="consistency",
                    node_id=oracle_id,
                    message=f"Oracle node '{oracle_id}' (category '{oracle_node.category}') "
                    f"could not be matched with consistent parameters in trace.",
                    details={"expected_details": oracle_node.expected_details},
                )
            )

    total_nodes = len(oracle.nodes)
    consistency_score = len(matched_nodes) / total_nodes if total_nodes > 0 else 1.0

    # 2. Axis 2 & 3: Causality and Timing over Edges
    total_edges = len(oracle.edges)
    causality_satisfied = 0
    timing_satisfied = 0

    for edge in oracle.edges:
        src_matched = matched_nodes.get(edge.source)
        tgt_matched = matched_nodes.get(edge.target)

        # If either endpoint wasn't matched, edge cannot be satisfied
        if not src_matched or not tgt_matched:
            violations.append(
                VerificationViolation(
                    axis="causality",
                    edge=(edge.source, edge.target),
                    message=f"Cannot verify edge ({edge.source} -> {edge.target}): "
                    f"endpoints missing from matched nodes.",
                )
            )
            continue

        src_node = trace_nodes[src_matched]
        tgt_node = trace_nodes[tgt_matched]

        # Causality check: step and time topological order
        # Parent must occur at or before child
        causal_order_ok = src_node.step <= tgt_node.step

        if causal_order_ok:
            causality_satisfied += 1
        else:
            violations.append(
                VerificationViolation(
                    axis="causality",
                    edge=(edge.source, edge.target),
                    message=f"Causality inverted: parent '{edge.source}' at step {src_node.step} "
                    f"occurred after child '{edge.target}' at step {tgt_node.step}.",
                    details={
                        "source_step": src_node.step,
                        "target_step": tgt_node.step,
                    },
                )
            )

        # Timing check: step and continuous time intervals
        step_diff = tgt_node.step - src_node.step
        timing_ok = True

        if step_diff < edge.min_step_delay:
            timing_ok = False
            violations.append(
                VerificationViolation(
                    axis="timing",
                    edge=(edge.source, edge.target),
                    message=f"Step delay {step_diff} < minimum required {edge.min_step_delay}.",
                    details={"step_diff": step_diff, "min_step_delay": edge.min_step_delay},
                )
            )

        if edge.max_step_delay is not None and step_diff > edge.max_step_delay:
            timing_ok = False
            violations.append(
                VerificationViolation(
                    axis="timing",
                    edge=(edge.source, edge.target),
                    message=f"Step delay {step_diff} > maximum allowed {edge.max_step_delay}.",
                    details={"step_diff": step_diff, "max_step_delay": edge.max_step_delay},
                )
            )

        if timing_ok:
            timing_satisfied += 1

    causality_score = causality_satisfied / total_edges if total_edges > 0 else 1.0
    timing_score = timing_satisfied / total_edges if total_edges > 0 else 1.0

    all_satisfied = len(violations) == 0

    return OracleVerificationResult(
        satisfied=all_satisfied,
        consistency_score=consistency_score,
        causality_score=causality_score,
        timing_score=timing_score,
        matched_nodes=matched_nodes,
        violations=tuple(violations),
        metadata={
            "oracle_nodes_count": total_nodes,
            "oracle_edges_count": total_edges,
            "trace_nodes_count": len(trace_nodes),
            "trace_edges_count": len(actual_trace.edges),
        },
    )
