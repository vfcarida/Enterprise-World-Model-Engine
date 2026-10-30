"""Causal intervention context managers and graph surgery (`do(X_t = x)`) primitives.

Scientific Grounding:
    - Peters, J., Janzing, D., & Schölkopf, B. (2017): "Elements of Causal Inference:
      Foundations and Learning Algorithms", MIT Press.
    - Pearl, J. (2009): "Causality: Models, Reasoning, and Inference", Cambridge University Press.
      For structural assignment $Y_{t+1} = f(X_t, U_t)$, applying the interventional operator
      $\\text{do}(X_t = x)$ replaces the structural mechanism for $X_t$ with the constant $x$
      and severs all inbound directed edges $(u \\to X_t)$ in the causal graph $\\mathcal{G}$,
      explicitly disabling observational confounding mechanisms into $X_t$.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from ewm_engine.core.actions import Action, Intervention, get_active_interventions
from ewm_engine.core.scm import CausalGraph

if TYPE_CHECKING:
    from ewm_engine.core.state import WorldState


class CausalExecutionLogEntry(BaseModel):
    """Immutable audit log entry recording causal graph surgery (`do()`) and step execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step: int = Field(default=0, ge=0, description="Discrete simulation step index.")
    operation: Literal["do_surgery", "step_execution", "context_enter", "context_exit"] = Field(
        description="Causal operation recorded by the router or engine."
    )
    intervened_variables: tuple[str, ...] = Field(
        default_factory=tuple,
        description="State variables subjected to do(X_t = x) intervention.",
    )
    forced_values: dict[str, Any] = Field(
        default_factory=dict,
        description="Forced constant values assigned to intervened variables.",
    )
    edges_before: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Directed edges in the causal graph prior to surgery.",
    )
    severed_inbound_edges: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Inbound edges (u -> X_t) severed by do(X_t = x) surgery.",
    )
    edges_after: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Directed edges remaining in the mutilated causal graph.",
    )
    confounding_disabled: bool = Field(
        default=False,
        description="True when observational confounding into intervened nodes is disabled.",
    )
    notes: str = Field(
        default="",
        description="Human-readable diagnostic description of the causal operation.",
    )


def normalize_node_name(node: str, graph: CausalGraph | None = None) -> str:
    """Resolve a variable identifier against a CausalGraph's node naming convention."""
    clean = node.split(":", 1)[1] if node.startswith("resource:") else node
    if graph is not None:
        if node in graph.nodes:
            return node
        if clean in graph.nodes:
            return clean
        prefixed = f"resource:{clean}"
        if prefixed in graph.nodes:
            return prefixed
    return clean


def parse_intervention_targets(
    spec: Action | Intervention | Mapping[str, Any],
) -> dict[str, Any]:
    """Extract target variable names and forced values from an Action, Intervention, or dict."""
    if isinstance(spec, Intervention):
        return spec.forced_nodes()
    if isinstance(spec, Action):
        params = spec.parameters
        if "target" in params and "value" in params:
            extracted: dict[str, Any] = {str(params["target"]): copy.deepcopy(params["value"])}
            extra_targets = params.get("targets")
            if isinstance(extra_targets, Mapping):
                for k, v in extra_targets.items():
                    extracted[str(k)] = copy.deepcopy(v)
            return extracted
        if "targets" in params and isinstance(params["targets"], Mapping):
            return {
                str(k): copy.deepcopy(v)
                for k, v in params["targets"].items()
                if not str(k).startswith("_")
            }
        return {
            str(k): copy.deepcopy(v)
            for k, v in params.items()
            if not str(k).startswith("_")
        }
    if isinstance(spec, Mapping):
        return {
            str(k): copy.deepcopy(v)
            for k, v in spec.items()
            if not str(k).startswith("_")
        }
    raise TypeError(
        f"Expected Action, Intervention, or Mapping for do() intervention, got {type(spec).__name__}."
    )


def sever_inbound_edges(
    graph: CausalGraph,
    targets: Sequence[str],
    *,
    step: int = 0,
    forced_values: Mapping[str, Any] | None = None,
    operation: Literal["do_surgery", "step_execution", "context_enter", "context_exit"] = "do_surgery",
    notes: str = "",
) -> tuple[CausalGraph, CausalExecutionLogEntry]:
    """Perform Pearl's causal surgery $\\text{do}(X_t = x)$ by severing all inbound edges to `targets`.

    For $Y_{t+1} = f(X_t, U_t)$, severing all $(u \\to X_t)$ edges removes parent and confounding
    influences on $X_t$ while preserving outbound causal edges $(X_t \\to Y_{t+1})$.
    """
    mutilated = CausalGraph(
        nodes=set(graph.nodes),
        edges=set(graph.edges),
        unobserved_nodes=set(graph.unobserved_nodes),
    )
    edges_before = tuple(sorted(graph.edges))
    resolved_targets: list[str] = []
    candidate_names: set[str] = set()

    for raw_t in targets:
        norm_t = normalize_node_name(raw_t, mutilated)
        mutilated.add_node(norm_t)
        resolved_targets.append(norm_t)
        clean_t = raw_t.split(":", 1)[1] if raw_t.startswith("resource:") else raw_t
        candidate_names.add(raw_t)
        candidate_names.add(norm_t)
        candidate_names.add(clean_t)
        candidate_names.add(f"resource:{clean_t}")

    severed: list[tuple[str, str]] = []
    for u, v in list(mutilated.edges):
        if v in candidate_names:
            mutilated.edges.remove((u, v))
            severed.append((u, v))

    edges_after = tuple(sorted(mutilated.edges))
    forced_dict = (
        {normalize_node_name(str(k), mutilated): copy.deepcopy(v) for k, v in forced_values.items()}
        if forced_values is not None
        else {}
    )
    default_notes = (
        f"Applied do({', '.join(f'{k}={v!r}' for k, v in forced_dict.items()) or ', '.join(resolved_targets)}); "
        f"severed {len(severed)} inbound edge(s)."
    )
    log_entry = CausalExecutionLogEntry(
        step=step,
        operation=operation,
        intervened_variables=tuple(sorted(set(resolved_targets))),
        forced_values=forced_dict,
        edges_before=edges_before,
        severed_inbound_edges=tuple(sorted(severed)),
        edges_after=edges_after,
        confounding_disabled=len(resolved_targets) > 0,
        notes=notes or default_notes,
    )
    return mutilated, log_entry


class InterventionContext:
    """Context manager forcing specific state nodes and overriding standard dynamics.

    Can be used standalone (`with InterventionContext({"X": 10.0}):`) or via
    `with engine.intervene({"X": 10.0}):` or `with Intervention(...):`.
    """

    def __init__(
        self,
        intervention: Intervention | Action | Mapping[str, Any],
        *,
        engine: Any | None = None,
        intervention_id: str = "ctx_intervention",
        description: str = "Context-managed causal node override",
    ) -> None:
        if isinstance(intervention, Intervention):
            self.intervention = intervention
        else:
            targets = parse_intervention_targets(intervention)
            self.intervention = Intervention(
                id=intervention_id,
                description=description,
                target_type="node_override",
                parameters=targets,
            )
        self._engine = engine
        self._saved_graph: CausalGraph | None = None
        self._saved_do_map: dict[str, Any] | None = None
        self.last_log_entry: CausalExecutionLogEntry | None = None

    @property
    def forced_nodes(self) -> dict[str, Any]:
        """Return the forced node mapping active within this context."""
        return self.intervention.forced_nodes()

    def apply_to_state(self, state: WorldState) -> WorldState:
        """Force the context's target nodes directly on `state`."""
        return self.intervention.apply_forced_nodes(state)

    def __enter__(self) -> Self:
        self.intervention.__enter__()
        if self._engine is not None:
            graph = getattr(self._engine, "causal_graph", None)
            if isinstance(graph, CausalGraph):
                self._saved_graph = CausalGraph(
                    nodes=set(graph.nodes),
                    edges=set(graph.edges),
                    unobserved_nodes=set(graph.unobserved_nodes),
                )
                forced = self.intervention.forced_nodes()
                mutilated, entry = sever_inbound_edges(
                    graph,
                    list(forced.keys()),
                    step=getattr(self._engine, "current_step", 0),
                    forced_values=forced,
                    operation="context_enter",
                    notes=f"Entered InterventionContext '{self.intervention.id}'.",
                )
                self._engine.causal_graph = mutilated
                self.last_log_entry = entry
                log_list = getattr(self._engine, "_causal_execution_log", None)
                if isinstance(log_list, list):
                    log_list.append(entry)
            do_map = getattr(self._engine, "_active_do_interventions", None)
            if isinstance(do_map, dict):
                self._saved_do_map = dict(do_map)
                for k, v in self.intervention.forced_nodes().items():
                    norm_k = normalize_node_name(k, getattr(self._engine, "causal_graph", None))
                    do_map[norm_k] = copy.deepcopy(v)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        try:
            if self._engine is not None:
                if self._saved_graph is not None:
                    current_graph = getattr(self._engine, "causal_graph", self._saved_graph)
                    edges_before = (
                        tuple(sorted(current_graph.edges))
                        if isinstance(current_graph, CausalGraph)
                        else ()
                    )
                    self._engine.causal_graph = self._saved_graph
                    exit_entry = CausalExecutionLogEntry(
                        step=getattr(self._engine, "current_step", 0),
                        operation="context_exit",
                        intervened_variables=tuple(sorted(self.intervention.forced_nodes().keys())),
                        forced_values=self.intervention.forced_nodes(),
                        edges_before=edges_before,
                        severed_inbound_edges=(),
                        edges_after=tuple(sorted(self._saved_graph.edges)),
                        confounding_disabled=False,
                        notes=f"Exited InterventionContext '{self.intervention.id}' and restored graph.",
                    )
                    log_list = getattr(self._engine, "_causal_execution_log", None)
                    if isinstance(log_list, list):
                        log_list.append(exit_entry)
                if self._saved_do_map is not None:
                    self._engine._active_do_interventions = self._saved_do_map
        finally:
            self.intervention.__exit__(exc_type, exc_val, exc_tb)


__all__ = [
    "CausalExecutionLogEntry",
    "InterventionContext",
    "get_active_interventions",
    "normalize_node_name",
    "parse_intervention_targets",
    "sever_inbound_edges",
]
