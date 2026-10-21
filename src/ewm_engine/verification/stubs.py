"""Documented stub adapters for external verification backends (MoonLight, LLM soft-check).

Conforms to Track T3:
- MoonLight STREL: Spatio-temporal Reach and Escape Logic on topological graph worlds
  (isolated via JVM bridge).
- LLM soft-check: Uses the existing zero-dependency async-callable pattern
  `Callable[[str], Awaitable[str]]` without embedding any LLM SDK into core.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.provenance.trace import SystemicTrace, TraceNode


class MoonLightSTRELAdapter:
    """Adapter for Spatio-Temporal Reach and Escape Logic (STREL).

    Evaluates spatio-temporal reach, escape, somewhere, and everywhere properties
    over topological causal graphs. When a `.strel` script path is provided, requires
    the `[strel]` extra and MoonLight JVM runtime; otherwise executes the built-in
    pure-Python topological STREL graph evaluator over `SystemicTrace`.
    """

    def __init__(
        self,
        script_path: str | None = None,
        max_hops: int = 3,
        node_predicate: Callable[[TraceNode], bool] | None = None,
    ) -> None:
        self.script_path = script_path
        self.max_hops = max_hops
        self.node_predicate = node_predicate or (lambda _node: True)
        if script_path is not None and importlib.util.find_spec("moonlight") is None:
            raise SimulationConfigurationError(
                "MoonLightSTRELAdapter requires the '[strel]' extra and a valid JVM runtime. "
                "See docs/guides/verify-trajectories.md for setup instructions."
            )

    def evaluate(
        self,
        trace: SystemicTrace,
        source_id: str | None = None,
        target_id: str | None = None,
        max_hops: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate STREL spatio-temporal properties over a SystemicTrace graph."""
        hop_limit = max_hops if max_hops is not None else self.max_hops
        nodes = trace.nodes
        if not nodes:
            return {
                "reach": False,
                "escape": False,
                "somewhere": False,
                "everywhere": True,
                "min_hops": None,
                "robustness": -1.0,
            }

        parents_map: dict[str, list[str]] = {nid: [] for nid in nodes}
        children_map: dict[str, list[TraceNode]] = {nid: [] for nid in nodes}
        for edge in trace.edges:
            if edge.source in nodes and edge.target in nodes:
                parents_map[edge.target].append(edge.source)
                children_map[edge.source].append(nodes[edge.target])

        roots = [nid for nid in nodes if len(parents_map[nid]) == 0] or [next(iter(nodes.keys()))]
        sources = [source_id] if source_id is not None and source_id in nodes else roots
        targets = (
            {target_id}
            if target_id is not None and target_id in nodes
            else {nid for nid in nodes if len(children_map[nid]) == 0}
        )

        # BFS from sources up to hop_limit
        queue: deque[tuple[str, int, bool]] = deque()
        visited_hops: dict[str, int] = {}
        min_reach_hops: int | None = None
        max_escape_hops = 0

        for src in sources:
            pred_ok = bool(self.node_predicate(nodes[src]))
            queue.append((src, 0, pred_ok))
            visited_hops[src] = 0

        while queue:
            curr_id, dist, path_pred_ok = queue.popleft()
            if path_pred_ok:
                max_escape_hops = max(max_escape_hops, dist)
                if curr_id in targets and dist <= hop_limit:
                    if min_reach_hops is None or dist < min_reach_hops:
                        min_reach_hops = dist

            if dist >= hop_limit:
                continue

            for child in children_map[curr_id]:
                next_dist = dist + 1
                next_ok = path_pred_ok and bool(self.node_predicate(child))
                if child.id not in visited_hops or next_dist < visited_hops[child.id]:
                    visited_hops[child.id] = next_dist
                    queue.append((child.id, next_dist, next_ok))

        reachable_nodes = [nodes[nid] for nid, d in visited_hops.items() if d <= hop_limit]
        pred_values = [bool(self.node_predicate(n)) for n in reachable_nodes]
        reach_sat = min_reach_hops is not None
        escape_sat = max_escape_hops >= hop_limit
        somewhere_sat = any(pred_values)
        everywhere_sat = all(pred_values) if pred_values else True
        robustness = float(hop_limit - min_reach_hops) if min_reach_hops is not None else -1.0

        return {
            "reach": reach_sat,
            "escape": escape_sat,
            "somewhere": somewhere_sat,
            "everywhere": everywhere_sat,
            "min_hops": min_reach_hops,
            "robustness": robustness,
        }


class LLMSoftCheckAdapter:
    """LLM soft-check verifier for natural language or free-form trace attributes.

    Uses an external callable `(prompt, **kwargs) -> str` adhering to the core
    zero-dependency policy (no OpenAI, Anthropic, or LangChain SDKs in core).
    """

    def __init__(
        self,
        llm_callable: Callable[[str], str] | Callable[[str], Awaitable[str]] | None = None,
        rubric: str = "Verify semantic policy adherence.",
    ) -> None:
        """Initialize LLM soft-check adapter.

        Args:
            llm_callable: Host-provided callable invoking an LLM without core SDK coupling.
            rubric: Natural language evaluation criteria.
        """
        self.llm_callable = llm_callable
        self.rubric = rubric

    def verify_node(self, node_label: str, details: dict[str, Any]) -> bool:
        """Evaluate whether a trace node meets the semantic rubric."""
        if self.llm_callable is None:
            raise SimulationConfigurationError(
                "LLMSoftCheckAdapter requires a host-provided llm_callable: (prompt) -> str."
            )
        prompt = f"Rubric: {self.rubric}\nNode: {node_label}\nDetails: {details}\nDoes this meet the rubric? Answer YES or NO."
        response = self.llm_callable(prompt)
        if isinstance(response, str):
            return "YES" in response.upper()
        # If async callable passed in synchronous context, indicate configuration guidance
        raise TypeError("Synchronous verify_node requires synchronous llm_callable.")

    async def verify_node_async(self, node_label: str, details: dict[str, Any]) -> bool:
        """Asynchronously evaluate whether a trace node meets the semantic rubric."""
        if self.llm_callable is None:
            raise SimulationConfigurationError(
                "LLMSoftCheckAdapter requires a host-provided llm_callable: (prompt) -> Awaitable[str] | str."
            )
        prompt = f"Rubric: {self.rubric}\nNode: {node_label}\nDetails: {details}\nDoes this meet the rubric? Answer YES or NO."
        res = self.llm_callable(prompt)
        if inspect.isawaitable(res):
            text = await res
        else:
            text = str(res)
        return "YES" in text.upper()
