"""Documented stub adapters for external verification backends (MoonLight, LLM soft-check).

Conforms to Track T3:
- MoonLight STREL: Spatio-temporal Reach and Escape Logic on topological graph worlds
  (isolated via JVM bridge).
- LLM soft-check: Uses the existing zero-dependency async-callable pattern
  `Callable[[str], Awaitable[str]]` without embedding any LLM SDK into core.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.provenance.trace import SystemicTrace


class MoonLightSTRELAdapter:
    """Documented adapter for MoonLight Spatio-Temporal Reach and Escape Logic (STREL).

    Evaluates spatio-temporal properties over topological graph worlds using the MoonLight
    JVM engine. Requires external JVM runtime and the `[strel]` optional extra.
    """

    def __init__(self, script_path: str | None = None) -> None:
        self.script_path = script_path
        raise SimulationConfigurationError(
            "MoonLightSTRELAdapter requires the '[strel]' extra and a valid JVM runtime. "
            "See docs/guides/verify-trajectories.md for setup instructions."
        )

    def evaluate(self, trace: SystemicTrace) -> dict[str, Any]:
        raise NotImplementedError("MoonLightSTRELAdapter requires [strel] extra.")


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
