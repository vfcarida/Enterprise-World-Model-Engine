"""First-class pydantic-ai Agent toolset for EWM Engine (AGENTS extra).

Provides native integration between EWM Engine and pydantic-ai agents.
Since both libraries are built natively on Pydantic schemas, tool inputs
and outputs serialize with zero impedance mismatch.

Scientific Grounding:
    - Vafa et al. (NeurIPS 2024, arXiv:2406.03689): "Evaluating the World Model Implicit in a Generative Model."
      LLMs cannot serve as reliable internal world models; they must delegate simulation to a grounded kernel.
    - Kang et al. (ICML 2025, arXiv:2411.02385): "How Far is Video Generation from World Model."
    - Gkountouras et al. (arXiv:2410.19923): "Language Agents Meet Causality."
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

from ewm_engine.exceptions import SimulationConfigurationError
from ewm_engine.mcp.server import EwmMcpServer, ServerConfig


def is_pydantic_ai_available() -> bool:
    """Check if pydantic-ai is installed."""
    return importlib.util.find_spec("pydantic_ai") is not None


class EwmAgentToolset:
    """Collection of typed, validated EWM simulation tools for pydantic-ai and agent frameworks."""

    def __init__(self, provenance_strict: bool = False) -> None:
        self.server = EwmMcpServer(ServerConfig(provenance_strict=provenance_strict))

    def ewm_simulate(
        self,
        world_yaml: str,
        horizon: int = 5,
        samples: int = 1,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Simulate forward trajectories of an enterprise world under candidate assumptions."""
        return self.server.simulate(
            world_yaml=world_yaml,
            horizon=horizon,
            samples=samples,
            seed=seed,
        )

    def ewm_counterfactual(
        self,
        world_yaml: str,
        intervention: dict[str, Any],
        seed: int = 42,
        horizon: int = 5,
    ) -> dict[str, Any]:
        """Evaluate Pearl three-step counterfactual trajectory deltas under Twin-Rollout coupling."""
        return self.server.counterfactual(
            world_yaml=world_yaml,
            intervention=intervention,
            seed=seed,
            horizon=horizon,
        )

    def ewm_check_constraints(
        self,
        world_yaml: str,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate candidate action against operational constraints and physical invariants."""
        return self.server.check_constraints(
            world_yaml=world_yaml,
            action=action,
        )

    def ewm_decide(
        self,
        world_yaml: str,
        candidates: dict[str, Any],
        satisficing_threshold: float = 0.0,
        robustness_metric: str = "regret",
        horizon: int = 5,
        samples: int = 20,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Rank candidate policies and action interventions under deep parametric uncertainty."""
        return self.server.decide(
            world_yaml=world_yaml,
            candidates=candidates,
            satisficing_threshold=satisficing_threshold,
            robustness_metric=robustness_metric,
            horizon=horizon,
            samples=samples,
            seed=seed,
        )

    def ewm_explain(self, identifier: str) -> dict[str, Any]:
        """Retrieve the cryptographic provenance trace DAG for a simulation or decision."""
        return self.server.explain(identifier=identifier)

    def as_tools(self) -> list[Callable[..., Any]]:
        """Return callables list ready for pydantic-ai Agent(tools=[...])."""
        return [
            self.ewm_simulate,
            self.ewm_counterfactual,
            self.ewm_check_constraints,
            self.ewm_decide,
            self.ewm_explain,
        ]


def get_ewm_toolset(provenance_strict: bool = False) -> list[Callable[..., Any]]:
    """Return list of callable simulation tools for pydantic-ai or arbitrary agent frameworks."""
    toolset = EwmAgentToolset(provenance_strict=provenance_strict)
    return toolset.as_tools()


def create_ewm_agent(
    model: str = "test",
    provenance_strict: bool = False,
    system_prompt: str | None = None,
) -> Any:
    """Construct a pydantic-ai Agent wired to the EWM Engine simulation kernel."""
    if not is_pydantic_ai_available():
        raise SimulationConfigurationError(
            "pydantic-ai is required for create_ewm_agent. Install via `pip install ewm-engine[agents]`."
        )

    import pydantic_ai  # type: ignore[import-not-found]

    default_prompt = (
        "You are an enterprise decision-making agent grounded in the Enterprise World Model Engine.\n"
        "Rules:\n"
        "1. Never hallucinate physical quantities, resources, or business dynamics.\n"
        "2. Before recommending an action, validate its feasibility with ewm_check_constraints.\n"
        "3. Evaluate consequence distributions using ewm_simulate or ewm_counterfactual.\n"
        "4. Always cite canonical state fingerprints and evidence levels in your conclusions."
    )

    toolset = EwmAgentToolset(provenance_strict=provenance_strict)
    agent = pydantic_ai.Agent(
        model=model,
        system_prompt=system_prompt or default_prompt,
        tools=toolset.as_tools(),
    )
    return agent
