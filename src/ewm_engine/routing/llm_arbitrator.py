"""Frontier LLM Arbitrator for structured causal intervention extraction.

Delegates high-criticality qualitative events to frontier LLMs under strict
JSON-mode prompt engineering and validates outputs into CausalInterventionVector.
"""

from __future__ import annotations

import json
from typing import Any

from ewm_engine.routing.schemas import CausalInterventionVector

litellm: Any = None
try:
    import litellm as _litellm_mod
    from litellm import completion as _litellm_completion

    litellm = _litellm_mod
except ImportError:  # pragma: no cover

    def _litellm_completion(*args: Any, **kwargs: Any) -> Any:
        raise ImportError("litellm is required for FrontierLLMArbitrator")


completion: Any = _litellm_completion


class FrontierLLMArbitrator:
    """Delegates to frontier LLM with strict JSON-mode prompt engineering."""

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract_causal_vector(
        self,
        text_content: str,
        domain: str,
    ) -> CausalInterventionVector:
        """Extract structured causal intervention vector via JSON-mode.

        Args:
            text_content: Unstructured qualitative event description.
            domain: Target domain ('financial' or 'aiops').

        Returns:
            Validated CausalInterventionVector instance.

        Raises:
            ValueError: If domain is unknown or LLM returns invalid JSON/schema violations.
        """
        system_prompt = self._build_system_prompt(domain)
        user_prompt = self._build_user_prompt(text_content, domain)

        response: Any = None
        try:
            active_completion = completion
            if (
                active_completion is _litellm_completion
                and litellm is not None
                and getattr(litellm, "completion", _litellm_completion) is not _litellm_completion
            ):
                active_completion = litellm.completion

            response = active_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )

            raw_json = response.choices[0].message.content
            if not isinstance(raw_json, str):
                raise TypeError(f"Expected string JSON content, got {type(raw_json).__name__}")

            parsed = json.loads(raw_json)
            if not isinstance(parsed, dict):
                raise ValueError("Top-level JSON output must be an object")

            vector = CausalInterventionVector(**parsed)
            return vector

        except Exception as e:
            raw_out = "N/A"
            if response is not None:
                try:
                    raw_out = str(response.choices[0].message.content)
                except Exception:
                    raw_out = "N/A"
            raise ValueError(f"LLM extraction failed: {e}\nRaw output: {raw_out}") from e

    def _build_system_prompt(self, domain: str) -> str:
        """Construct strict JSON-mode system prompt."""
        if domain == "financial":
            return """You are a financial event extraction engine. Your ONLY output is valid JSON.

Extract the following fields from the input text:
- primary_target_entities: List of ticker symbols (e.g., ["AAPL", "NVDA"])
- directional_sentiment: Float in [-1.0, 1.0] (bearish to bullish)
- magnitude: Float in [0.0, 1.0] (severity of shock)
- contagion_pathways: List of dicts, each with {"source": str, "target": str, "weight": float}
- confidence: Float in [0.0, 1.0]
- extraction_metadata: Dict with {"source_type": str, "timestamp": str, "reasoning": str}

Return ONLY valid JSON. No conversational text. No markdown code fences."""

        if domain == "aiops":
            return """You are an operational event extraction engine. Your ONLY output is valid JSON.

Extract the following fields from the input text:
- primary_target_entities: List of service/pod names (e.g., ["payment-service", "auth-api"])
- directional_sentiment: Float in [-1.0, 1.0] (degradation to improvement)
- magnitude: Float in [0.0, 1.0] (severity of impact)
- contagion_pathways: List of dicts, each with {"source": str, "target": str, "weight": float}
- confidence: Float in [0.0, 1.0]
- extraction_metadata: Dict with {"source_type": str, "timestamp": str, "reasoning": str}

Return ONLY valid JSON. No conversational text. No markdown code fences."""

        raise ValueError(f"Unknown domain: {domain}")

    def _build_user_prompt(self, text_content: str, domain: str) -> str:
        """Construct user prompt with input text."""
        return f"""Extract structured causal information from the following {domain} event:

<event_text>
{text_content}
</event_text>

Return JSON with the required fields. Be precise. Estimate confidence conservatively."""
