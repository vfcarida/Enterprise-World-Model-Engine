"""Unit and benchmark tests for the Semantic Vector Router."""
# ruff: noqa: E402

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from ewm_engine.routing.semantic_router import SemanticVectorRouter, compute_rankme


@pytest.fixture(scope="module")
def router() -> SemanticVectorRouter:
    """Module-scoped SemanticVectorRouter fixture."""
    return SemanticVectorRouter()


@pytest.mark.unit
def test_semantic_router_routes_unexpected_regulatory_action_to_frontier_llm(
    router: SemanticVectorRouter,
) -> None:
    """Text: 'Unexpected regulatory action against tech conglomerate' -> route to frontier_llm."""
    text = "Unexpected regulatory action against tech conglomerate"
    destination, confidence, reasoning = router.route(text)
    assert destination == "frontier_llm"
    assert confidence >= 0.75
    assert "High semantic complexity" in reasoning


@pytest.mark.unit
def test_semantic_router_routes_standard_earnings_to_numerical_slm(
    router: SemanticVectorRouter,
) -> None:
    """Text: 'Standard Q3 earnings beat consensus by 2%' -> route to numerical_slm."""
    text = "Standard Q3 earnings beat consensus by 2%"
    destination, confidence, reasoning = router.route(text)
    assert destination == "numerical_slm"
    assert confidence == pytest.approx(0.7)
    assert "Standard processing sufficient" in reasoning


@pytest.mark.unit
def test_semantic_router_moderate_complexity_and_env_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moderate complexity events respect custom and environment-configured thresholds."""
    text = "Corporate merger and executive resignation announced in Q4 filing"

    default_router = SemanticVectorRouter()
    dest_default, conf_default, reason_default = default_router.route(text)
    assert dest_default == "numerical_slm"
    assert conf_default == pytest.approx(0.4)
    assert "Moderate-complexity keywords detected" in reason_default

    # Configure lower threshold via environment variable
    monkeypatch.setenv("EWM_ROUTER_COMPLEXITY_THRESHOLD", "0.55")
    env_router = SemanticVectorRouter()
    assert env_router.complexity_threshold == pytest.approx(0.55)
    dest_env, conf_env, _ = env_router.route(text)
    assert dest_env == "frontier_llm"
    assert conf_env == pytest.approx(0.6)


@pytest.mark.unit
def test_semantic_router_embedding_determinism_and_rankme(
    router: SemanticVectorRouter,
) -> None:
    """Verify embeddings are deterministic, L2-normalized, 768-dim, and high RankMe."""
    e1 = router.embed_text("Semiconductor supply chain disruption in Taiwan")
    e2 = router.embed_text("Semiconductor supply chain disruption in Taiwan")
    assert e1.shape == (1, 768)
    assert torch.allclose(e1, e2, atol=1e-6)
    l2_norm = float(torch.linalg.vector_norm(e1, ord=2).item())
    assert l2_norm == pytest.approx(1.0, abs=1e-5)

    # Empty/whitespace text edge case
    empty_emb = router.embed_text("   ")
    assert empty_emb.shape == (1, 768)

    # RankMe across 40 diverse texts must show high effective dimensionality (no collapse)
    diverse_texts = [
        f"Event {i}: {topic} impacting {entity} at timestamp {i * 17}"
        for i, (topic, entity) in enumerate(
            [
                ("liquidity crunch", "regional banks"),
                ("memory leak", "payment-service"),
                ("tariff escalation", "automotive suppliers"),
                ("bgp route leak", "ingress-gateway"),
                ("sovereign downgrade", "treasury desk"),
                ("thread pool exhaustion", "auth-api"),
                ("litigation settlement", "pharma corp"),
                ("disk latency spike", "kafka-broker"),
            ]
            * 5
        )
    ]
    batch_embeddings = torch.cat([router.embed_text(t) for t in diverse_texts], dim=0)
    rankme_score = router.compute_rankme(batch_embeddings)
    assert rankme_score > 10.0

    # Collapsed rank-1 embeddings have RankMe == 1.0
    collapsed = e1.repeat(10, 1)
    assert compute_rankme(collapsed) == pytest.approx(1.0, abs=1e-4)
    assert compute_rankme(e1.squeeze(0)) == pytest.approx(1.0)


@pytest.mark.unit
def test_semantic_router_with_mocked_huggingface_pretrained() -> None:
    """Verify SemanticVectorRouter delegates to AutoTokenizer/AutoModel when mocked."""

    @dataclass
    class _DummyOut:
        last_hidden_state: torch.Tensor

    mock_tok_instance = MagicMock()
    mock_tok_instance.return_value = {
        "input_ids": torch.ones((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }
    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.return_value = _DummyOut(
        last_hidden_state=torch.ones((1, 4, 768), dtype=torch.float32)
    )

    with (
        patch(
            "ewm_engine.routing.semantic_router.AutoTokenizer.from_pretrained",
            return_value=mock_tok_instance,
        ) as mock_tok_fp,
        patch(
            "ewm_engine.routing.semantic_router.AutoModel.from_pretrained",
            return_value=mock_model_instance,
        ) as mock_mod_fp,
    ):
        hf_router = SemanticVectorRouter(model_name="answerdotai/ModernBERT-base")
        emb = hf_router.embed_text("Unexpected crisis in credit markets")
        assert emb.shape == (1, 768)
        mock_tok_fp.assert_called_once_with("answerdotai/ModernBERT-base")
        mock_mod_fp.assert_called_once_with("answerdotai/ModernBERT-base")
        mock_model_instance.eval.assert_called_once()


@pytest.mark.benchmark
def test_semantic_router_100_texts_under_10_seconds(
    router: SemanticVectorRouter, benchmark: Any
) -> None:
    """Performance: Embed + classify 100 texts in <10 seconds."""
    sample_texts = [
        f"Financial and operational event #{i}: unexpected regulatory audit or standard Q3 guidance"
        for i in range(100)
    ]

    def _classify_100() -> int:
        routed = 0
        for text in sample_texts:
            dest, _, _ = router.route(text)
            if dest in ("frontier_llm", "numerical_slm"):
                routed += 1
        return routed

    count = benchmark(_classify_100)
    assert count == 100

    # Verify single 100-text batch takes <10s (and per-event <100ms)
    single_start = time.perf_counter()
    _classify_100()
    single_batch_s = time.perf_counter() - single_start
    assert single_batch_s < 10.0, f"100 texts took {single_batch_s:.2f}s (>= 10s)"
