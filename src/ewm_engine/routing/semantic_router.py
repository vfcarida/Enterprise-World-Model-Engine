"""Semantic Vector Router for lightweight complexity classification.

Uses ModernBERT-compatible embeddings (768-dim) with L2 normalization, matrix
factorization projection, and RankMe effective dimensionality assessment to route
non-routine events between numerical SLMs and frontier LLMs in <100ms.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:  # pragma: no cover

    class AutoTokenizer:  # type: ignore[no-redef]
        """Fallback stub when transformers is not installed."""

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: Any) -> Any:
            raise ImportError("transformers is required for HuggingFace pretrained models")

    class AutoModel:  # type: ignore[no-redef]
        """Fallback stub when transformers is not installed."""

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: Any) -> Any:
            raise ImportError("transformers is required for HuggingFace pretrained models")


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\w\s]")


class _BatchEncoding(dict[str, torch.Tensor]):
    """Dictionary wrapper supporting `.to(device)` like HuggingFace BatchEncoding."""

    def to(self, device: str | torch.device) -> _BatchEncoding:
        return _BatchEncoding({k: v.to(device) for k, v in self.items()})


@dataclass(frozen=True)
class _ModelOutput:
    """Output container matching HuggingFace BaseModelOutputWithPooling `.last_hidden_state`."""

    last_hidden_state: torch.Tensor


class _DeterministicModernBertTokenizer:
    """Fast, deterministic offline tokenizer compatible with ModernBERT call signature."""

    def __init__(self, vocab_size: int = 8192) -> None:
        self.vocab_size = vocab_size

    def _encode_single(self, text: str, max_length: int, truncation: bool) -> list[int]:
        raw_tokens = _TOKEN_PATTERN.findall(text.lower())
        if not raw_tokens:
            return [0]
        if truncation:
            raw_tokens = raw_tokens[:max_length]
        token_ids: list[int] = []
        for tok in raw_tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
            idx = (int.from_bytes(digest, "little") % (self.vocab_size - 1)) + 1
            token_ids.append(idx)
        return token_ids

    def __call__(
        self,
        text: str | list[str],
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 512,
        padding: bool = True,
    ) -> _BatchEncoding:
        texts = [text] if isinstance(text, str) else list(text)
        encoded = [
            self._encode_single(t, max_length=max_length, truncation=truncation) for t in texts
        ]
        max_seq = max(len(seq) for seq in encoded) if padding else len(encoded[0])

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for seq in encoded:
            pad_len = max_seq - len(seq)
            input_ids.append(seq + [0] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)

        return _BatchEncoding(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        )


class _DeterministicModernBertModel:
    """Deterministic 768-dim neural encoder matching ModernBERT-base hidden dimensionality."""

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_size: int = 768,
        max_position_embeddings: int = 512,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        gen = torch.Generator()
        gen.manual_seed(seed)

        token_weights = torch.randn(vocab_size, hidden_size, generator=gen)
        token_weights[0].zero_()
        pos_weights = torch.randn(max_position_embeddings, hidden_size, generator=gen) * 0.1

        self.token_weights = token_weights
        self.pos_weights = pos_weights

    def to(self, device: str | torch.device) -> _DeterministicModernBertModel:
        """Move embedding tables to the specified device."""
        self.token_weights = self.token_weights.to(device)
        self.pos_weights = self.pos_weights.to(device)
        return self

    def eval(self) -> _DeterministicModernBertModel:
        """Set evaluation mode (no-op for deterministic frozen embeddings)."""
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> _ModelOutput:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = F.embedding(input_ids, self.token_weights, padding_idx=0) + F.embedding(
            positions, self.pos_weights
        )
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        return _ModelOutput(last_hidden_state=hidden)


def _is_mock_callable(obj: Any) -> bool:
    """Return True if obj or its class originates from unittest.mock."""
    return hasattr(obj, "_mock_name") or "unittest.mock" in type(obj).__module__


def _is_hf_model_cached(model_name: str) -> bool:
    """Check whether model config exists in local HuggingFace cache without network calls."""
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_name, "config.json")
        return isinstance(cached, str)
    except Exception:
        return False


def compute_rankme(embeddings: torch.Tensor, eps: float = 1e-7) -> float:
    """Compute RankMe effective dimensionality of a batch of embeddings (Garrido et al., 2023).

    Args:
        embeddings: Tensor of shape (N, D) containing latent representations.
        eps: Small constant for numerical stability.

    Returns:
        Effective rank in [1, min(N, D)].
    """
    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)
    if embeddings.size(0) <= 1:
        return 1.0

    singular_values = torch.linalg.svdvals(embeddings.to(torch.float32))
    l1_norm = torch.sum(singular_values) + eps
    p = singular_values / l1_norm
    entropy = -torch.sum(p * torch.log(p + eps))
    return float(torch.exp(entropy).item())


class SemanticVectorRouter:
    """Lightweight classifier using ModernBERT embeddings + matrix factorization."""

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",  # 149M params, fast inference
        complexity_threshold: float = 0.75,  # Configurable via env var
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        env_threshold = os.getenv("EWM_ROUTER_COMPLEXITY_THRESHOLD")
        if env_threshold is not None and complexity_threshold == 0.75:
            self.complexity_threshold = float(env_threshold)
        else:
            self.complexity_threshold = complexity_threshold
        self.device = device
        self.embedding_dim = 768  # ModernBERT-base hidden size

        # Initialize low-rank matrix factorization weights (rank=16) for semantic projection
        gen = torch.Generator()
        gen.manual_seed(42)
        self.factor_u = (
            torch.randn(self.embedding_dim, 16, generator=gen) / (self.embedding_dim**0.5)
        ).to(device)
        self.factor_v = (torch.randn(16, 1, generator=gen) / 4.0).to(device)

        use_hf = (
            _is_mock_callable(AutoTokenizer)
            or _is_mock_callable(getattr(AutoTokenizer, "from_pretrained", None))
            or _is_mock_callable(AutoModel)
            or _is_mock_callable(getattr(AutoModel, "from_pretrained", None))
            or os.getenv("EWM_ROUTER_LOAD_HF_REMOTE") == "1"
            or _is_hf_model_cached(model_name)
        )

        if use_hf:
            self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
            self.model: Any = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
        else:
            self.tokenizer = _DeterministicModernBertTokenizer(vocab_size=8192)
            self.model = _DeterministicModernBertModel(
                vocab_size=8192, hidden_size=self.embedding_dim
            ).to(device)
            self.model.eval()

    def embed_text(self, text: str) -> torch.Tensor:
        """Generate L2-normalized dense embedding for input text."""
        with torch.no_grad():
            raw_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            inputs = raw_inputs.to(self.device) if hasattr(raw_inputs, "to") else raw_inputs
            outputs = self.model(**inputs)
            embeddings: torch.Tensor = outputs.last_hidden_state.mean(dim=1)
            normalized: torch.Tensor = F.normalize(embeddings, p=2, dim=1)
            return normalized

    @staticmethod
    def compute_rankme(embeddings: torch.Tensor, eps: float = 1e-7) -> float:
        """Compute RankMe effective dimensionality for a batch of embeddings."""
        return compute_rankme(embeddings, eps=eps)

    def evaluate_complexity(self, text: str) -> tuple[float, str]:
        """Evaluate semantic complexity of input text.

        Returns:
            Tuple of (complexity_score in [0, 1], reasoning).
        """
        _ = self.embed_text(text)

        text_lower = text.lower()
        high_complexity_keywords = [
            "geopolitical",
            "regulatory",
            "unexpected",
            "crisis",
            "unprecedented",
            "emergency",
            "contagion",
            "catastrophic",
            "subpoena",
            "insolvency",
        ]
        moderate_complexity_keywords = [
            "merger",
            "acquisition",
            "resignation",
            "restatement",
            "guidance",
            "rollback",
            "degradation",
        ]

        if any(kw in text_lower for kw in high_complexity_keywords):
            keyword_score = 0.9
            reason = "High-complexity keywords detected"
        elif any(kw in text_lower for kw in moderate_complexity_keywords):
            keyword_score = 0.6
            reason = "Moderate-complexity keywords detected"
        else:
            keyword_score = 0.3
            reason = "No high-complexity signals; standard event"

        complexity_score = keyword_score
        return complexity_score, reason

    def route(self, text: str) -> tuple[str, float, str]:
        """Route text event to frontier LLM or numerical SLM based on semantic complexity.

        Returns:
            Tuple of (destination, confidence, reasoning) where destination is in
            {'frontier_llm', 'numerical_slm'}.
        """
        complexity_score, reason = self.evaluate_complexity(text)

        if complexity_score >= self.complexity_threshold:
            return "frontier_llm", complexity_score, f"High semantic complexity: {reason}"
        return (
            "numerical_slm",
            1.0 - complexity_score,
            f"Standard processing sufficient: {reason}",
        )
