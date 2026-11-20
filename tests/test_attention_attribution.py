"""Tests for supporting token-level attention attribution."""

from __future__ import annotations

import pytest

from src.explainability.attention_attribution import (
    ATTENTION_DISCLAIMER,
    compute_attention_attribution,
    compute_mock_attention_attribution,
)


def test_mandatory_disclaimer_content() -> None:
    """The attention attribution disclaimer must include the exact mandatory policy sentence."""
    assert "Attention weights are not guaranteed to be faithful legal explanations." in ATTENTION_DISCLAIMER


def test_missing_model_and_tokenizer_handled_gracefully() -> None:
    """Invoking attention attribution without a loaded model must return an explicit non-fatal record."""
    result = compute_attention_attribution(
        model=None,
        tokenizer=None,
        original_text="The party shall give notice.",
        simplified_text="The party may give notice.",
    )

    assert result["available"] is False
    assert "not provided" in result["reason"]
    assert result["disclaimer"] == ATTENTION_DISCLAIMER
    assert result["top_tokens"] == []


def test_mock_attention_attribution_deterministic() -> None:
    """Mock attention path must be deterministic and return well-formed weights and top tokens."""
    orig = "The seller shall deliver the goods within thirty days."
    simp = "The seller may deliver the goods within thirty days."

    res1 = compute_mock_attention_attribution(orig, simp, top_k=5, seed=42)
    res2 = compute_mock_attention_attribution(orig, simp, top_k=5, seed=42)

    assert res1["available"] is True
    assert res1["top_k"] == 5
    assert res1["disclaimer"] == ATTENTION_DISCLAIMER
    assert len(res1["tokens"]) > 0
    assert len(res1["top_tokens"]) == 5

    # Determinism check
    assert res1["weights"] == res2["weights"]
    assert res1["top_tokens"] == res2["top_tokens"]

    # Invariant: weights sum to ~1.0
    total_w = sum(res1["weights"])
    assert 0.99 <= total_w <= 1.01


def test_mock_attention_empty_input_fails_clearly() -> None:
    """Empty or non-string inputs to mock attention must raise ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        compute_mock_attention_attribution("", "Simplified text")

    with pytest.raises(ValueError, match="non-empty string"):
        compute_mock_attention_attribution("Original text", "   ")


def test_model_without_attentions_handled_safely() -> None:
    """A model returning no attentions must return available=False with disclaimer."""
    from types import SimpleNamespace
    import torch

    class DummyModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(attentions=None)

        def eval(self):
            pass

    class DummyTokenizer:
        def __call__(self, *args, **kwargs):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }

    res = compute_attention_attribution(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        original_text="The seller shall deliver.",
        simplified_text="The seller may deliver.",
    )

    assert res["available"] is False
    assert "without attentions" in res["reason"]
    assert res["disclaimer"] == ATTENTION_DISCLAIMER

