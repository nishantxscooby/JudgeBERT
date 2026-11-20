"""Supporting token-level attention attribution for cross-encoder models.

Treats attention weights strictly as auxiliary supporting evidence.

Mandatory Non-Legal Attribution Policy:
    Attention weights are not guaranteed to be faithful legal explanations.
    They reflect internal representation geometry rather than causal statutory reasoning,
    and are never presented as a legal justification.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ATTENTION_DISCLAIMER = (
    "Attention weights are not guaranteed to be faithful legal explanations. "
    "They are reported as weak supporting evidence only, and are never presented as a legal justification."
)


def compute_mock_attention_attribution(
    original_text: str,
    simplified_text: str,
    top_k: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Deterministic offline attention attribution for testing and non-GPU environments.

    Constructs simulated [CLS] cross-encoder attention distributions over
    tokenized original and simplified segments without requiring Hugging Face downloads.
    """
    if not isinstance(original_text, str) or not original_text.strip():
        raise ValueError("original_text must be a non-empty string.")
    if not isinstance(simplified_text, str) or not simplified_text.strip():
        raise ValueError("simplified_text must be a non-empty string.")

    rng = np.random.RandomState(seed)

    # Simple word tokenisation
    orig_words = re.findall(r"\w+|[^\w\s]", original_text)
    simp_words = re.findall(r"\w+|[^\w\s]", simplified_text)

    tokens = ["[CLS]"] + orig_words + ["[SEP]"] + simp_words + ["[SEP]"]
    segments = (
        ["special"]
        + ["original"] * len(orig_words)
        + ["special"]
        + ["simplified"] * len(simp_words)
        + ["special"]
    )

    # Generate deterministic pseudo-attention weights
    raw_weights = rng.exponential(scale=1.0, size=len(tokens))
    # Give slightly higher weight to capitalized or key legal words
    for idx, tok in enumerate(tokens):
        if tok.lower() in ("shall", "must", "may", "not", "करार", "संविदा", "शून्य", "thirty", "sixty"):
            raw_weights[idx] += 3.0
    weights = raw_weights / np.sum(raw_weights)

    token_records = []
    for pos, (tok, seg, w) in enumerate(zip(tokens, segments, weights)):
        token_records.append({
            "position": pos,
            "token": tok,
            "segment": seg,
            "weight": float(np.round(w, 5)),
        })

    # Top-k excluding special tokens
    content_records = [r for r in token_records if r["segment"] != "special"]
    top_records = sorted(content_records, key=lambda x: x["weight"], reverse=True)[:top_k]

    return {
        "available": True,
        "method": "mock_last_layer_cls",
        "top_k": top_k,
        "disclaimer": ATTENTION_DISCLAIMER,
        "tokens": tokens,
        "weights": [r["weight"] for r in token_records],
        "top_tokens": top_records,
    }


def compute_attention_attribution(
    model: Any | None,
    tokenizer: Any | None,
    original_text: str,
    simplified_text: str,
    top_k: int = 10,
    method: str = "last_layer_cls",
) -> dict[str, Any]:
    """Extract last-layer [CLS] attention attribution from a loaded JudgeBert model.

    Safely handles:
    - Missing model or tokenizer
    - Models executing without attentions enabled
    - Token conversion and segment labeling
    """
    if model is None or tokenizer is None:
        return {
            "available": False,
            "reason": "Model or tokenizer not provided; attention attribution unavailable.",
            "method": method,
            "disclaimer": ATTENTION_DISCLAIMER,
            "top_tokens": [],
        }

    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "reason": "PyTorch is not installed in the active environment.",
            "method": method,
            "disclaimer": ATTENTION_DISCLAIMER,
            "top_tokens": [],
        }

    try:
        # Tokenize pair
        inputs = tokenizer(
            original_text,
            simplified_text,
            return_tensors="pt",
            truncation=True,
            max_length=192,
        )

        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model.eval()
        with torch.no_grad():
            output = model(**inputs, output_attentions=True)

        attentions = getattr(output, "attentions", None)
        if attentions is None or len(attentions) == 0:
            return {
                "available": False,
                "reason": "Model was executed without attentions or encoder does not return attentions.",
                "method": method,
                "disclaimer": ATTENTION_DISCLAIMER,
                "top_tokens": [],
            }

        # Last layer attention: shape (batch, num_heads, seq_len, seq_len)
        last_layer = attentions[-1]
        mean_attention = last_layer.mean(dim=1).squeeze(0)  # (seq_len, seq_len)

        # [CLS] token is at index 0: row 0 represents CLS attention over all tokens
        cls_weights = mean_attention[0, :].cpu().numpy()

        input_ids = inputs["input_ids"][0].cpu().tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        token_type_ids = (
            inputs["token_type_ids"][0].cpu().tolist()
            if "token_type_ids" in inputs
            else [0] * len(tokens)
        )

        token_records = []
        for pos, (tok, ttype, w) in enumerate(zip(tokens, token_type_ids, cls_weights)):
            seg = "original" if ttype == 0 else "simplified"
            if tok in ("[CLS]", "[SEP]", "[PAD]"):
                seg = "special"
            token_records.append({
                "position": pos,
                "token": tok,
                "segment": seg,
                "weight": float(np.round(w, 5)),
            })

        content_records = [r for r in token_records if r["segment"] != "special"]
        top_records = sorted(content_records, key=lambda x: x["weight"], reverse=True)[:top_k]

        return {
            "available": True,
            "method": method,
            "top_k": top_k,
            "disclaimer": ATTENTION_DISCLAIMER,
            "tokens": tokens,
            "weights": [r["weight"] for r in token_records],
            "top_tokens": top_records,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Attention extraction failed: {exc}",
            "method": method,
            "disclaimer": ATTENTION_DISCLAIMER,
            "top_tokens": [],
        }
