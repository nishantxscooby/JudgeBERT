"""Explainability layer for legal meaning preservation.

Provides auditable surface cue alignment as the primary explanatory signal,
along with supporting attention attribution and visual highlight helpers.

Strict Integrity Guarantees:
- Explanations describe text and model behavior only; they never fabricate
  or substitute human gold ground truth.
- Attention weights are strictly demarcated as weak supporting evidence
  and are never presented as authoritative legal reasoning.
"""

from __future__ import annotations

from src.explainability.attention_attribution import (
    ATTENTION_DISCLAIMER,
    compute_attention_attribution,
    compute_mock_attention_attribution,
)
from src.explainability.cue_alignment import (
    align_legal_cues,
    extract_all_cues,
    get_visual_highlights,
    render_highlighted_html,
)
from src.explainability.explanations import explain_pair

ATTENTION_LEGAL_DISCLAIMER = ATTENTION_DISCLAIMER

__all__ = [
    "ATTENTION_DISCLAIMER",
    "ATTENTION_LEGAL_DISCLAIMER",
    "align_legal_cues",
    "compute_attention_attribution",
    "compute_mock_attention_attribution",
    "explain_pair",
    "extract_all_cues",
    "get_visual_highlights",
    "render_highlighted_html",
]
