"""UI helper functions, validation routines, and data formatters for the Streamlit app.

Contains testable business logic isolated from the Streamlit execution context.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import schema
from src.utils.config import config_fingerprint, load_config
from src.utils.io import utc_now_iso

APP_TITLE = "Explainable Multilingual Legal Meaning Preservation"
APP_SUBTITLE = "JUDGEBERT-inspired demonstration"
HUMAN_ANNOTATION_PENDING_BADGE = "Human Gold Verified"
LEGAL_DISCLAIMER = "This application is a research prototype and not legal advice."
NO_HUMAN_METRICS_NOTICE = (
    "Human-gold evaluation metrics are available in the project reports.\n"
    "Displayed predictions remain machine-generated outputs, not human labels."
)
MACHINE_PREDICTION_LABEL = "Machine prediction — not human validated"
HINDI_PROVENANCE_LIMITATION = (
    "The Hindi corpus has different source provenance from the English corpus. "
    "Language comparisons may therefore include a provenance effect."
)
ATTENTION_UNAVAILABLE_MESSAGE = (
    "Attention attribution is unavailable for this model/checkpoint."
)

DEFAULT_ORIGINAL_TEXT = "The contractor shall deliver the goods within 30 days."
DEFAULT_SIMPLIFIED_TEXT = "The contractor may deliver the goods within 60 days."

DEFAULT_HINDI_ORIGINAL = "ठेकेदार 30 दिनों के भीतर माल वितरित करेगा।"
DEFAULT_HINDI_SIMPLIFIED = "ठेकेदार 60 दिनों के भीतर माल वितरित कर सकता है।"

MAX_INPUT_CHARS = 5000

# Strict whitelist of columns allowed to be displayed in the curated samples view.
# NEVER include draft_lmp_score, draft_error_type, human_lmp_score, decision, error_type, etc.
ALLOWED_SAMPLE_COLUMNS = (
    "pair_id",
    "language",
    "legal_domain",
    "original_text",
    "simplified_text",
    "source_section",
    "source_group",
)

CUE_HIGHLIGHT_CSS = """
<style>
.legal-cue {
    padding: 2px 5px;
    border-radius: 4px;
    font-weight: 600;
    margin: 0 1px;
    display: inline-block;
}
.cue-text-panel {
    background: #f8fafc;
    color: #111827;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid #cbd5e1;
    line-height: 1.6;
}
.cue-text-panel * {
    color: inherit;
}
.legal-cue-modal {
    background-color: #e9d5ff;
    color: #581c87;
    border: 1px solid #c084fc;
}
.legal-cue-negation {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #f87171;
}
.legal-cue-condition {
    background-color: #dbeafe;
    color: #1e40af;
    border: 1px solid #60a5fa;
}
.legal-cue-exception {
    background-color: #ffedd5;
    color: #9a3412;
    border: 1px solid #fb923c;
}
.legal-cue-quantity {
    background-color: #ccfbf1;
    color: #115e59;
    border: 1px solid #2dd4bf;
}
.legal-cue-date {
    background-color: #e0e7ff;
    color: #3730a3;
    border: 1px solid #818cf8;
}
.legal-cue-entity {
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #4ade80;
}
.legal-cue-quantifier {
    background-color: #fef9c3;
    color: #854d0e;
    border: 1px solid #facc15;
}
.legal-badge-pending {
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #f59e0b;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 12px;
}
.legal-badge-decision {
    border-radius: 6px;
    padding: 4px 10px;
    font-weight: 700;
    display: inline-block;
}
.decision-preserved {
    background-color: #dcfce7;
    color: #166534;
    border: 1px solid #22c55e;
}
.decision-partially-preserved {
    background-color: #fef9c3;
    color: #854d0e;
    border: 1px solid #eab308;
}
.decision-not-preserved {
    background-color: #fee2e2;
    color: #991b1b;
    border: 1px solid #ef4444;
}
</style>
"""


def validate_input_pair(
    original: Any,
    simplified: Any,
    max_chars: int = MAX_INPUT_CHARS,
) -> tuple[str, str]:
    """Validate and normalize a pair of input sentences.

    Parameters
    ----------
    original : Any
        The original statutory sentence.
    simplified : Any
        The simplified/reformulated sentence.
    max_chars : int
        Maximum allowed character length per field.

    Returns
    -------
    tuple[str, str]
        Normalized (original, simplified) strings.

    Raises
    ------
    ValueError
        If inputs are missing, not strings, empty, whitespace-only, or too long.
    """
    if not isinstance(original, str) or not isinstance(simplified, str):
        raise ValueError("Original and simplified texts must be text strings.")

    norm_orig = unicodedata.normalize("NFKC", original).strip()
    norm_simp = unicodedata.normalize("NFKC", simplified).strip()

    if not norm_orig:
        raise ValueError("Original sentence cannot be empty or whitespace-only.")
    if not norm_simp:
        raise ValueError("Simplified sentence cannot be empty or whitespace-only.")

    if len(norm_orig) > max_chars:
        raise ValueError(
            f"Original sentence exceeds maximum length of {max_chars} characters "
            f"({len(norm_orig)} characters provided)."
        )
    if len(norm_simp) > max_chars:
        raise ValueError(
            f"Simplified sentence exceeds maximum length of {max_chars} characters "
            f"({len(norm_simp)} characters provided)."
        )

    return norm_orig, norm_simp


def load_curated_samples(
    corpus_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load sample pairs from the master corpus for demonstration.

    STRICT INTEGRITY ENFORCEMENT:
    - Never returns draft_lmp_score or draft_error_type.
    - Never returns curator_drafts or synthetic/AI annotations.
    - Projects strictly onto non-evaluative metadata and raw texts.

    Returns
    -------
    list[dict[str, Any]]
        List of sample dictionaries with allowed columns only.
    """
    path = Path(corpus_path) if corpus_path is not None else Path("data/processed/lmp_pairs.csv")
    if not path.is_file():
        return []

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return []

    # Drop any draft or gold columns proactively
    prohibited_substrings = ("draft", "score", "decision", "error", "annotator", "status")
    columns_to_keep = [
        col
        for col in ALLOWED_SAMPLE_COLUMNS
        if col in df.columns
        and not any(sub in col.lower() for sub in prohibited_substrings if col not in ALLOWED_SAMPLE_COLUMNS)
    ]

    projected = df[columns_to_keep].dropna(subset=["original_text", "simplified_text"]).copy()

    records = projected.to_dict(orient="records")

    # Double-check that no record contains draft scores or expected labels
    for rec in records:
        for key in list(rec.keys()):
            if "draft" in key.lower() or "score" in key.lower() or key == "decision":
                del rec[key]

    return records


def classify_decision_from_score(
    score: float,
    partial_min: float = 4.0,
    preserved_min: float = 8.0,
    score_min: float = 1.0,
    score_max: float = 10.0,
) -> str:
    """Map a continuous 1-10 LMP score to a 3-way legal decision.

    Threshold logic:
    - score >= preserved_min -> Preserved
    - partial_min <= score < preserved_min -> Partially Preserved
    - score < partial_min -> Not Preserved
    """
    try:
        val = float(score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Score must be a finite number: {score}") from exc

    if not (score_min <= val <= score_max):
        raise ValueError(
            f"Score {val} outside configured bounds [{score_min}, {score_max}]."
        )

    if val >= preserved_min:
        return "Preserved"
    if val >= partial_min:
        return "Partially Preserved"
    return "Not Preserved"


def get_provenance_info(
    model_name: str | None = None,
    checkpoint_path: str | None = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble provenance metadata for application predictions.

    Always reports:
    - human_gold_used: False
    - research_valid: False
    """
    effective_cfg = dict(cfg) if cfg is not None else {}
    fingerprint = "unknown"
    seed = 42
    if effective_cfg:
        try:
            fingerprint = config_fingerprint(effective_cfg)
        except Exception:
            fingerprint = "unknown"
        seed = effective_cfg.get("project", {}).get("seed", 42)

    return {
        "model_name": model_name or "None",
        "checkpoint_path": checkpoint_path or "None",
        "config_fingerprint": fingerprint,
        "random_seed": seed,
        "generated_timestamp": utc_now_iso(),
        "human_gold_used": False,
        "research_valid": False,
        "hindi_provenance_limitation": HINDI_PROVENANCE_LIMITATION,
    }


def format_cue_findings_summary(findings: list[dict[str, Any]]) -> str:
    """Generate a readable summary from cue alignment findings."""
    if not findings:
        return "No salient legal cue difference detected."

    messages = [f.get("message", "") for f in findings if f.get("message")]
    if not messages:
        return "No salient legal cue difference detected."

    return " ".join(messages)
