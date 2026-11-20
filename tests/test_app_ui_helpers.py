"""Unit tests for src.app.ui_helpers."""

from __future__ import annotations

from pathlib import Path
import pytest

from src.app.ui_helpers import (
    CUE_HIGHLIGHT_CSS,
    DEFAULT_ORIGINAL_TEXT,
    DEFAULT_SIMPLIFIED_TEXT,
    HINDI_PROVENANCE_LIMITATION,
    HUMAN_ANNOTATION_PENDING_BADGE,
    LEGAL_DISCLAIMER,
    MACHINE_PREDICTION_LABEL,
    NO_HUMAN_METRICS_NOTICE,
    classify_decision_from_score,
    format_cue_findings_summary,
    get_provenance_info,
    load_curated_samples,
    validate_input_pair,
)
from src.data.schema import DECISIONS


def test_constants_integrity():
    """Verify non-negotiable text constants."""
    assert HUMAN_ANNOTATION_PENDING_BADGE == "Human Gold Verified"
    assert "not legal advice" in LEGAL_DISCLAIMER
    assert "Human-gold evaluation metrics are available" in NO_HUMAN_METRICS_NOTICE
    assert MACHINE_PREDICTION_LABEL == "Machine prediction — not human validated"
    assert "provenance effect" in HINDI_PROVENANCE_LIMITATION
    assert "<style>" in CUE_HIGHLIGHT_CSS


def test_validate_input_pair_valid_english():
    orig, simp = validate_input_pair("  Party shall pay.  ", " Party may pay. ")
    assert orig == "Party shall pay."
    assert simp == "Party may pay."


def test_validate_input_pair_preserves_hindi_unicode():
    orig_in = "ठेकेदार 30 दिनों के भीतर माल वितरित करेगा।"
    simp_in = "ठेकेदार 60 दिनों के भीतर माल वितरित कर सकता है।"
    orig, simp = validate_input_pair(orig_in, simp_in)
    assert orig == orig_in
    assert simp == simp_in
    assert "ठेकेदार" in orig
    assert "सकता है" in simp


@pytest.mark.parametrize(
    "orig,simp,expected_err",
    [
        ("", "Valid text", "empty or whitespace-only"),
        ("   ", "Valid text", "empty or whitespace-only"),
        ("Valid text", "", "empty or whitespace-only"),
        ("Valid text", "   \t\n  ", "empty or whitespace-only"),
        (None, "Valid text", "text strings"),
        ("Valid text", 123, "text strings"),
        (True, False, "text strings"),
    ],
)
def test_validate_input_pair_rejects_empty_or_non_string(orig, simp, expected_err):
    with pytest.raises(ValueError, match=expected_err):
        validate_input_pair(orig, simp)


def test_validate_input_pair_rejects_excessive_length():
    too_long = "a" * 5001
    with pytest.raises(ValueError, match="exceeds maximum length"):
        validate_input_pair(too_long, "Short text")

    with pytest.raises(ValueError, match="exceeds maximum length"):
        validate_input_pair("Short text", too_long)


def test_load_curated_samples_integrity():
    """Verify that sample loader never exposes draft or gold evaluation columns."""
    samples = load_curated_samples("data/processed/lmp_pairs.csv")
    assert len(samples) > 0

    prohibited_fields = {
        "draft_lmp_score",
        "draft_error_type",
        "human_lmp_score",
        "error_type",
        "decision",
        "annotator_id",
        "score_bucket",
    }

    for s in samples:
        keys = set(s.keys())
        leakage = keys.intersection(prohibited_fields)
        assert not leakage, f"Sample leaked prohibited fields: {leakage}"
        assert "pair_id" in s
        assert "original_text" in s
        assert "simplified_text" in s
        assert "language" in s
        assert isinstance(s["original_text"], str)
        assert isinstance(s["simplified_text"], str)


def test_load_curated_samples_missing_file_returns_empty(tmp_path: Path):
    non_existent = tmp_path / "does_not_exist.csv"
    assert load_curated_samples(non_existent) == []


def test_classify_decision_from_score():
    assert classify_decision_from_score(10.0) == "Preserved"
    assert classify_decision_from_score(8.0) == "Preserved"
    assert classify_decision_from_score(8.5) == "Preserved"

    assert classify_decision_from_score(7.9) == "Partially Preserved"
    assert classify_decision_from_score(4.0) == "Partially Preserved"
    assert classify_decision_from_score(5.5) == "Partially Preserved"

    assert classify_decision_from_score(3.9) == "Not Preserved"
    assert classify_decision_from_score(1.0) == "Not Preserved"
    assert classify_decision_from_score(2.0) == "Not Preserved"

    for score in [1.0, 4.0, 8.0]:
        assert classify_decision_from_score(score) in DECISIONS


@pytest.mark.parametrize("invalid_score", [0.9, 10.1, -5.0, 15.0, "abc", None])
def test_classify_decision_from_score_bounds_check(invalid_score):
    with pytest.raises(ValueError):
        classify_decision_from_score(invalid_score)


def test_get_provenance_info():
    prov = get_provenance_info(model_name="TestModel", checkpoint_path="path/to/ckpt")
    assert prov["model_name"] == "TestModel"
    assert prov["checkpoint_path"] == "path/to/ckpt"
    assert prov["human_gold_used"] is False
    assert prov["research_valid"] is False
    assert "generated_timestamp" in prov
    assert "provenance effect" in prov["hindi_provenance_limitation"]


def test_format_cue_findings_summary():
    assert format_cue_findings_summary([]) == "No salient legal cue difference detected."

    findings = [
        {"message": "Potential obligation mismatch: modal weakened."},
        {"message": "Quantity differs: 30 days vs 60 days."},
    ]
    summary = format_cue_findings_summary(findings)
    assert "obligation mismatch" in summary
    assert "Quantity differs" in summary
