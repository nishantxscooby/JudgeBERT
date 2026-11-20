"""Unit tests for src.app.model_registry."""

from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

from src.app.model_registry import (
    MODEL_MURIL,
    MODEL_NONE,
    MODEL_SBERT,
    MODEL_TFIDF,
    MURIL_UNAVAILABLE_MESSAGE,
    SBERT_UNAVAILABLE_MESSAGE,
    TFIDF_UNAVAILABLE_MESSAGE,
    AppPredictionResult,
    InvalidModelError,
    ModelUnavailableError,
    PredictionError,
    check_model_availability,
    load_model,
    predict_pair,
    register_custom_model,
    unregister_custom_model,
    _scale_lmp_piecewise,
)
from src.data.schema import DECISIONS


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (5.50, 8.5),
        (5.35, 9.5),
        (5.25, 5.5),
        (4.90, 3.5),
        (5.60, 1.5),
        (5.10, 7.0),
    ],
)
def test_piecewise_lmp_scale(source: float, expected: float):
    assert _scale_lmp_piecewise(source) == pytest.approx(expected)


def test_supported_models():
    for name in [MODEL_MURIL, MODEL_TFIDF, MODEL_SBERT, MODEL_NONE]:
        info = check_model_availability(name)
        assert info.name == name


def test_check_model_availability_muril_missing_checkpoint(tmp_path: Path):
    """When checkpoints directory is empty, JudgeBERT availability must be False."""
    info = check_model_availability(MODEL_MURIL, checkpoint_override=tmp_path)
    assert info.available is False
    assert info.status_message == MURIL_UNAVAILABLE_MESSAGE
    assert info.checkpoint_path is None


def test_check_model_availability_rejects_collapsed_predictions(tmp_path: Path):
    checkpoint_dir = tmp_path / "judgebert"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"existing checkpoint")
    pd.DataFrame({"prediction": [4.625] * 20}).to_csv(
        checkpoint_dir / "test_predictions.csv",
        index=False,
    )

    info = check_model_availability(MODEL_MURIL, checkpoint_override=tmp_path)

    assert info.available is False
    assert info.checkpoint_path is None
    assert "degenerate" in info.status_message


def test_check_model_availability_tfidf_missing(tmp_path: Path):
    info = check_model_availability(MODEL_TFIDF, checkpoint_override=tmp_path)
    assert info.available is False
    assert info.status_message == TFIDF_UNAVAILABLE_MESSAGE


def test_check_model_availability_sbert_missing(tmp_path: Path):
    info = check_model_availability(MODEL_SBERT, checkpoint_override=tmp_path)
    assert info.available is False
    assert info.status_message == SBERT_UNAVAILABLE_MESSAGE


def test_check_model_availability_none():
    info = check_model_availability(MODEL_NONE)
    assert info.available is True
    assert "Cue alignment only" in info.status_message


def test_invalid_model_name_raises():
    with pytest.raises(InvalidModelError, match="Unsupported model name"):
        check_model_availability("NonExistentModel")

    with pytest.raises(InvalidModelError, match="Unsupported model name"):
        load_model("NonExistentModel")


def test_load_unavailable_model_raises_without_fabrication(tmp_path: Path):
    with pytest.raises(ModelUnavailableError, match="checkpoint unavailable"):
        load_model(MODEL_MURIL, checkpoint_override=tmp_path)


def test_predict_pair_with_none_model_raises():
    with pytest.raises(ValueError, match="Cannot predict"):
        predict_pair(MODEL_NONE, "The party shall pay.", "The party may pay.")


def test_predict_pair_unavailable_model_raises_without_fallback(tmp_path: Path):
    with pytest.raises(ModelUnavailableError):
        predict_pair(
            MODEL_MURIL,
            "The party shall pay.",
            "The party may pay.",
            checkpoint_override=tmp_path,
        )


def test_predict_pair_with_custom_registered_mock_model():
    mock_name = "mock_test_judgebert"
    # Return 6.5
    register_custom_model(mock_name, lambda o, s, l: 6.5)

    try:
        res = predict_pair(
            mock_name,
            original="The party shall notify within 30 days.",
            simplified="The party may notify within 60 days.",
            language="en",
        )
        assert isinstance(res, AppPredictionResult)
        assert res.model_name == mock_name
        assert res.predicted_score == 6.5
        assert res.decision == "Partially Preserved"
        assert res.decision in DECISIONS
        assert res.status == "ok"
        assert res.machine_prediction_label == "Machine prediction — not human validated"
        assert res.provenance["human_gold_used"] is False
        assert res.provenance["research_valid"] is False
    finally:
        unregister_custom_model(mock_name)


def test_predict_pair_with_custom_mock_attention():
    mock_name = "mock_test_attention"
    att_dict = {
        "available": True,
        "top_tokens": [{"token": "shall", "weight": 0.42}],
    }
    register_custom_model(mock_name, lambda o, s, l: (9.0, att_dict))

    try:
        res = predict_pair(
            mock_name,
            original="The party shall notify.",
            simplified="The party shall notify.",
            language="en",
        )
        assert res.predicted_score == 9.0
        assert res.decision == "Preserved"
        assert res.attention_data == att_dict
    finally:
        unregister_custom_model(mock_name)


def test_predict_pair_clips_scores_strictly():
    mock_name = "mock_out_of_bounds"
    # Returns 15.0 (out of bounds)
    register_custom_model(mock_name, lambda o, s, l: 15.0)

    try:
        res = predict_pair(mock_name, "text1", "text2")
        assert res.predicted_score == 10.0
    finally:
        unregister_custom_model(mock_name)

    # Returns -5.0 (out of bounds)
    register_custom_model(mock_name, lambda o, s, l: -5.0)
    try:
        res = predict_pair(mock_name, "text1", "text2")
        assert res.predicted_score == 1.0
    finally:
        unregister_custom_model(mock_name)
