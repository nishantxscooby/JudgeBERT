"""Integration tests for the Streamlit application using AppTest."""

from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd
import pytest
import streamlit.testing.v1 as st_test

from src.app import streamlit_app
from src.app.model_registry import (
    MODEL_MURIL,
    MODEL_NONE,
    MURIL_UNAVAILABLE_MESSAGE,
    check_model_availability,
    register_custom_model,
    unregister_custom_model,
)
from src.app.ui_helpers import (
    HINDI_PROVENANCE_LIMITATION,
    HUMAN_ANNOTATION_PENDING_BADGE,
    LEGAL_DISCLAIMER,
    MACHINE_PREDICTION_LABEL,
    NO_HUMAN_METRICS_NOTICE,
)
from src.explainability import ATTENTION_LEGAL_DISCLAIMER


APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "app" / "streamlit_app.py")


def _rendered_text(at: st_test.AppTest) -> str:
    """Concatenate all visible text strings from markdown, info, warning, error, title, etc."""
    parts: list[str] = []
    collections = ("title", "subheader", "caption", "markdown", "info", "warning", "error", "success")
    for collection in collections:
        for elem in getattr(at, collection, []):
            val = getattr(elem, "value", "")
            if val:
                parts.append(str(val))
    # Also check sidebar
    if hasattr(at, "sidebar"):
        for collection in collections:
            for elem in getattr(at.sidebar, collection, []):
                val = getattr(elem, "value", "")
                if val:
                    parts.append(str(val))
    return "\n".join(parts)


def test_import_streamlit_app():
    """Verify streamlit_app imports cleanly without starting a server or raising."""
    assert hasattr(streamlit_app, "main")
    assert hasattr(streamlit_app, "render_header")
    assert callable(streamlit_app.main)


def test_app_renders_without_exceptions():
    at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception


def test_app_displays_human_annotation_pending_badge():
    at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    text = _rendered_text(at)
    assert HUMAN_ANNOTATION_PENDING_BADGE in text


def test_app_displays_legal_and_metrics_disclaimers():
    at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    text = _rendered_text(at)
    assert "not legal advice" in text
    assert "Human-gold evaluation metrics are available" in text


def test_app_displays_attention_disclaimer():
    at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    text = _rendered_text(at)
    assert "Attention weights are not guaranteed to be faithful legal explanations" in text


def test_app_displays_hindi_provenance_limitation():
    at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    text = _rendered_text(at)
    assert "provenance effect" in text


def test_app_handles_calibrated_checkpoint_status_dynamically():
    """The app must match its registry status without assuming bundled weights."""
    at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    errors = [e.value for e in at.error]
    muril_info = check_model_availability(MODEL_MURIL)
    unavailable_shown = any(MURIL_UNAVAILABLE_MESSAGE in str(err) for err in errors)
    assert unavailable_shown is (not muril_info.available)

    # Cue alignment should still have run
    text = _rendered_text(at)
    assert "Primary Auditable Legal Cue Alignment" in text
    assert "Potential obligation mismatch" in text or "obligation" in text.lower()


def test_app_none_model_selection_runs_cleanly():
    at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    # Select 'None (Cue alignment only)' in sidebar selectbox (index 1 of selectboxes)
    at.sidebar.selectbox[1].select(MODEL_NONE)
    at.run()

    # No errors should be present
    assert len(at.error) == 0
    text = _rendered_text(at)
    assert "Primary Auditable Legal Cue Alignment" in text


def test_app_with_custom_mock_model_renders_prediction():
    mock_name = "mock_model_app_test"
    register_custom_model(mock_name, lambda o, s, l: 8.5)

    try:
        at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.sidebar.selectbox[1].select(mock_name)
        at.run()

        assert not at.exception
        text = _rendered_text(at)
        assert MACHINE_PREDICTION_LABEL in text
        assert "Preserved" in text
    finally:
        unregister_custom_model(mock_name)


def test_app_curated_samples_never_exposes_drafts():
    at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    text = _rendered_text(at)
    assert "draft_lmp_score" not in text
    assert "draft_error_type" not in text
    assert "curator_drafts" not in text


def test_app_leaves_master_dataset_untouched():
    from src.data.load_data import assert_gold

    p = Path("data/processed/lmp_pairs.csv")
    digest_before = hashlib.sha256(p.read_bytes()).hexdigest().upper()

    at = st_test.AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    digest_after = hashlib.sha256(p.read_bytes()).hexdigest().upper()
    assert digest_before == digest_after

    df = pd.read_csv(p)
    assert len(df) == 340
    assert_gold(df)
    assert df["human_lmp_score"].notna().sum() == 340
    assert (df["human_lmp_score"] != df["draft_lmp_score"]).any()
