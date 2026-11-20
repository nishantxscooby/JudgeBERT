import pandas as pd
import streamlit as st
from src.annotation import annotation_app as app

def test_review_dataframe_no_string_width(monkeypatch):
    # Prepare a minimal corpus DataFrame with one row and a dummy annotator
    corpus = pd.DataFrame({
        "pair_id": ["P1"],
        "human_lmp_score": [5],
        "decision": ["Preserved"],
        "error_type": [""],
        "notes": [""],
        "annotated_at": [pd.Timestamp.now()],
        "language": ["en"],
        "legal_domain": ["contract"],
        "source": ["src"],
        "source_section": ["1"],
    })
    captured = {}
    def fake_dataframe(df, *args, **kwargs):
        captured["kwargs"] = kwargs
        return df
    monkeypatch.setattr(st, "dataframe", fake_dataframe)
    monkeypatch.setattr(app, "_load_own_scores", lambda _: corpus.copy())
    # Run the review tab (it will invoke our fake dataframe)
    app.tab_review(corpus, "A01")
    # Verify that width is not a string and use_container_width is used (or omitted)
    assert "width" not in captured["kwargs"] or isinstance(captured["kwargs"].get("width"), int)
    # In our implementation we use use_container_width=True
    assert captured["kwargs"].get("use_container_width") is True
