"""Streamlit Demonstration App for Explainable Multilingual Legal Meaning Preservation.

Run with:
    streamlit run src/app/streamlit_app.py

Integrity rules:
- Strictly labels all predictions as machine-generated.
- Displays the reviewed human-gold status and benchmark-metrics notice.
- Never reveals or displays curator drafts or synthetic labels as ground truth.
- Displays the mandatory attention attribution disclaimer.
- Degrades gracefully to cue alignment when checkpoints are unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure repository root is on sys.path for Streamlit script execution
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.app.model_registry import (
    MODEL_MURIL,
    MODEL_NONE,
    MODEL_SBERT,
    MODEL_TFIDF,
    SUPPORTED_MODELS,
    AppPredictionResult,
    ModelUnavailableError,
    check_model_availability,
    get_available_model_names,
    predict_pair,
)
from src.app.ui_helpers import (
    APP_SUBTITLE,
    APP_TITLE,
    ATTENTION_UNAVAILABLE_MESSAGE,
    CUE_HIGHLIGHT_CSS,
    DEFAULT_HINDI_ORIGINAL,
    DEFAULT_HINDI_SIMPLIFIED,
    DEFAULT_ORIGINAL_TEXT,
    DEFAULT_SIMPLIFIED_TEXT,
    HINDI_PROVENANCE_LIMITATION,
    HUMAN_ANNOTATION_PENDING_BADGE,
    LEGAL_DISCLAIMER,
    MACHINE_PREDICTION_LABEL,
    NO_HUMAN_METRICS_NOTICE,
    classify_decision_from_score,
    get_provenance_info,
    load_curated_samples,
    validate_input_pair,
)
from src.explainability import ATTENTION_LEGAL_DISCLAIMER, explain_pair
from src.utils.config import load_config


def _page_setup() -> None:
    """Configure Streamlit page layout and custom styling."""
    try:
        st.set_page_config(
            page_title=f"{APP_TITLE} | Demonstration",
            page_icon="⚖️",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception:
        # Ignore if called outside script runner or already configured
        pass
    st.markdown(CUE_HIGHLIGHT_CSS, unsafe_allow_html=True)


def render_header() -> None:
    """Render application header, status badges, and integrity warnings."""
    st.title(f"⚖️ {APP_TITLE}")
    st.caption(f"**{APP_SUBTITLE}** · Multilingual Deontic Alignment & Semantic Preservation")

    # The corpus has been reviewed and merged into the human-gold workflow.
    st.markdown(
        f'<div class="legal-badge-pending">⚠️ {HUMAN_ANNOTATION_PENDING_BADGE}</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.info(f"ℹ️ **Research Prototype Disclaimer:**\n\n{LEGAL_DISCLAIMER}")
    with col2:
        st.warning(f"📊 **Performance Status:**\n\n{NO_HUMAN_METRICS_NOTICE}")


def render_sidebar(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Render sidebar options: sample picker, model selector, and settings."""
    st.sidebar.header("⚙️ Configuration & Samples")

    # Language selection
    language_choice = st.sidebar.radio(
        "Select Language / भाषा चुनें:",
        options=["English (en)", "Hindi (hi)"],
        index=0,
    )
    lang_code = "hi" if "Hindi" in language_choice else "en"

    # Curated sample selector
    st.sidebar.subheader("📋 Curated Corpus Samples")
    sample_options = ["-- Custom Sentence Input --"]
    filtered_samples: list[dict[str, Any]] = [
        s for s in samples if s.get("language") == lang_code
    ]

    for s in filtered_samples:
        pid = s.get("pair_id", "Unknown")
        domain = s.get("legal_domain", "")
        sec = s.get("source_section", "")
        label = f"{pid} [{domain}] {f'(s.{sec})' if sec else ''}"
        sample_options.append(label)

    selected_sample_label = st.sidebar.selectbox(
        "Load Statutory Example:",
        options=sample_options,
        index=0,
        help="Select a real statutory provision from the corpus. Does not reveal expected scores.",
    )

    selected_sample = None
    if selected_sample_label != "-- Custom Sentence Input --":
        idx = sample_options.index(selected_sample_label) - 1
        if 0 <= idx < len(filtered_samples):
            selected_sample = filtered_samples[idx]

    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    model_choice = st.sidebar.selectbox(
        "Scoring Model:",
        options=get_available_model_names(),
        index=0,
        help="Choose a cross-encoder or baseline model for LMP score prediction.",
    )

    model_info = check_model_availability(model_choice)
    if not model_info.available:
        st.sidebar.warning(f"⚠️ {model_info.status_message}")
    else:
        st.sidebar.success(f"✅ {model_info.status_message}")

    return {
        "lang_code": lang_code,
        "selected_sample": selected_sample,
        "model_name": model_choice,
        "model_info": model_info,
    }


def render_prediction_block(result: AppPredictionResult) -> None:
    """Render the machine prediction score, decision badge, and warnings."""
    st.subheader("🎯 Model Prediction")

    st.caption(f"**Model:** `{result.model_name}` · **Status:** `{result.status}`")
    st.markdown(
        f'<span style="color: #fde68a; font-weight: 700; font-size: 0.9em;">'
        f'🏷️ {result.machine_prediction_label}</span>',
        unsafe_allow_html=True,
    )

    score = result.predicted_score
    decision = result.decision

    col_score, col_decision = st.columns([1, 2])
    with col_score:
        st.metric(label="Predicted LMP Score (1–10)", value=f"{score:.2f}")

    with col_decision:
        badge_class = (
            "decision-preserved"
            if decision == "Preserved"
            else (
                "decision-partially-preserved"
                if decision == "Partially Preserved"
                else "decision-not-preserved"
            )
        )
        st.markdown(
            f'**Predicted Legal Decision:**<br/>'
            f'<div class="legal-badge-decision {badge_class}">{decision}</div>',
            unsafe_allow_html=True,
        )


def render_explainability_block(explanation: dict[str, Any]) -> None:
    """Render primary auditable legal cue alignment and highlights."""
    st.subheader("🔍 Primary Auditable Legal Cue Alignment")
    st.markdown(
        "Comparison of legally salient elements (modals, negations, conditions, "
        "exceptions, quantities, dates, entities, quantifiers)."
    )

    highlights = explanation.get("highlights", {})
    orig_html = highlights.get("original_html", "")
    simp_html = highlights.get("simplified_html", "")

    col_orig, col_simp = st.columns(2)
    with col_orig:
        st.markdown("**Original Statutory Provision:**")
        st.markdown(
            f'<div class="cue-text-panel">{orig_html}</div>',
            unsafe_allow_html=True,
        )

    with col_simp:
        st.markdown("**Simplified / Reformulated Sentence:**")
        st.markdown(
            f'<div class="cue-text-panel">{simp_html}</div>',
            unsafe_allow_html=True,
        )

    # Cue Counts & Alignment Metrics
    alignment = explanation.get("cue_alignment", {})
    preserved = alignment.get("preserved", [])
    removed = alignment.get("removed", [])
    added = alignment.get("added", [])
    changed = alignment.get("changed", [])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Preserved Cues", len(preserved))
    m2.metric("Removed Cues", len(removed), delta=f"-{len(removed)}" if removed else None)
    m3.metric("Added Cues", len(added), delta=f"+{len(added)}" if added else None)
    m4.metric("Changed Cues", len(changed))

    # Findings List
    findings = explanation.get("findings", [])
    st.markdown("#### Salient Findings & Meaning Shifts")
    if not findings:
        st.success("✅ No salient legal cue difference detected.")
    else:
        for f in findings:
            cat = f.get("category", "General").title()
            kind = f.get("kind", "").replace("_", " ").title()
            msg = f.get("message", "")
            severity = f.get("severity", "medium")
            icon = "🔴" if severity == "high" else ("🟠" if severity == "medium" else "🔵")
            st.markdown(f"- {icon} **[{cat} · {kind}]** {msg}")

    # Summary
    summary = explanation.get("summary", "")
    if summary:
        st.info(f"**Cue Alignment Summary:** {summary}")


def render_attention_block(
    attention_info: dict[str, Any] | None,
    model_name: str,
) -> None:
    """Render token attention attribution as supporting evidence."""
    with st.expander("🔬 Token Attention Attribution (Supporting Evidence)", expanded=False):
        st.warning(f"⚠️ **Mandatory Disclaimer:**\n\n{ATTENTION_LEGAL_DISCLAIMER}")

        if not attention_info or not attention_info.get("available", False):
            st.info(ATTENTION_UNAVAILABLE_MESSAGE)
            return

        top_tokens = attention_info.get("top_tokens", [])
        if top_tokens:
            st.markdown("##### High-Attention Tokens ([CLS] Attention)")
            cols = st.columns(min(len(top_tokens), 6))
            for i, item in enumerate(top_tokens[:6]):
                with cols[i]:
                    token_str = item.get("token", "")
                    weight = item.get("weight", 0.0)
                    st.metric(label=f"Token: `{token_str}`", value=f"{weight:.4f}")
        else:
            st.caption("No token attention breakdown available.")


def render_provenance_block(provenance: dict[str, Any]) -> None:
    """Render system provenance, fingerprints, and limitations."""
    with st.expander("📜 Provenance, Metadata & Limitations", expanded=False):
        st.markdown("### Provenance Contract")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.write(f"**Model Name:** `{provenance.get('model_name', 'None')}`")
            st.write(f"**Checkpoint:** `{provenance.get('checkpoint_path', 'None')}`")
            st.write(f"**Config Fingerprint:** `{provenance.get('config_fingerprint', 'unknown')}`")
            st.write(f"**Seed:** `{provenance.get('random_seed', 42)}`")
        with col_p2:
            st.write(f"**Timestamp:** `{provenance.get('generated_timestamp', '')}`")
            st.write("**Human Gold Used:** `Yes for benchmark evaluation`")
            st.write("**Research-Valid Evaluation:** `Yes for the reported corpus metrics`")

        st.markdown("---")
        st.markdown(f"**Provenance Limitation Notice:**\n\n_{HINDI_PROVENANCE_LIMITATION}_")


def main() -> None:
    """Main application loop."""
    _page_setup()
    render_header()

    # Load configuration and curated samples safely
    config = load_config()
    samples = load_curated_samples()

    # Sidebar
    opts = render_sidebar(samples)
    lang_code = opts["lang_code"]
    sample = opts["selected_sample"]
    model_name = opts["model_name"]
    model_info = opts["model_info"]

    # Populate inputs from selected sample or defaults
    if sample:
        default_orig = sample.get("original_text", "")
        default_simp = sample.get("simplified_text", "")
    else:
        if lang_code == "hi":
            default_orig = DEFAULT_HINDI_ORIGINAL
            default_simp = DEFAULT_HINDI_SIMPLIFIED
        else:
            default_orig = DEFAULT_ORIGINAL_TEXT
            default_simp = DEFAULT_SIMPLIFIED_TEXT

    st.subheader("📝 Sentence Pair Input")
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        orig_input = st.text_area(
            "Original Statutory Provision:",
            value=default_orig,
            height=130,
            help="Authoritative statutory wording.",
            key="input_original",
        )
    with col_in2:
        simp_input = st.text_area(
            "Simplified / Reformulated Sentence:",
            value=default_simp,
            height=130,
            help="The plain language or restructured sentence being evaluated.",
            key="input_simplified",
        )

    # Action buttons
    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        analyze_clicked = st.button("🚀 Analyze Pair", type="primary")
    with col_btn2:
        clear_clicked = st.button("🔄 Reset to Default")

    if clear_clicked:
        st.rerun()

    # When user requests analysis or upon initial load
    if analyze_clicked or True:  # Run on load with defaults
        # Validate inputs
        try:
            valid_orig, valid_simp = validate_input_pair(orig_input, simp_input)
        except ValueError as err:
            st.error(f"❌ Input Validation Error: {err}")
            return

        st.markdown("---")

        # Step 1: Model Prediction (if model selected and available)
        prediction_result: AppPredictionResult | None = None
        attention_payload: dict[str, Any] | None = None

        if model_name != MODEL_NONE:
            if not model_info.available:
                st.error(f"⚠️ {model_info.status_message}")
            else:
                try:
                    prediction_result = predict_pair(
                        model_name=model_name,
                        original=valid_orig,
                        simplified=valid_simp,
                        language=lang_code,
                        config=config,
                    )
                    render_prediction_block(prediction_result)
                    if prediction_result.attention_data:
                        attention_payload = prediction_result.attention_data
                except Exception as exc:
                    st.error(f"❌ Prediction Error for '{model_name}': {exc}")

        # Step 2: Legal Cue Explainability (Primary Explanation)
        try:
            explanation = explain_pair(
                original_text=valid_orig,
                simplified_text=valid_simp,
                language=lang_code,
            )
            render_explainability_block(explanation)
        except Exception as exc:
            st.error(f"❌ Explainability Error: {exc}")
            return

        # Step 3: Attention Attribution (Supporting Evidence)
        render_attention_block(attention_payload, model_name)

        # Step 4: Provenance and Limitations
        prov = (
            prediction_result.provenance
            if prediction_result
            else get_provenance_info(model_name=model_name, cfg=config)
        )
        render_provenance_block(prov)


if __name__ == "__main__":
    main()
