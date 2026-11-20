"""High-level explainability API and command-line interface.

Orchestrates:
1. Legal cue extraction and surface alignment
2. Structured finding generation and plain-language summary synthesis
3. Visual highlight span generation and safe HTML rendering
4. Supporting token-level attention attribution
5. Traceable JSON serialization

Data Integrity Policy:
    Explanations analyze text surface forms and model predictions only.
    They never generate or imply human gold labels, agreement metrics,
    or final empirical research findings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.explainability.attention_attribution import (
    ATTENTION_DISCLAIMER,
    compute_attention_attribution,
    compute_mock_attention_attribution,
)
from src.explainability.cue_alignment import (
    align_legal_cues,
    detect_language,
    get_visual_highlights,
    render_highlighted_html,
)
from src.models.provenance import atomic_write_json, ensure_output_directory
from src.utils.config import load_config
from src.utils.io import provenance

GENERAL_DISCLAIMER = "This is an auditable cue comparison, not legal advice."


def explain_pair(
    original_text: str,
    simplified_text: str,
    pair_id: str | None = None,
    model_prediction: float | None = None,
    model_name: str | None = None,
    attention: Any | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    """Generate a complete structured explanation for a sentence pair.

    Args:
        original_text: The original statutory text.
        simplified_text: The simplified or reformulated legal text.
        pair_id: Optional unique identifier.
        model_prediction: Optional numerical model prediction (1–10).
        model_name: Optional name of the model providing the prediction.
        attention: Optional attention input (tuple of (model, tokenizer),
                   precomputed dict, or 'mock').
        language: Language code ('en' or 'hi'); auto-detected if omitted.

    Returns:
        JSON-serializable dictionary adhering to the structured explanation contract.
    """
    if not isinstance(original_text, str) or not original_text.strip():
        raise ValueError("original_text must be a non-empty string.")
    if not isinstance(simplified_text, str) or not simplified_text.strip():
        raise ValueError("simplified_text must be a non-empty string.")

    lang = language or detect_language(original_text)

    # 1. Primary explanation: surface legal cue alignment
    alignment = align_legal_cues(
        original_text=original_text,
        simplified_text=simplified_text,
        language=lang,
    )

    # 2. Visual highlights and safe HTML rendering
    orig_cues = alignment["cue_alignment"]["original"]
    simp_cues = alignment["cue_alignment"]["simplified"]

    orig_highlights = get_visual_highlights(original_text, orig_cues)
    simp_highlights = get_visual_highlights(simplified_text, simp_cues)

    orig_html = render_highlighted_html(original_text, orig_highlights["spans"])
    simp_html = render_highlighted_html(simplified_text, simp_highlights["spans"])

    # 3. Model prediction metadata (if provided)
    pred_payload = None
    if model_prediction is not None:
        try:
            numeric_score = float(model_prediction)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"model_prediction must be a valid float: {exc}") from exc

        pred_payload = {
            "score": numeric_score,
            "model_name": str(model_name or "unknown"),
            "note": "Model prediction is an automated estimate and not human ground truth.",
        }

    # 4. Attention attribution (supporting evidence only)
    attention_payload: dict[str, Any]
    if isinstance(attention, dict):
        attention_payload = attention
    elif attention == "mock":
        attention_payload = compute_mock_attention_attribution(
            original_text=original_text,
            simplified_text=simplified_text,
        )
    elif isinstance(attention, (tuple, list)) and len(attention) == 2:
        model_obj, tokenizer_obj = attention
        attention_payload = compute_attention_attribution(
            model=model_obj,
            tokenizer=tokenizer_obj,
            original_text=original_text,
            simplified_text=simplified_text,
        )
    else:
        attention_payload = {
            "available": False,
            "reason": "Attention evidence not requested or not provided.",
            "disclaimer": ATTENTION_DISCLAIMER,
            "top_tokens": [],
        }

    # Ensure attention disclaimer is present
    if "disclaimer" not in attention_payload:
        attention_payload["disclaimer"] = ATTENTION_DISCLAIMER

    # 5. Assemble final JSON-serializable explanation
    return {
        "pair_id": pair_id,
        "original_text": original_text,
        "simplified_text": simplified_text,
        "language": lang,
        "cue_alignment": alignment["cue_alignment"],
        "findings": alignment["findings"],
        "summary": alignment["summary"],
        "highlights": {
            "original": orig_highlights,
            "simplified": simp_highlights,
            "original_html": orig_html,
            "simplified_html": simp_html,
        },
        "model_prediction": pred_payload,
        "attention": attention_payload,
        "research_valid": False,
        "human_gold_used": False,
        "disclaimer": GENERAL_DISCLAIMER,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for generating explanations."""
    parser = argparse.ArgumentParser(
        description="Explain legal meaning preservation between sentence pairs."
    )
    parser.add_argument("--original", help="Original statutory text")
    parser.add_argument("--simplified", help="Simplified or reformulated text")
    parser.add_argument("--pair-id", default=None, help="Optional pair ID")
    parser.add_argument("--prediction", type=float, default=None, help="Optional model prediction")
    parser.add_argument("--model-name", default=None, help="Name of model that produced prediction")
    parser.add_argument("--language", choices=["en", "hi"], default=None, help="Language code")
    parser.add_argument("--mock-attention", action="store_true", help="Include mocked attention evidence")
    parser.add_argument("--output", default=None, help="Path to write single JSON explanation")
    parser.add_argument("--input", default=None, help="Batch input file (CSV or JSON)")
    parser.add_argument("--output-dir", default=None, help="Directory for batch outputs")
    args = parser.parse_args(argv)

    try:
        # Single pair CLI path
        if args.original and args.simplified:
            attention_mode = "mock" if args.mock_attention else None
            explanation = explain_pair(
                original_text=args.original,
                simplified_text=args.simplified,
                pair_id=args.pair_id,
                model_prediction=args.prediction,
                model_name=args.model_name,
                attention=attention_mode,
                language=args.language,
            )

            if args.output:
                out_path = Path(args.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(explanation, out_path)
                print(f"Explanation successfully written to {out_path}")
            else:
                print(json.dumps(explanation, indent=2, ensure_ascii=False))
            return 0

        # Batch input file path
        if args.input:
            input_path = Path(args.input)
            if not input_path.is_file():
                raise FileNotFoundError(f"Batch input file not found: {input_path}")

            if input_path.suffix.lower() == ".csv":
                df = pd.read_csv(input_path, encoding="utf-8")
            elif input_path.suffix.lower() == ".json":
                with input_path.open("r", encoding="utf-8") as fh:
                    raw_data = json.load(fh)
                df = pd.DataFrame(raw_data)
            else:
                raise ValueError(f"Unsupported batch file format: {input_path.suffix}")

            for req_col in ("original_text", "simplified_text"):
                if req_col not in df.columns:
                    raise ValueError(f"Batch input is missing required column: {req_col!r}")

            out_dir = ensure_output_directory(args.output_dir or "results/explanations")
            batch_results = []

            for idx, row in df.iterrows():
                p_id = row.get("pair_id", f"PAIR_{idx:04d}")
                pred_val = row.get("prediction", None)
                m_name = row.get("model_name", None)
                lang = row.get("language", None)

                expl = explain_pair(
                    original_text=str(row["original_text"]),
                    simplified_text=str(row["simplified_text"]),
                    pair_id=str(p_id) if pd.notna(p_id) else None,
                    model_prediction=float(pred_val) if pd.notna(pred_val) else None,
                    model_name=str(m_name) if pd.notna(m_name) else None,
                    attention="mock" if args.mock_attention else None,
                    language=str(lang) if pd.notna(lang) else None,
                )
                batch_results.append(expl)

            batch_file = out_dir / "batch_explanations.json"
            batch_payload = {
                "count": len(batch_results),
                "explanations": batch_results,
            }
            atomic_write_json(batch_payload, batch_file)
            print(f"Processed {len(batch_results)} pairs -> {batch_file}")
            return 0

        parser.print_help(sys.stderr)
        return 1

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
