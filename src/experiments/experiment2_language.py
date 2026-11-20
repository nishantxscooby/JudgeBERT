"""Experiment 2 — Cross-lingual Comparison (English vs Hindi).

Evaluates model performance across English ('en') and Hindi ('hi') subsets of
the test split.

Integrity and Provenance Limitations:
    As documented in `reports/source_provenance.md`:
    - English statutory texts are verbatim quotes extracted from official PDFs.
    - Hindi statutory texts are author-rendered due to font corruption in official
      government PDFs (legacy Kruti/APS glyph mapping failures where Devanagari text
      is unextractable).
    - Consequently, the English and Hindi sub-corpora represent different provenance
      categories. Performance differences between English and Hindi conflate a true
      cross-lingual difference with a source-provenance effect.

    Final mode requires authoritative human gold annotations on both languages.
    Smoke mode validates grouping, schema, and cross-lingual delta calculations.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import schema
from src.evaluation.alignment import align_predictions
from src.evaluation.metrics import (
    regression_metrics,
    stratified_regression_metrics,
)
from src.evaluation.provenance import load_prediction_artifact
from src.experiments.common import (
    ExperimentBlockedError,
    assert_no_draft_or_synthetic_leakage,
    build_experiment_provenance,
    create_mock_predictions,
    create_synthetic_experiment_fixture,
    ensure_output_directory,
    load_experiment_config,
    load_gold_or_block,
    validate_split_and_ids,
    write_blocked_status,
)
from src.models.provenance import atomic_write_csv, atomic_write_json
from src.utils.io import TODO_HUMAN_ANNOTATION

EXPERIMENT_NAME = "Experiment 2 — Cross-lingual Comparison (English vs Hindi)"
LANGUAGES = ("en", "hi")
MODELS = ("tfidf", "sbert", "judgebert")

PROVENANCE_LIMITATION_DISCLAIMER = (
    "PROVENANCE LIMITATION (reports/source_provenance.md): Official Hindi statute "
    "PDFs use non-Unicode legacy encodings, making verbatim quote extraction impossible. "
    "All Hindi rows are author-rendered (verbatim_quote=False), whereas English rows are "
    "verbatim statutory quotes (verbatim_quote=True). Cross-lingual performance differences "
    "mix a language effect with a provenance effect."
)


def _generate_language_report(
    en_metrics: Mapping[str, Any],
    hi_metrics: Mapping[str, Any],
    gap_metrics: Mapping[str, Any],
    is_smoke: bool = False,
) -> str:
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        "> [!IMPORTANT]",
        f"> **Provenance Limitation:** {PROVENANCE_LIMITATION_DISCLAIMER}",
        "",
    ]
    if is_smoke:
        lines.extend([
            "> [!WARNING]",
            "> **NON-RESEARCH SMOKE TEST OUTPUT.**",
            "> Generated using synthetic data to verify pipeline orchestration. Do not cite.",
            "",
        ])

    lines.extend([
        "## Cross-Lingual Performance Comparison on Test Split",
        "",
        "| Model | Metric | English (en) | Hindi (hi) | Gap (Hindi - English) |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ])

    for model in MODELS:
        m_en = en_metrics.get(model, {})
        m_hi = hi_metrics.get(model, {})
        gaps = gap_metrics.get(model, {})
        for metric in ("rmse", "mae", "spearman", "pearson", "r2"):
            v_en = m_en.get(metric, float("nan"))
            v_hi = m_hi.get(metric, float("nan"))
            delta = gaps.get(metric, float("nan"))
            lines.append(
                f"| **{model.upper()}** | {metric.upper()} | {v_en:.4f} | {v_hi:.4f} | {delta:+.4f} |"
            )

    return "\n".join(lines) + "\n"


def run_experiment2(
    cfg: Mapping[str, Any] | None = None,
    output_dir: str | Path = "results/experiment2",
    *,
    smoke_test: bool = False,
    predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute Experiment 2.

    Args:
        cfg: Project configuration mapping.
        output_dir: Directory where results will be written.
        smoke_test: If True, executes with synthetic mock fixtures.
        predictions_dir: Optional directory to load precomputed prediction artifacts.

    Returns:
        Structured dictionary containing language-specific metrics and provenance.
    """
    effective_cfg = dict(cfg) if cfg is not None else load_experiment_config()
    out_path = ensure_output_directory(output_dir)
    provenance = build_experiment_provenance(
        effective_cfg,
        experiment_name=EXPERIMENT_NAME,
        model_identity="language_cross_lingual",
        is_smoke=smoke_test,
        extra={"provenance_limitation": PROVENANCE_LIMITATION_DISCLAIMER},
    )

    if not smoke_test:
        try:
            gold = load_gold_or_block(effective_cfg, context=EXPERIMENT_NAME)
        except ExperimentBlockedError as exc:
            write_blocked_status(out_path, EXPERIMENT_NAME, str(exc), provenance)
            raise

        assert_no_draft_or_synthetic_leakage(gold, context=EXPERIMENT_NAME)
        validate_split_and_ids(gold)
        test_gold = gold[gold["split"] == "test"].copy().reset_index(drop=True)

        if test_gold.empty:
            raise ExperimentBlockedError(
                f"Blocked: {TODO_HUMAN_ANNOTATION} — Test split contains 0 human gold rows."
            )

        # Validate language representation
        present_languages = set(test_gold["language"].dropna().unique())
        missing_languages = set(LANGUAGES) - present_languages
        if missing_languages:
            raise ValueError(
                f"Test split is missing required languages {sorted(missing_languages)}; "
                f"present: {sorted(present_languages)}"
            )

        model_predictions: dict[str, pd.DataFrame] = {}
        pred_base = Path(predictions_dir) if predictions_dir else Path("results")
        for model in MODELS:
            artifact_file = pred_base / f"{model}_predictions.csv"
            if artifact_file.is_file():
                artifact = load_prediction_artifact(artifact_file)
                model_predictions[model] = artifact.frame
            else:
                raise FileNotFoundError(
                    f"Required prediction artifact for {model!r} not found at {artifact_file}."
                )
    else:
        synth_data = create_synthetic_experiment_fixture(n_per_split=6, seed=42)
        test_gold = synth_data[synth_data["split"] == "test"].copy().reset_index(drop=True)
        model_predictions = create_mock_predictions(synth_data, models=MODELS, seed=42)

    en_gold = test_gold[test_gold["language"] == "en"].copy().reset_index(drop=True)
    hi_gold = test_gold[test_gold["language"] == "hi"].copy().reset_index(drop=True)

    en_metrics_by_model: dict[str, Any] = {}
    hi_metrics_by_model: dict[str, Any] = {}
    gap_by_model: dict[str, Any] = {}

    prediction_records = []

    for model in MODELS:
        pred_df = model_predictions[model]

        # English evaluation
        en_pred = pred_df[pred_df["pair_id"].isin(en_gold["pair_id"])].copy().reset_index(drop=True)
        en_aligned = align_predictions(
            canonical=en_gold,
            predictions=en_pred,
            prediction_column="prediction",
        )
        en_reg = regression_metrics(
            en_aligned["human_lmp_score"].to_numpy(),
            en_aligned["prediction"].to_numpy(),
        )
        en_metrics_by_model[model] = en_reg

        # Hindi evaluation
        hi_pred = pred_df[pred_df["pair_id"].isin(hi_gold["pair_id"])].copy().reset_index(drop=True)
        hi_aligned = align_predictions(
            canonical=hi_gold,
            predictions=hi_pred,
            prediction_column="prediction",
        )
        hi_reg = regression_metrics(
            hi_aligned["human_lmp_score"].to_numpy(),
            hi_aligned["prediction"].to_numpy(),
        )
        hi_metrics_by_model[model] = hi_reg

        # Compute gaps (Hindi - English)
        gaps = {}
        for k in ("rmse", "mae", "spearman", "pearson", "r2"):
            gaps[k] = float(hi_reg.get(k, 0.0) - en_reg.get(k, 0.0))
        gap_by_model[model] = gaps

        # Collect prediction records
        for _, row in en_aligned.iterrows():
            prediction_records.append({
                "pair_id": row["pair_id"],
                "language": "en",
                "model": model,
                "human_lmp_score": row["human_lmp_score"],
                "prediction": row["prediction"],
            })
        for _, row in hi_aligned.iterrows():
            prediction_records.append({
                "pair_id": row["pair_id"],
                "language": "hi",
                "model": model,
                "human_lmp_score": row["human_lmp_score"],
                "prediction": row["prediction"],
            })

    predictions_table = pd.DataFrame(prediction_records)

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "languages": list(LANGUAGES),
        "models": list(MODELS),
        "split_evaluated": "test",
        "sample_counts": {"en": len(en_gold), "hi": len(hi_gold)},
        "metrics_by_language": {
            "english": en_metrics_by_model,
            "hindi": hi_metrics_by_model,
        },
        "language_gap": gap_by_model,
        "provenance_limitation": PROVENANCE_LIMITATION_DISCLAIMER,
        "provenance": provenance,
    }

    # Write artifacts atomically
    atomic_write_json(payload, out_path / "language_metrics.json")
    atomic_write_json(provenance, out_path / "provenance.json")
    atomic_write_csv(predictions_table, out_path / "language_predictions.csv")

    report_md = _generate_language_report(
        en_metrics_by_model,
        hi_metrics_by_model,
        gap_by_model,
        is_smoke=smoke_test,
    )
    report_file = out_path / "language_report.md"
    temp_report = report_file.with_name(f".{report_file.name}.tmp")
    temp_report.write_text(report_md, encoding="utf-8")
    temp_report.replace(report_file)

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Experiment 2."""
    parser = argparse.ArgumentParser(
        description=f"Run {EXPERIMENT_NAME} (English vs Hindi)."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="results/experiment2", help="Output directory")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run orchestration smoke test with synthetic fixtures",
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Directory containing model prediction artifacts",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_experiment_config(args.config)
        run_experiment2(
            cfg=cfg,
            output_dir=args.output_dir,
            smoke_test=args.smoke_test,
            predictions_dir=args.predictions_dir,
        )
        mode_str = "smoke test" if args.smoke_test else "final"
        print(f"SUCCESS: {EXPERIMENT_NAME} ({mode_str}) completed. Artifacts written to {args.output_dir}")
        return 0
    except ExperimentBlockedError as exc:
        print(f"Blocked: {TODO_HUMAN_ANNOTATION} — {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
