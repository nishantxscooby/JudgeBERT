"""Experiment 1 — Baselines vs JUDGEBERT.

Compares:
1. TF-IDF + train-calibrated cosine similarity
2. Multilingual Sentence-BERT + train-calibrated cosine similarity
3. JUDGEBERT-inspired MuRIL cross-encoder regression

Integrity guarantees:
- Final evaluation is performed strictly on the test split.
- Training split is strictly for training and calibration.
- Validation split is strictly for hyperparameter/threshold selection.
- Gold labels only: evaluation on test split requires authoritative human gold.
  Unannotated corpus blocks execution with 'Blocked: TODO_HUMAN_ANNOTATION'.
- Smoke test mode provides dry-run orchestration on synthetic fixtures,
  explicitly marked as non-research output.
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
from src.evaluation.bootstrap import (
    bootstrap_regression_metrics,
    paired_metric_delta,
)
from src.evaluation.decision_metrics import (
    apply_decision_thresholds,
    decision_metrics,
    fit_decision_thresholds,
)
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

EXPERIMENT_NAME = "Experiment 1 — Baselines vs JUDGEBERT"
MODELS = ("tfidf", "sbert", "judgebert")


def _generate_markdown_report(
    metrics_report: Mapping[str, Any],
    paired_report: Mapping[str, Any],
    is_smoke: bool = False,
) -> str:
    lines = [
        f"# {EXPERIMENT_NAME}",
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
        "## Overall Test Performance",
        "",
        "| Model | Pearson r | Spearman ρ | RMSE | MAE | R² |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for model_name, m in metrics_report.items():
        overall = m.get("overall", {})
        p = overall.get("pearson", float("nan"))
        s = overall.get("spearman", float("nan"))
        rmse = overall.get("rmse", float("nan"))
        mae = overall.get("mae", float("nan"))
        r2 = overall.get("r2", float("nan"))
        lines.append(f"| **{model_name.upper()}** | {p:.4f} | {s:.4f} | {rmse:.4f} | {mae:.4f} | {r2:.4f} |")

    lines.append("")
    lines.append("## Paired Differences vs JUDGEBERT")
    lines.append("")
    lines.append("| Comparison | Metric | Delta (JB - Baseline) | 95% CI Lower | 95% CI Upper | p-value |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: |")

    for comp_name, deltas in paired_report.items():
        for metric_name, interval in deltas.items():
            diff = interval.get("delta", float("nan"))
            low = interval.get("ci_lower", float("nan"))
            high = interval.get("ci_upper", float("nan"))
            pval = interval.get("p_value", float("nan"))
            lines.append(
                f"| {comp_name} | {metric_name} | {diff:+.4f} | {low:+.4f} | {high:+.4f} | {pval:.4f} |"
            )

    return "\n".join(lines) + "\n"


def run_experiment1(
    cfg: Mapping[str, Any] | None = None,
    output_dir: str | Path = "results/experiment1",
    *,
    smoke_test: bool = False,
    predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute Experiment 1.

    Args:
        cfg: Project configuration mapping.
        output_dir: Directory where results will be written.
        smoke_test: If True, executes with synthetic mock fixtures.
        predictions_dir: Optional directory to load precomputed prediction artifacts.

    Returns:
        Structured dictionary containing metrics, deltas, and provenance.
    """
    effective_cfg = dict(cfg) if cfg is not None else load_experiment_config()
    out_path = ensure_output_directory(output_dir)
    provenance = build_experiment_provenance(
        effective_cfg,
        experiment_name=EXPERIMENT_NAME,
        model_identity="baselines_and_judgebert",
        is_smoke=smoke_test,
    )

    if not smoke_test:
        # Strict gold gating — raises ExperimentBlockedError if human gold is unavailable
        try:
            gold = load_gold_or_block(effective_cfg, context=EXPERIMENT_NAME)
        except ExperimentBlockedError as exc:
            write_blocked_status(out_path, EXPERIMENT_NAME, str(exc), provenance)
            raise

        assert_no_draft_or_synthetic_leakage(gold, context=EXPERIMENT_NAME)
        validate_split_and_ids(gold)
        test_gold = gold[gold["split"] == "test"].copy().reset_index(drop=True)
        val_gold = gold[gold["split"] == "validation"].copy().reset_index(drop=True)

        if test_gold.empty:
            raise ExperimentBlockedError(
                f"Blocked: {TODO_HUMAN_ANNOTATION} — Test split contains 0 human gold rows."
            )

        # In final mode, load predictions from prediction artifacts if available
        # or error out if artifacts are absent
        model_predictions: dict[str, pd.DataFrame] = {}
        pred_base = Path(predictions_dir) if predictions_dir else Path("results")
        for model in MODELS:
            artifact_file = pred_base / f"{model}_predictions.csv"
            if artifact_file.is_file():
                artifact = load_prediction_artifact(artifact_file)
                model_predictions[model] = artifact.frame
            else:
                raise FileNotFoundError(
                    f"Required prediction artifact for {model!r} not found at {artifact_file}. "
                    "Run model inference before evaluating."
                )
    else:
        # Smoke test orchestration with synthetic data
        synth_data = create_synthetic_experiment_fixture(n_per_split=6, seed=42)
        test_gold = synth_data[synth_data["split"] == "test"].copy().reset_index(drop=True)
        val_gold = synth_data[synth_data["split"] == "validation"].copy().reset_index(drop=True)
        model_predictions = create_mock_predictions(synth_data, models=MODELS, seed=42)

    # Evaluate each model on test split
    metrics_by_model: dict[str, Any] = {}
    aligned_tables: dict[str, pd.DataFrame] = {}
    combined_predictions = test_gold[["pair_id", "language", "legal_domain", "human_lmp_score", "decision"]].copy()

    eval_cfg = effective_cfg.get("evaluation") or {}
    bootstrap_cfg = eval_cfg.get("bootstrap") or {}
    n_resamples = bootstrap_cfg.get("n_resamples", 200 if smoke_test else 1000)
    confidence = float(bootstrap_cfg.get("confidence", 0.95))
    seed = int((effective_cfg.get("project") or {}).get("seed", 42))

    for model in MODELS:
        pred_df = model_predictions[model]
        test_pred = pred_df[pred_df["pair_id"].isin(test_gold["pair_id"])].copy().reset_index(drop=True)
        aligned = align_predictions(
            canonical=test_gold,
            predictions=test_pred,
            prediction_column="prediction",
        )
        aligned_tables[model] = aligned
        combined_predictions[f"pred_{model}"] = aligned["prediction"]

        # Stratified regression metrics
        strat_metrics = stratified_regression_metrics(
            aligned,
            group_columns=("language", "legal_domain"),
            truth_column="human_lmp_score",
            prediction_column="prediction",
        )

        # Bootstrap confidence intervals
        boot_intervals = bootstrap_regression_metrics(
            aligned["human_lmp_score"],
            aligned["prediction"],
            aligned["source_group"],
            metrics=["pearson", "spearman", "rmse", "mae", "r2"],
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )

        # Decision metrics
        dec_results: dict[str, Any] = {}
        if not val_gold.empty and "decision" in test_gold.columns:
            val_preds = model_predictions[model]
            val_pred_filtered = val_preds[val_preds["pair_id"].isin(val_gold["pair_id"])].copy().reset_index(drop=True)
            val_aligned = align_predictions(
                canonical=val_gold,
                predictions=val_pred_filtered,
                prediction_column="prediction",
            )
            thresholds = fit_decision_thresholds(
                val_aligned,
                score_column="prediction",
                decision_column="decision",
                score_min=float(schema.SCORE_MIN),
                score_max=float(schema.SCORE_MAX),
            )
            predicted_decisions = apply_decision_thresholds(
                aligned["prediction"], thresholds
            )
            dec_results = {
                "thresholds": thresholds.to_dict(),
                "metrics": decision_metrics(aligned["decision"], predicted_decisions),
            }

        metrics_by_model[model] = {
            "overall": strat_metrics.get("overall", {}),
            "stratified": strat_metrics,
            "bootstrap": {k: v.to_dict() for k, v in boot_intervals.items()},
            "decision": dec_results,
        }

    # Paired metric differences: JUDGEBERT vs baselines
    paired_deltas: dict[str, Any] = {}
    if "judgebert" in aligned_tables:
        jb_aligned = aligned_tables["judgebert"]
        for base in ("tfidf", "sbert"):
            if base in aligned_tables:
                comp_key = f"judgebert_vs_{base}"
                paired_deltas[comp_key] = {}
                base_aligned = aligned_tables[base]
                for metric in ("rmse", "mae", "spearman", "pearson"):
                    delta_int = paired_metric_delta(
                        jb_aligned["human_lmp_score"],
                        jb_aligned["prediction"],
                        base_aligned["prediction"],
                        jb_aligned["source_group"],
                        metric=metric,
                        n_resamples=n_resamples,
                        confidence=confidence,
                        seed=seed,
                    )
                    paired_deltas[comp_key][metric] = delta_int.to_dict()

    result_payload = {
        "experiment_name": EXPERIMENT_NAME,
        "models_evaluated": list(MODELS),
        "split_evaluated": "test",
        "n_samples": len(test_gold),
        "metrics": metrics_by_model,
        "paired_differences": paired_deltas,
        "provenance": provenance,
    }

    # Write output artifacts atomically
    atomic_write_json(result_payload, out_path / "metrics.json")
    atomic_write_json(provenance, out_path / "provenance.json")
    atomic_write_csv(combined_predictions, out_path / "predictions.csv")

    report_md = _generate_markdown_report(metrics_by_model, paired_deltas, is_smoke=smoke_test)
    report_file = out_path / "report.md"
    temp_report = report_file.with_name(f".{report_file.name}.tmp")
    temp_report.write_text(report_md, encoding="utf-8")
    temp_report.replace(report_file)

    return result_payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Experiment 1."""
    parser = argparse.ArgumentParser(
        description=f"Run {EXPERIMENT_NAME} (Baselines vs JUDGEBERT on test split)."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="results/experiment1", help="Output directory")
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
        run_experiment1(
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
