"""Experiment 3 — Held-Out Domain Generalisation.

Evaluates how effectively models trained/calibrated on primary domains
(Contract and Sale of Goods) generalize to a completely held-out legal domain
(Cyber Law / Information Technology Act, 2000).

Configuration:
- Training domains: contract, sale_of_goods
- Held-out test domain: cyber

Integrity & Anti-Leakage Constraints:
- The held-out domain must never appear in training/calibration data.
- Source groups are strictly isolated; no source group may bridge training and held-out test data.
- Final mode requires authoritative human gold annotations on the held-out domain.
- Smoke test mode provides complete dry-run validation using synthetic data.
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
from src.evaluation.metrics import stratified_regression_metrics
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

EXPERIMENT_NAME = "Experiment 3 — Held-Out Domain Generalisation"
MODELS = ("tfidf", "sbert", "judgebert")


def validate_domain_configuration(
    train_domains: Sequence[str],
    held_out_domain: str,
    configured_domains: Sequence[str],
) -> None:
    """Strictly assert valid domain partition without contamination."""
    if not held_out_domain or not isinstance(held_out_domain, str):
        raise ValueError("Held-out domain must be a non-empty string.")

    if not train_domains:
        raise ValueError("Training domains list must not be empty.")

    if held_out_domain in train_domains:
        raise ValueError(
            f"Domain contamination: Held-out domain {held_out_domain!r} "
            f"must not appear in training domains {list(train_domains)}."
        )

    if held_out_domain not in configured_domains:
        raise ValueError(
            f"Held-out domain {held_out_domain!r} is absent from configured "
            f"dataset domains: {list(configured_domains)}."
        )

    invalid_train = set(train_domains) - set(configured_domains)
    if invalid_train:
        raise ValueError(
            f"Unknown training domain(s): {sorted(invalid_train)}; "
            f"configured: {list(configured_domains)}."
        )


def _generate_generalization_report(
    held_out_domain: str,
    train_domains: Sequence[str],
    metrics_by_model: Mapping[str, Any],
    is_smoke: bool = False,
) -> str:
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        f"- **Training Domains:** `{', '.join(train_domains)}`",
        f"- **Held-Out Test Domain:** `{held_out_domain}`",
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
        f"## Zero-Shot Performance on Held-Out Domain: {held_out_domain.title()}",
        "",
        "| Model | Pearson r | Spearman ρ | RMSE | MAE | R² |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])

    for model, m in metrics_by_model.items():
        overall = m.get("overall", {})
        p = overall.get("pearson", float("nan"))
        s = overall.get("spearman", float("nan"))
        rmse = overall.get("rmse", float("nan"))
        mae = overall.get("mae", float("nan"))
        r2 = overall.get("r2", float("nan"))
        lines.append(f"| **{model.upper()}** | {p:.4f} | {s:.4f} | {rmse:.4f} | {mae:.4f} | {r2:.4f} |")

    return "\n".join(lines) + "\n"


def run_experiment3(
    cfg: Mapping[str, Any] | None = None,
    output_dir: str | Path = "results/experiment3",
    *,
    smoke_test: bool = False,
    predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute Experiment 3.

    Args:
        cfg: Project configuration mapping.
        output_dir: Directory where results will be written.
        smoke_test: If True, executes with synthetic mock fixtures.
        predictions_dir: Optional directory to load precomputed prediction artifacts.

    Returns:
        Structured dictionary containing generalization metrics and provenance.
    """
    effective_cfg = dict(cfg) if cfg is not None else load_experiment_config()
    out_path = ensure_output_directory(output_dir)

    dataset_cfg = effective_cfg.get("dataset") or {}
    exp3_cfg = (effective_cfg.get("experiments") or {}).get("experiment3") or {}

    configured_domains = dataset_cfg.get("domains", ["contract", "cyber", "sale_of_goods"])
    held_out_domain = dataset_cfg.get("generalization_domain", "cyber")
    train_domains = exp3_cfg.get("train_domains", ["contract", "sale_of_goods"])

    # Validate domain configuration integrity
    validate_domain_configuration(train_domains, held_out_domain, configured_domains)

    provenance = build_experiment_provenance(
        effective_cfg,
        experiment_name=EXPERIMENT_NAME,
        model_identity="domain_generalization",
        is_smoke=smoke_test,
        extra={
            "held_out_domain": held_out_domain,
            "train_domains": list(train_domains),
        },
    )

    if not smoke_test:
        try:
            gold = load_gold_or_block(effective_cfg, context=EXPERIMENT_NAME)
        except ExperimentBlockedError as exc:
            write_blocked_status(out_path, EXPERIMENT_NAME, str(exc), provenance)
            raise

        assert_no_draft_or_synthetic_leakage(gold, context=EXPERIMENT_NAME)
        validate_split_and_ids(gold)

        # Source group anti-leakage check between train domains and held-out domain
        train_gold = gold[gold["legal_domain"].isin(train_domains)]
        held_out_gold = gold[gold["legal_domain"] == held_out_domain].copy().reset_index(drop=True)

        if held_out_gold.empty:
            raise ExperimentBlockedError(
                f"Blocked: {TODO_HUMAN_ANNOTATION} — Held-out domain {held_out_domain!r} "
                "contains 0 human gold rows."
            )

        if "source_group" in gold.columns:
            train_groups = set(train_gold["source_group"].dropna().unique())
            test_groups = set(held_out_gold["source_group"].dropna().unique())
            group_overlap = train_groups & test_groups
            if group_overlap:
                raise ValueError(
                    f"Anti-leakage violation: Source groups cross domain split: {sorted(group_overlap)}"
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
        held_out_gold = synth_data[synth_data["legal_domain"] == held_out_domain].copy().reset_index(drop=True)
        model_predictions = create_mock_predictions(synth_data, models=MODELS, seed=42)

    # Evaluate models on held-out domain
    metrics_by_model: dict[str, Any] = {}
    aligned_rows = []

    for model in MODELS:
        pred_df = model_predictions[model]
        held_out_pred = pred_df[pred_df["pair_id"].isin(held_out_gold["pair_id"])].copy().reset_index(drop=True)
        aligned = align_predictions(
            canonical=held_out_gold,
            predictions=held_out_pred,
            prediction_column="prediction",
        )
        strat_metrics = stratified_regression_metrics(
            aligned,
            group_columns=("language",),
            truth_column="human_lmp_score",
            prediction_column="prediction",
        )
        metrics_by_model[model] = {
            "overall": strat_metrics.get("overall", {}),
            "stratified": strat_metrics,
        }

        for _, row in aligned.iterrows():
            aligned_rows.append({
                "pair_id": row["pair_id"],
                "language": row["language"],
                "legal_domain": held_out_domain,
                "model": model,
                "human_lmp_score": row["human_lmp_score"],
                "prediction": row["prediction"],
            })

    predictions_table = pd.DataFrame(aligned_rows)

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "held_out_domain": held_out_domain,
        "train_domains": list(train_domains),
        "models_evaluated": list(MODELS),
        "n_samples": len(held_out_gold),
        "metrics": metrics_by_model,
        "provenance": provenance,
    }

    # Write output artifacts atomically
    atomic_write_json(payload, out_path / "domain_generalization_metrics.json")
    atomic_write_json(provenance, out_path / "provenance.json")
    atomic_write_csv(predictions_table, out_path / "domain_generalization_predictions.csv")

    report_md = _generate_generalization_report(
        held_out_domain=held_out_domain,
        train_domains=train_domains,
        metrics_by_model=metrics_by_model,
        is_smoke=smoke_test,
    )
    report_file = out_path / "domain_generalization_report.md"
    temp_report = report_file.with_name(f".{report_file.name}.tmp")
    temp_report.write_text(report_md, encoding="utf-8")
    temp_report.replace(report_file)

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Experiment 3."""
    parser = argparse.ArgumentParser(
        description=f"Run {EXPERIMENT_NAME} (Held-out domain generalisation)."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="results/experiment3", help="Output directory")
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
        run_experiment3(
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
