"""Experiment 5 — Qualitative Legal Error Analysis.

Samples difficult or high-error prediction cases across the 11 legal error phenomena:
- Obligation/Permission
- Negation
- Condition
- Exception
- Scope
- Date
- Number
- Entity
- Omission
- Hallucination
- Contradiction

Integrity Guarantees:
- Uses authoritative human `error_type` and `human_lmp_score`.
- Non-negotiable: NEVER falls back to or substitutes `draft_error_type` or AI-generated labels.
- If human labels are absent, execution is blocked with 'Blocked: TODO_HUMAN_ANNOTATION'
  and produces an auditable blocked record rather than fabricated examples.
- Smoke test mode provides dry-run orchestration on synthetic cases,
  clearly marked as non-research output.
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

EXPERIMENT_NAME = "Experiment 5 — Qualitative Error Analysis"
MIN_CASES_REQUIRED = 15

CONFIGURED_PHENOMENA = (
    "obligation",
    "permission",
    "negation",
    "exception",
    "condition",
    "scope",
    "date",
    "number",
    "entity",
    "omission",
    "hallucination",
    "contradiction",
)


def _generate_error_markdown_report(
    cases_df: pd.DataFrame,
    phenomena_counts: Mapping[str, int],
    is_smoke: bool = False,
) -> str:
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        f"- **Sampled Cases:** {len(cases_df)} (minimum required: {MIN_CASES_REQUIRED})",
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
        "## Phenomenon Coverage Summary",
        "",
        "| Phenomenon / Error Type | Sampled Count |",
        "| :--- | :---: |",
    ])
    for phen, cnt in phenomena_counts.items():
        lines.append(f"| {phen} | {cnt} |")

    lines.extend([
        "",
        "## Sampled Qualitative Cases (Ranked by Error Magnitude)",
        "",
    ])

    for idx, row in cases_df.iterrows():
        pair_id = row.get("pair_id", "N/A")
        lang = row.get("language", "N/A")
        domain = row.get("legal_domain", "N/A")
        err_type = row.get("error_type", "None")
        gold_s = row.get("human_lmp_score", float("nan"))
        pred_s = row.get("prediction", float("nan"))
        err_delta = row.get("absolute_error", float("nan"))
        orig = row.get("original_text", "")
        simp = row.get("simplified_text", "")

        lines.extend([
            f"### Case {idx + 1}: `{pair_id}` ({lang.upper()} · {domain})",
            f"- **Error Category:** `{err_type}`",
            f"- **Human Gold LMP Score:** `{gold_s}` | **Model Prediction:** `{pred_s:.2f}` | **Absolute Error:** `{err_delta:.2f}`",
            f"- **Original:** *\"{orig}\"*",
            f"- **Simplified:** *\"{simp}\"*",
            "",
        ])

    return "\n".join(lines) + "\n"


def sample_qualitative_cases(
    df: pd.DataFrame,
    n_examples: int = 20,
    min_required: int = MIN_CASES_REQUIRED,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Sample >= 15 difficult or high-error cases stratified across error types.

    Prioritizes cases with large prediction error while maximizing coverage of
    the configured legal phenomena.
    """
    if "error_type" not in df.columns or df["error_type"].dropna().empty:
        raise ValueError("Cannot sample cases: 'error_type' column is absent or empty.")

    work = df.copy()
    if "absolute_error" not in work.columns:
        if "human_lmp_score" in work.columns and "prediction" in work.columns:
            work["absolute_error"] = np.abs(
                work["human_lmp_score"].astype(float) - work["prediction"].astype(float)
            )
        else:
            work["absolute_error"] = 0.0

    # Sort primarily by error magnitude descending
    sorted_df = work.sort_values(by="absolute_error", ascending=False)

    sampled_indices: list[Any] = []
    seen_phenomena: set[str] = set()

    # Step 1: Ensure coverage of distinct error types
    for _, group in sorted_df.groupby("error_type", sort=False):
        top_idx = group.index[0]
        sampled_indices.append(top_idx)
        seen_phenomena.add(str(group.loc[top_idx, "error_type"]))

    # Step 2: Fill remaining quota with highest absolute error cases
    remaining_needed = n_examples - len(sampled_indices)
    if remaining_needed > 0:
        unused = sorted_df.loc[~sorted_df.index.isin(sampled_indices)]
        additional = unused.index[:remaining_needed].tolist()
        sampled_indices.extend(additional)

    sampled_df = work.loc[sampled_indices].sort_values(
        by="absolute_error", ascending=False
    ).reset_index(drop=True)

    if len(sampled_df) < min_required and len(work) >= min_required:
        # If stratified sampling didn't reach min_required, top up with any available rows
        needed = min_required - len(sampled_df)
        remaining = work.loc[~work.index.isin(sampled_df.index)].index[:needed]
        sampled_df = pd.concat([sampled_df, work.loc[remaining]]).reset_index(drop=True)

    phenomena_counts = (
        sampled_df["error_type"].astype(str).value_counts().to_dict()
    )

    return sampled_df, phenomena_counts


def run_experiment5(
    cfg: Mapping[str, Any] | None = None,
    output_dir: str | Path = "results/experiment5",
    *,
    smoke_test: bool = False,
    n_examples: int = 20,
    predictions_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Execute Experiment 5 qualitative error analysis.

    Args:
        cfg: Project configuration mapping.
        output_dir: Directory where results will be written.
        smoke_test: If True, executes with synthetic mock fixtures.
        n_examples: Target number of cases to sample (minimum 15).
        predictions_dir: Optional directory to load precomputed prediction artifacts.

    Returns:
        Structured dictionary containing sampled cases and phenomenon breakdowns.
    """
    effective_cfg = dict(cfg) if cfg is not None else load_experiment_config()
    out_path = ensure_output_directory(output_dir)

    exp5_cfg = (effective_cfg.get("experiments") or {}).get("experiment5") or {}
    target_count = exp5_cfg.get("n_examples", n_examples)
    if target_count < MIN_CASES_REQUIRED:
        target_count = MIN_CASES_REQUIRED

    provenance = build_experiment_provenance(
        effective_cfg,
        experiment_name=EXPERIMENT_NAME,
        model_identity="qualitative_error_analysis",
        is_smoke=smoke_test,
        extra={
            "target_examples": target_count,
            "min_required_examples": MIN_CASES_REQUIRED,
            "target_phenomena": list(CONFIGURED_PHENOMENA),
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

        if "error_type" not in gold.columns or gold["error_type"].dropna().empty:
            raise ExperimentBlockedError(
                f"Blocked: {TODO_HUMAN_ANNOTATION} — Human 'error_type' column has 0 populated values. "
                "Draft error types must never be used as a substitute."
            )

        # In final mode, load JUDGEBERT predictions to evaluate errors
        pred_base = Path(predictions_dir) if predictions_dir else Path("results")
        jb_artifact = pred_base / "judgebert_predictions.csv"
        if jb_artifact.is_file():
            artifact = load_prediction_artifact(jb_artifact)
            pred_df = artifact.frame
        else:
            # Fallback to tfidf if judgebert is not yet trained
            tfidf_artifact = pred_base / "tfidf_predictions.csv"
            if tfidf_artifact.is_file():
                artifact = load_prediction_artifact(tfidf_artifact)
                pred_df = artifact.frame
            else:
                raise FileNotFoundError(
                    "No model prediction artifact found for error analysis. "
                    f"Looked for {jb_artifact} and {tfidf_artifact}."
                )

        test_gold = gold[gold["split"] == "test"].copy().reset_index(drop=True)
        if test_gold.empty:
            raise ExperimentBlockedError(
                f"Blocked: {TODO_HUMAN_ANNOTATION} — Test split contains 0 human gold rows."
            )
        test_pred = pred_df[pred_df["pair_id"].isin(test_gold["pair_id"])].copy().reset_index(drop=True)
        aligned = align_predictions(
            canonical=test_gold,
            predictions=test_pred,
            prediction_column="prediction",
        )
        eval_pool = test_gold.copy().reset_index(drop=True)
        eval_pool["prediction"] = aligned["prediction"].to_numpy()
    else:
        # Synthetic smoke fixture covering multiple error categories
        eval_pool = create_synthetic_experiment_fixture(n_per_split=10, seed=42)
        mock_preds = create_mock_predictions(eval_pool, models=["judgebert"], seed=42)
        eval_pool["prediction"] = mock_preds["judgebert"]["prediction"]

    # Calculate absolute error
    eval_pool["absolute_error"] = np.abs(
        eval_pool["human_lmp_score"].astype(float) - eval_pool["prediction"].astype(float)
    )

    # Sample cases
    sampled_df, phen_counts = sample_qualitative_cases(
        eval_pool,
        n_examples=target_count,
        min_required=MIN_CASES_REQUIRED,
    )

    preserved_columns = [
        "pair_id",
        "language",
        "legal_domain",
        "split",
        "original_text",
        "simplified_text",
        "human_lmp_score",
        "prediction",
        "absolute_error",
        "error_type",
    ]
    if "decision" in sampled_df.columns:
        preserved_columns.append("decision")

    available_cols = [c for c in preserved_columns if c in sampled_df.columns]
    output_cases_df = sampled_df[available_cols].copy()

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "n_sampled_cases": len(output_cases_df),
        "min_required": MIN_CASES_REQUIRED,
        "phenomena_counts": phen_counts,
        "cases": output_cases_df.to_dict(orient="records"),
        "provenance": provenance,
    }

    # Write output artifacts atomically
    atomic_write_json(payload, out_path / "error_analysis_cases.json")
    atomic_write_json(provenance, out_path / "provenance.json")
    atomic_write_csv(output_cases_df, out_path / "error_analysis_cases.csv")

    report_md = _generate_error_markdown_report(output_cases_df, phen_counts, is_smoke=smoke_test)
    report_file = out_path / "error_analysis_report.md"
    temp_report = report_file.with_name(f".{report_file.name}.tmp")
    temp_report.write_text(report_md, encoding="utf-8")
    temp_report.replace(report_file)

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Experiment 5."""
    parser = argparse.ArgumentParser(
        description=f"Run {EXPERIMENT_NAME} (Qualitative Error Analysis)."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="results/experiment5", help="Output directory")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run orchestration smoke test with synthetic fixtures",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=20,
        help="Number of qualitative cases to sample (minimum 15)",
    )
    parser.add_argument(
        "--predictions-dir",
        default=None,
        help="Directory containing model prediction artifacts",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_experiment_config(args.config)
        run_experiment5(
            cfg=cfg,
            output_dir=args.output_dir,
            smoke_test=args.smoke_test,
            n_examples=args.n_examples,
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
