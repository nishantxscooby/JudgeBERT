"""Public, gold-gated evaluation entry point and command-line interface."""

from __future__ import annotations

import argparse
import math
import os
import sys
import uuid
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

from src.data import schema
from src.data.load_data import assert_gold, load_master
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
from src.evaluation.metrics import stratified_regression_metrics
from src.evaluation.provenance import (
    ValidatedPredictionArtifact,
    load_evaluation_result,
    load_prediction_artifact,
    pair_id_sha256,
)
from src.models.provenance import (
    dataframe_sha256,
    ensure_output_directory,
    file_sha256,
    source_tree_sha256,
)
from src.models.train_judgebert import (
    FINAL_TRAINING_INPUT_COLUMNS,
    final_training_input_projection,
)
from src.utils.config import ConfigError, config_fingerprint, load_config
from src.utils.io import git_revision, provenance, write_json

_METRICS_FILENAME = "metrics.json"
_PROVENANCE_FILENAME = "metrics.metadata.json"
_JUDGEBERT_REQUIRED_INPUT_COLUMNS = (
    "pair_id",
    "original_text",
    "simplified_text",
)
_JUDGEBERT_OPTIONAL_INPUT_COLUMNS = (
    "language",
    "legal_domain",
    "source_group",
    "split",
)
_CANONICAL_GOLD_COLUMNS = (
    "pair_id",
    "original_text",
    "simplified_text",
    "human_lmp_score",
    "decision",
    "annotation_status",
    "split",
    "source_group",
    "language",
    "legal_domain",
)


def _evaluation_settings(cfg: dict[str, Any]) -> tuple[list[str], bool, int, float]:
    evaluation = cfg.get("evaluation") or {}
    metrics = list(
        evaluation.get("metrics")
        or ["pearson", "spearman", "rmse", "mae", "r2"]
    )
    bootstrap = evaluation.get("bootstrap") or {}
    enabled = bootstrap.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("evaluation.bootstrap.enabled must be a boolean")
    n_resamples = bootstrap.get("n_resamples", 1000)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("evaluation.bootstrap.n_resamples must be a positive integer")
    confidence = bootstrap.get("confidence", 0.95)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError(
            "evaluation.bootstrap.confidence must be a finite number between 0 and 1"
        )
    return metrics, enabled, n_resamples, float(confidence)


def _canonical_gold_projection(
    gold: Any, *, pair_ids: Sequence[str] | None = None
) -> Any:
    """Return the stable value-bearing human-gold projection used for hashing."""
    missing = sorted(set(_CANONICAL_GOLD_COLUMNS) - set(gold.columns))
    if missing:
        raise ValueError(
            "Authoritative canonical gold is missing provenance columns: "
            f"{missing}"
        )
    work = gold.copy().reset_index(drop=True)
    raw_ids = work["pair_id"]
    normalized_ids = raw_ids.astype("string").str.strip()
    if (
        raw_ids.isna().any()
        or normalized_ids.eq("").fillna(True).any()
        or normalized_ids.duplicated().any()
    ):
        raise ValueError("Authoritative canonical gold pair IDs must be unique and nonblank")
    work["pair_id"] = normalized_ids.astype(str)
    if pair_ids is not None:
        requested = list(pair_ids)
        indexed = work.set_index("pair_id", drop=False)
        missing_ids = [pair_id for pair_id in requested if pair_id not in indexed.index]
        if missing_ids:
            raise ValueError(
                "Evaluated pair IDs are absent from authoritative canonical gold: "
                f"{missing_ids[:10]}"
            )
        work = indexed.loc[requested].reset_index(drop=True)
    return work.loc[:, _CANONICAL_GOLD_COLUMNS].copy().reset_index(drop=True)


def _file_identity(path: Path) -> tuple[int, int]:
    """Return the filesystem identity used to prove publication ownership."""
    status = path.stat()
    return status.st_dev, status.st_ino


def _unlink_if_identity(path: Path, identity: tuple[int, int]) -> None:
    """Remove a path only while it still names an artifact owned by this run."""
    try:
        if _file_identity(path) == identity:
            path.unlink()
    except FileNotFoundError:
        pass


def _atomic_write_json(
    value: Any,
    destination: Path,
    *,
    ownership: dict[Path, tuple[int, int]] | None = None,
) -> None:
    """Publish JSON without ever replacing an existing destination."""
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        write_json(value, temporary)
        identity = _file_identity(temporary)
        if ownership is not None:
            ownership[destination] = identity
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite existing evaluation output {destination}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def _validate_artifact_run(
    artifact: ValidatedPredictionArtifact,
    *,
    expected_config_fingerprint: str,
    expected_seed: int,
) -> None:
    artifact_provenance = artifact.metadata["provenance"]
    if artifact_provenance["config_fingerprint"] != expected_config_fingerprint:
        raise ValueError(
            f"Prediction artifact for {artifact.model_name} was produced with a "
            "different effective config fingerprint"
        )
    if artifact.metadata["seed"] != expected_seed:
        raise ValueError(
            f"Prediction artifact for {artifact.model_name} has seed "
            f"{artifact.metadata['seed']!r}, expected {expected_seed}"
        )


def _validate_calibration_against_gold(
    artifact: ValidatedPredictionArtifact, aligned: Any
) -> None:
    calibration = artifact.metadata.get("calibration")
    if calibration is None:
        return
    canonical_splits = dict(
        zip(aligned["pair_id"].tolist(), aligned["split"].tolist(), strict=True)
    )
    for record in calibration["fit_records"]:
        pair_id = str(record["pair_id"]).strip()
        if pair_id not in canonical_splits:
            raise ValueError(
                f"calibration fit record has unknown canonical pair_id {pair_id!r}"
            )
        canonical_split = canonical_splits[pair_id]
        if record["split"] != canonical_split or canonical_split == "test":
            raise ValueError(
                f"calibration fit record {pair_id} claims split "
                f"{record['split']!r} but canonical split is {canonical_split!r}"
            )


def _validate_checkpoint_fit_against_gold(
    artifact: ValidatedPredictionArtifact, gold: Any
) -> None:
    """Bind a final checkpoint's fit bytes to authoritative canonical gold."""
    lineage = artifact.checkpoint_fit_lineage
    if lineage is None:
        return
    if lineage.input_data_columns != FINAL_TRAINING_INPUT_COLUMNS:
        raise ValueError(
            "JUDGEBERT checkpoint fit input columns do not match the canonical gold "
            "training projection"
        )

    canonical = gold.copy().reset_index(drop=True)
    raw_ids = canonical["pair_id"]
    canonical_ids = raw_ids.astype("string").str.strip()
    if (
        raw_ids.isna().any()
        or canonical_ids.eq("").fillna(True).any()
        or canonical_ids.duplicated().any()
    ):
        raise ValueError("Authoritative canonical gold pair IDs must be unique and nonblank")
    canonical["pair_id"] = canonical_ids.astype(str)
    canonical = canonical.set_index("pair_id", drop=False)

    train_ids = list(lineage.train_ids)
    validation_ids = list(lineage.validation_ids)
    fit_ids = train_ids + validation_ids
    if (
        not train_ids
        or not validation_ids
        or len(fit_ids) != len(set(fit_ids))
    ):
        raise ValueError(
            "JUDGEBERT checkpoint fit IDs must contain disjoint nonempty train and "
            "validation rows"
        )
    canonical_train_ids = canonical.loc[
        canonical["split"].astype("string").str.strip().eq("train"),
        "pair_id",
    ].tolist()
    canonical_validation_ids = canonical.loc[
        canonical["split"].astype("string").str.strip().eq("validation"),
        "pair_id",
    ].tolist()
    if train_ids != canonical_train_ids or validation_ids != canonical_validation_ids:
        raise ValueError(
            "JUDGEBERT checkpoint fit IDs must equal the complete canonical "
            "train/validation populations in split-canonical order"
        )
    missing = [pair_id for pair_id in fit_ids if pair_id not in canonical.index]
    if missing:
        raise ValueError(
            "JUDGEBERT checkpoint fit IDs are absent from canonical gold: "
            f"{missing[:10]}"
        )

    fit_gold = canonical.loc[fit_ids].reset_index(drop=True)
    expected_splits = ["train"] * len(train_ids) + ["validation"] * len(
        validation_ids
    )
    observed_splits = fit_gold["split"].astype("string").str.strip().astype(str).tolist()
    if observed_splits != expected_splits:
        raise ValueError(
            "JUDGEBERT checkpoint fit split/order does not match canonical gold"
        )
    statuses = fit_gold["annotation_status"].astype("string").str.strip()
    if fit_gold["annotation_status"].isna().any() or not statuses.isin(
        schema.GOLD_STATUSES
    ).all():
        raise ValueError(
            "JUDGEBERT checkpoint fit rows are not all authoritative human gold"
        )

    projection = final_training_input_projection(fit_gold)
    if dataframe_sha256(projection) != lineage.input_data_sha256:
        raise ValueError(
            "JUDGEBERT checkpoint fit bytes do not match authoritative canonical gold"
        )


def _validate_judgebert_input_against_gold(
    artifact: ValidatedPredictionArtifact, gold: Any
) -> None:
    """Recompute label-free inference bytes from authoritative canonical rows."""
    if artifact.model_name != "judgebert":
        return
    raw_columns = artifact.metadata.get("input_data_columns")
    if (
        not isinstance(raw_columns, list)
        or any(not isinstance(column, str) for column in raw_columns)
        or len(raw_columns) != len(set(raw_columns))
    ):
        raise ValueError("JUDGEBERT prediction input_data_columns is malformed")
    columns = tuple(raw_columns)
    expected_columns = _JUDGEBERT_REQUIRED_INPUT_COLUMNS + tuple(
        column
        for column in _JUDGEBERT_OPTIONAL_INPUT_COLUMNS
        if column in columns
    )
    if columns != expected_columns:
        raise ValueError(
            "JUDGEBERT prediction input_data_columns do not describe the supported "
            "label-free canonical projection"
        )
    missing_columns = sorted(set(columns) - set(gold.columns))
    if missing_columns:
        raise ValueError(
            "Authoritative canonical gold is missing JUDGEBERT prediction input "
            f"columns: {missing_columns}"
        )

    canonical = gold.copy().reset_index(drop=True)
    raw_ids = canonical["pair_id"]
    canonical_ids = raw_ids.astype("string").str.strip()
    if (
        raw_ids.isna().any()
        or canonical_ids.eq("").fillna(True).any()
        or canonical_ids.duplicated().any()
    ):
        raise ValueError("Authoritative canonical gold pair IDs must be unique and nonblank")
    canonical["pair_id"] = canonical_ids.astype(str)
    canonical = canonical.set_index("pair_id", drop=False)

    prediction_ids = artifact.frame["pair_id"].astype("string").str.strip().astype(str).tolist()
    missing_ids = [pair_id for pair_id in prediction_ids if pair_id not in canonical.index]
    if missing_ids:
        raise ValueError(
            "JUDGEBERT prediction input IDs are absent from canonical gold: "
            f"{missing_ids[:10]}"
        )
    projection = canonical.loc[prediction_ids, list(columns)].reset_index(drop=True)
    if dataframe_sha256(projection) != artifact.metadata["input_data_sha256"]:
        raise ValueError(
            "JUDGEBERT prediction input bytes do not match authoritative canonical gold"
        )


def _evaluate_model(
    artifact: ValidatedPredictionArtifact,
    aligned: Any,
    target: Any,
    *,
    metrics: list[str],
    bootstrap_enabled: bool,
    n_resamples: int,
    confidence: float,
    seed: int,
    report_decisions: bool,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "artifact": artifact.to_dict(),
        "score_range": [schema.SCORE_MIN, schema.SCORE_MAX],
        "regression": stratified_regression_metrics(
            target,
            group_columns=("language", "legal_domain", "split"),
        ),
        "bootstrap": {},
    }
    if bootstrap_enabled:
        intervals = bootstrap_regression_metrics(
            target["human_lmp_score"],
            target["prediction"],
            target["source_group"],
            metrics=metrics,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        report["bootstrap"] = {
            name: interval.to_dict() for name, interval in intervals.items()
        }

    if report_decisions:
        validation = aligned.loc[aligned["split"] == "validation"].copy()
        if validation.empty:
            raise ValueError(
                "Decision threshold fitting requires nonempty validation gold rows"
            )
        thresholds = fit_decision_thresholds(
            validation,
            score_column="prediction",
            decision_column="decision",
            provenance={
                "prediction_run_identity": artifact.run_identity,
                "prediction_file_sha256": artifact.predictions_sha256,
                "fit_policy": "validation_only",
            },
            score_min=float(schema.SCORE_MIN),
            score_max=float(schema.SCORE_MAX),
        )
        predicted_decisions = apply_decision_thresholds(
            target["prediction"], thresholds
        )
        report["decision"] = {
            "thresholds": thresholds.to_dict(),
            "metrics": decision_metrics(target["decision"], predicted_decisions),
        }
    return report


def evaluate_prediction_files(
    *,
    cfg: dict[str, Any],
    prediction_paths: Sequence[str | Path],
    output_dir: str | Path,
    evaluation_split: str = "test",
    allow_fit_data: bool = False,
) -> dict[str, Any]:
    """Evaluate traceable prediction files against authoritative human gold.

    The authoritative gold seam is intentionally the first operation. A corpus
    with no human annotations therefore fails before prediction IO or output
    directory creation.
    """
    gold = load_master(cfg=cfg, subset="gold")
    assert_gold(gold, "final evaluation")
    if not isinstance(allow_fit_data, bool):
        raise ValueError("allow_fit_data must be a boolean")

    paths = list(prediction_paths)
    if not paths:
        raise ValueError("At least one prediction artifact is required")
    if evaluation_split not in schema.SPLITS:
        raise ValueError(
            f"Unknown evaluation split {evaluation_split!r}; expected {schema.SPLITS}"
        )
    includes_fit_data = evaluation_split in {"train", "validation"}
    if includes_fit_data and not allow_fit_data:
        raise ValueError(
            "Evaluation on train/validation fit data is blocked without explicit permission"
        )

    project = cfg.get("project") or {}
    seed = project.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Effective project seed must be an integer")
    training = cfg.get("training") or {}
    model_seed = training.get("seed", seed)
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise ValueError("Effective model seed must be an integer")
    dataset = cfg.get("dataset") or {}
    configured_range = []
    for field in ("score_min", "score_max"):
        value = dataset.get(field)
        if isinstance(value, bool):
            raise ValueError(f"dataset.{field} must be a number, not a boolean")
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"dataset.{field} must be a finite number")
        configured_range.append(value)
    if configured_range != [schema.SCORE_MIN, schema.SCORE_MAX]:
        raise ValueError(
            f"Effective score range must be [{schema.SCORE_MIN}, {schema.SCORE_MAX}]"
        )

    effective_fingerprint = config_fingerprint(cfg)
    artifacts = [load_prediction_artifact(path) for path in paths]
    model_names = [artifact.model_name for artifact in artifacts]
    duplicates = sorted(
        {name for name in model_names if model_names.count(name) > 1}
    )
    if duplicates:
        raise ValueError(f"Duplicate prediction model identities: {duplicates}")

    metrics, bootstrap_enabled, n_resamples, confidence = _evaluation_settings(cfg)
    report_decisions = (cfg.get("evaluation") or {}).get(
        "report_decision_metrics", True
    )
    if not isinstance(report_decisions, bool):
        raise ValueError("evaluation.report_decision_metrics must be a boolean")
    model_reports: dict[str, Any] = {}
    aligned_models: dict[str, Any] = {}
    for artifact in artifacts:
        _validate_artifact_run(
            artifact,
            expected_config_fingerprint=effective_fingerprint,
            expected_seed=model_seed,
        )
        _validate_checkpoint_fit_against_gold(artifact, gold)
        _validate_judgebert_input_against_gold(artifact, gold)
        aligned = align_predictions(
            gold,
            artifact.frame,
            require_metadata=artifact.model_name != "judgebert",
        )
        _validate_calibration_against_gold(artifact, aligned)
        target = aligned.loc[aligned["split"] == evaluation_split].copy()
        if target.empty:
            raise ValueError(
                f"No human gold rows are available for evaluation split "
                f"{evaluation_split!r}"
            )
        model_reports[artifact.model_name] = _evaluate_model(
            artifact,
            aligned,
            target,
            metrics=metrics,
            bootstrap_enabled=bootstrap_enabled,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
            report_decisions=report_decisions,
        )
        aligned_models[artifact.model_name] = target

    paired_comparisons: dict[str, Any] = {}
    if bootstrap_enabled:
        for model_a, model_b in combinations(model_names, 2):
            target_a = aligned_models[model_a]
            target_b = aligned_models[model_b]
            paired_comparisons[f"{model_a}_minus_{model_b}"] = {
                "model_a": model_a,
                "model_b": model_b,
                "metrics": {
                    metric: paired_metric_delta(
                        target_a["human_lmp_score"],
                        target_a["prediction"],
                        target_b["prediction"],
                        target_a["source_group"],
                        metric=metric,
                        n_resamples=n_resamples,
                        confidence=confidence,
                        seed=seed,
                    ).to_dict()
                    for metric in metrics
                },
            }

    canonical_ids = gold["pair_id"].astype(str).str.strip().tolist()
    evaluated = next(iter(aligned_models.values()))
    evaluated_ids = evaluated["pair_id"].tolist()
    canonical_gold_projection = _canonical_gold_projection(gold)
    evaluated_gold_projection = _canonical_gold_projection(
        gold, pair_ids=evaluated_ids
    )
    canonical_gold_sha256 = dataframe_sha256(canonical_gold_projection)
    evaluated_gold_sha256 = dataframe_sha256(evaluated_gold_projection)
    evaluation_provenance = provenance(cfg, artifact="final_evaluation")
    evaluation_provenance["evaluation_code_sha256"] = source_tree_sha256()
    evaluation_provenance["git_revision"] = git_revision()
    result: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_split": evaluation_split,
        "allow_fit_data": allow_fit_data,
        "includes_fit_data": includes_fit_data,
        "fit_data_evaluation_explicitly_allowed": bool(
            includes_fit_data and allow_fit_data
        ),
        "report_decision_metrics": report_decisions,
        "score_range": [schema.SCORE_MIN, schema.SCORE_MAX],
        "canonical_gold": {
            "n_rows": len(gold),
            "pair_id_sha256": pair_id_sha256(canonical_ids),
            "columns": list(_CANONICAL_GOLD_COLUMNS),
            "sha256": canonical_gold_sha256,
        },
        "evaluated_gold": {
            "n_rows": len(evaluated_ids),
            "pair_ids": evaluated_ids,
            "pair_id_sha256": pair_id_sha256(evaluated_ids),
            "columns": list(_CANONICAL_GOLD_COLUMNS),
            "sha256": evaluated_gold_sha256,
        },
        "label_policy": {
            "score": "human_lmp_score",
            "decision": "canonical_human_gold_decision",
            "accepted_annotation_statuses": sorted(
                dataset.get("gold_status_values") or schema.GOLD_STATUSES
            ),
            "draft_labels_used": False,
        },
        "provenance": evaluation_provenance,
        "bootstrap_policy": {
            "enabled": bootstrap_enabled,
            "unit": "source_group_cluster",
            "paired_models_use_identical_resamples": True,
            "n_resamples": n_resamples if bootstrap_enabled else 0,
            "confidence": confidence if bootstrap_enabled else None,
            "seed": seed,
        },
        "models": model_reports,
        "paired_comparisons": paired_comparisons,
        "output_linkage": {
            "metrics_file": _METRICS_FILENAME,
            "provenance_file": _PROVENANCE_FILENAME,
        },
    }

    directory = ensure_output_directory(output_dir)
    metrics_path = directory / _METRICS_FILENAME
    provenance_path = directory / _PROVENANCE_FILENAME
    lock_path = directory / ".evaluation.lock"
    try:
        lock_handle = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Evaluation output conflict: another writer holds {lock_path}"
        ) from exc
    lock_status = os.fstat(lock_handle.fileno())
    lock_identity = (lock_status.st_dev, lock_status.st_ino)
    try:
        with lock_handle:
            lock_handle.write(f"pid={os.getpid()}\n")
    except BaseException:
        _unlink_if_identity(lock_path, lock_identity)
        raise

    created_outputs: dict[Path, tuple[int, int]] = {}
    completed = False
    try:
        if metrics_path.exists() or provenance_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing evaluation output in {directory}"
            )
        _atomic_write_json(result, metrics_path, ownership=created_outputs)
        linkage = {
            "schema_version": 1,
            "artifact": "final_evaluation_result",
            "result_file": _METRICS_FILENAME,
            "result_sha256": file_sha256(metrics_path),
            "canonical_pair_id_sha256": result["canonical_gold"][
                "pair_id_sha256"
            ],
            "canonical_gold_sha256": result["canonical_gold"]["sha256"],
            "config_fingerprint": effective_fingerprint,
            "seed": seed,
            "evaluation_split": evaluation_split,
            "model_run_identities": {
                name: report["artifact"]["run_identity"]
                for name, report in model_reports.items()
            },
        }
        _atomic_write_json(linkage, provenance_path, ownership=created_outputs)
        loaded = load_evaluation_result(metrics_path)
        completed = True
        return loaded
    finally:
        if not completed:
            for created_path, identity in reversed(tuple(created_outputs.items())):
                _unlink_if_identity(created_path, identity)
        _unlink_if_identity(lock_path, lock_identity)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate traceable model predictions against human gold labels."
    )
    parser.add_argument("--config", required=True, help="Effective YAML config path")
    parser.add_argument(
        "--predictions",
        required=True,
        nargs="+",
        help="One or more prediction CSV artifacts",
    )
    parser.add_argument("--output-dir", required=True, help="Evaluation output directory")
    parser.add_argument(
        "--split",
        choices=list(schema.SPLITS),
        default="test",
        help="Split to evaluate (default: test)",
    )
    parser.add_argument(
        "--allow-fit-data",
        action="store_true",
        help="Explicitly permit reporting on train or validation rows",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the final evaluator CLI, returning a process exit code."""
    args = _build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=args.predictions,
            output_dir=args.output_dir,
            evaluation_split=args.split,
            allow_fit_data=args.allow_fit_data,
        )
    except (ConfigError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Evaluation metrics written to {Path(args.output_dir) / _METRICS_FILENAME}")
    return 0


__all__ = ["evaluate_prediction_files", "main"]


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
