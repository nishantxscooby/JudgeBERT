"""Prediction artifact provenance validation for final evaluation."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import schema
from src.models.calibration import LinearScoreCalibrator

_KNOWN_MODELS = frozenset({"tfidf", "sbert", "judgebert"})
_REGRESSION_METRICS = frozenset({"pearson", "spearman", "rmse", "mae", "r2"})
_GOLD_PROJECTION_COLUMNS = (
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


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pair_id_sha256(pair_ids: Any) -> str:
    """Hash ordered pair IDs using the model-result contract."""
    payload = json.dumps(
        list(pair_ids), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Prediction provenance {field} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"Prediction provenance {field} must be a SHA-256 digest"
        ) from exc
    return value.casefold()


def _parse_json_bytes(payload: bytes, *, path: Path) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON artifact {path}: {exc}") from exc


def _parse_csv_bytes(payload: bytes, *, path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            io.BytesIO(payload),
            encoding="utf-8",
            keep_default_na=False,
            na_values=[""],
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 CSV artifact {path}: {exc}") from exc


def _sidecar_for(path: Path) -> Path:
    candidates = [
        path.with_suffix(".metadata.json"),
        path.with_name(f"{path.stem}_metadata.json"),
    ]
    if path.stem.endswith("_predictions"):
        model_stem = path.stem.removesuffix("_predictions")
        candidates.append(path.with_name(f"{model_stem}_metadata.json"))
    existing = tuple(
        candidate for candidate in dict.fromkeys(candidates) if candidate.is_file()
    )
    if not existing:
        raise FileNotFoundError(
            f"Prediction provenance sidecar not found for exact artifact: {path}"
        )
    if len(existing) > 1:
        raise ValueError(
            f"Ambiguous prediction provenance sidecars for {path}: "
            f"{[str(item) for item in existing]}"
        )
    return existing[0]


def _validated_ids(frame: pd.DataFrame) -> list[str]:
    if "pair_id" not in frame.columns:
        raise ValueError("Prediction artifact is missing required column 'pair_id'")
    values = frame["pair_id"]
    normalized = values.astype("string").str.strip()
    blank = values.isna() | normalized.eq("").fillna(True)
    if blank.any():
        raise ValueError("Prediction artifact contains null or blank pair_id values")
    ids = normalized.astype(str)
    duplicates = ids.duplicated(keep=False)
    if duplicates.any():
        examples = sorted(ids.loc[duplicates].unique())[:5]
        raise ValueError(f"Duplicate pair_id values in prediction artifact: {examples}")
    return ids.tolist()


def _validate_calibration(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        raise ValueError("Calibrated prediction run is missing calibration provenance")
    LinearScoreCalibrator.from_dict(payload)
    return _stable_sha256(dict(payload))


def _validate_loading_policy(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} loading_policy must be a mapping")
    allow_download = value.get("allow_download")
    local_files_only = value.get("local_files_only")
    if (
        not isinstance(allow_download, bool)
        or not isinstance(local_files_only, bool)
        or local_files_only is not (not allow_download)
        or value.get("trust_remote_code") is not False
    ):
        raise ValueError(
            f"{context} loading_policy allow_download and local_files_only must be "
            "boolean opposites and trust_remote_code must be exactly false"
        )
    return value


def _validate_score_range(value: Any) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not np.isfinite(bound)
            for bound in value
        )
    ):
        raise ValueError(
            "Prediction provenance score_range values must be finite non-bool numbers"
        )
    if [float(bound) for bound in value] != [
        float(schema.SCORE_MIN),
        float(schema.SCORE_MAX),
    ]:
        raise ValueError(
            f"Prediction provenance score_range must be "
            f"[{schema.SCORE_MIN}, {schema.SCORE_MAX}]"
        )


def _all_column_values(frame: pd.DataFrame, column: str) -> set[Any]:
    if column not in frame.columns:
        return set()
    return set(frame[column].dropna().tolist())


def _validated_artifact_reference(
    value: Any, *, field: str
) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Prediction provenance {field} artifact reference is missing")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Prediction provenance {field} artifact reference path is invalid")
    path = Path(raw_path).expanduser()
    resolved = path.resolve()
    if not path.is_absolute() or raw_path != str(resolved):
        raise ValueError(
            f"Prediction provenance {field} artifact reference path must be resolved"
        )
    sha256 = _require_sha256(value.get("sha256"), field=f"{field}.sha256")
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Prediction provenance referenced {field} artifact not found: {resolved}"
        )
    return resolved, sha256


@dataclass(frozen=True)
class ValidatedCheckpointFitLineage:
    """Fit identity read from validated checkpoint and manifest artifacts."""

    checkpoint_path: Path
    checkpoint_sha256: str
    manifest_path: Path
    manifest_sha256: str
    input_data_columns: tuple[str, ...]
    input_data_sha256: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedPredictionArtifact:
    """A prediction CSV cryptographically bound to a validated final-run sidecar."""

    path: Path
    sidecar_path: Path
    frame: pd.DataFrame
    metadata: dict[str, Any]
    predictions_sha256: str
    sidecar_sha256: str
    model_name: str
    run_identity: str
    calibration_identity: str | None
    checkpoint_identity: str | None
    checkpoint_fit_lineage: ValidatedCheckpointFitLineage | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "run_identity": self.run_identity,
            "prediction_file": {
                "path": str(self.path),
                "sha256": self.predictions_sha256,
            },
            "provenance_sidecar": {
                "path": str(self.sidecar_path),
                "sha256": self.sidecar_sha256,
            },
            "calibration_identity": self.calibration_identity,
            "checkpoint_identity": self.checkpoint_identity,
            "checkpoint_lineage": (
                {
                    "checkpoint": {
                        "path": str(self.checkpoint_fit_lineage.checkpoint_path),
                        "sha256": self.checkpoint_fit_lineage.checkpoint_sha256,
                    },
                    "training_manifest": {
                        "path": str(self.checkpoint_fit_lineage.manifest_path),
                        "sha256": self.checkpoint_fit_lineage.manifest_sha256,
                    },
                    "input_data_columns": list(
                        self.checkpoint_fit_lineage.input_data_columns
                    ),
                    "input_data_sha256": self.checkpoint_fit_lineage.input_data_sha256,
                    "fit_ids": {
                        "train": list(self.checkpoint_fit_lineage.train_ids),
                        "validation": list(
                            self.checkpoint_fit_lineage.validation_ids
                        ),
                    },
                }
                if self.checkpoint_fit_lineage is not None
                else None
            ),
            "run_provenance": {
                "run_mode": self.metadata["run_mode"],
                "seed": self.metadata["seed"],
                "config_fingerprint": self.metadata["provenance"][
                    "config_fingerprint"
                ],
                **(
                    {
                        "config_sha256": self.metadata[
                            "inference_config_identity"
                        ]["config_sha256"]
                    }
                    if self.model_name == "judgebert"
                    else {}
                ),
                "git_revision": self.metadata["provenance"]["git_revision"],
                "code_sha256": self.metadata["code_sha256"],
            },
            "score_range": [schema.SCORE_MIN, schema.SCORE_MAX],
            "splits": list(self.metadata["splits"]),
        }


def load_prediction_artifact(path: str | Path) -> ValidatedPredictionArtifact:
    """Read and validate a final prediction artifact and its exact sidecar."""
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Prediction artifact not found: {artifact_path}")
    sidecar_path = _sidecar_for(artifact_path)
    artifact_bytes = artifact_path.read_bytes()
    sidecar_bytes = sidecar_path.read_bytes()
    actual_hash = hashlib.sha256(artifact_bytes).hexdigest()
    sidecar_hash = hashlib.sha256(sidecar_bytes).hexdigest()
    frame = _parse_csv_bytes(artifact_bytes, path=artifact_path)
    metadata_raw = _parse_json_bytes(sidecar_bytes, path=sidecar_path)
    if not isinstance(metadata_raw, Mapping):
        raise ValueError("Prediction provenance sidecar root must be a mapping")
    metadata = dict(metadata_raw)
    model_name = metadata.get("model_name")
    if model_name not in _KNOWN_MODELS:
        raise ValueError(
            f"Prediction provenance has unknown model_name {model_name!r}; "
            f"expected {sorted(_KNOWN_MODELS)}"
        )

    recorded_hash = _require_sha256(
        metadata.get("predictions_sha256"), field="predictions_sha256"
    )
    if recorded_hash != actual_hash:
        raise ValueError(
            "Prediction provenance sidecar is stale or tampered: "
            "predictions_sha256 does not match the exact artifact"
        )
    if "output_sha256" in metadata:
        output_hash = _require_sha256(
            metadata.get("output_sha256"), field="output_sha256"
        )
        if output_hash != actual_hash:
            raise ValueError(
                "Prediction provenance output_sha256 does not match the exact artifact"
            )

    pair_ids = _validated_ids(frame)
    raw_n_predictions = metadata.get("n_predictions")
    if isinstance(raw_n_predictions, bool) or not isinstance(raw_n_predictions, int):
        raise ValueError(
            "Prediction provenance n_predictions must be a non-bool integer"
        )
    if raw_n_predictions != len(pair_ids):
        raise ValueError("Prediction provenance n_predictions does not match the artifact")
    expected_pair_digest = _require_sha256(
        metadata.get("pair_id_sha256"), field="pair_id_sha256"
    )
    if expected_pair_digest != pair_id_sha256(pair_ids):
        raise ValueError(
            "Prediction provenance pair_id_sha256 does not match ordered artifact IDs"
        )

    required_columns = {"prediction"}
    if model_name in {"tfidf", "sbert"}:
        required_columns.add("split")
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Prediction artifact is missing columns: {missing_columns}")
    scores = pd.to_numeric(frame["prediction"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    if not np.isfinite(scores).all():
        raise ValueError("Prediction artifact scores must all be finite")
    if ((scores < schema.SCORE_MIN) | (scores > schema.SCORE_MAX)).any():
        raise ValueError(
            f"Prediction artifact scores must be within "
            f"{schema.SCORE_MIN}-{schema.SCORE_MAX}"
        )
    if "split" in frame.columns:
        split_values = frame["split"]
        normalized_splits = split_values.astype("string").str.strip()
        bad_split = (
            split_values.isna()
            | normalized_splits.eq("").fillna(True)
            | ~normalized_splits.isin(schema.SPLITS)
        )
        if bad_split.any():
            raise ValueError("Prediction artifact contains unknown or blank split values")
        normalized_split_values = normalized_splits.astype(str).tolist()
        observed_splits = list(dict.fromkeys(normalized_split_values))
    else:
        normalized_split_values = []
        observed_splits = []
    if metadata.get("splits") != observed_splits:
        raise ValueError("Prediction provenance splits do not match the exact artifact")
    _validate_score_range(metadata.get("score_range"))

    records = metadata.get("records")
    if records is not None:
        expected_records = [
            {
                "pair_id": pair_id,
                **(
                    {"split": normalized_split_values[index]}
                    if normalized_split_values
                    else {}
                ),
            }
            for index, pair_id in enumerate(pair_ids)
        ]
        if records != expected_records:
            raise ValueError("Prediction provenance records do not match the exact artifact")
    listed_ids = metadata.get("pair_ids")
    if listed_ids is not None and listed_ids != pair_ids:
        raise ValueError("Prediction provenance pair_ids do not match the exact artifact")

    provenance = metadata.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Prediction sidecar is missing provenance")
    for field in ("config_fingerprint", "git_revision"):
        value = provenance.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Prediction provenance is missing {field}")
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Prediction provenance seed must be an integer")
    provenance_seed = provenance.get("seed")
    if isinstance(provenance_seed, bool) or not isinstance(provenance_seed, int):
        raise ValueError("Prediction provenance provenance seed must be an integer")
    if provenance_seed != seed:
        raise ValueError("Prediction provenance seed fields do not match")
    code_hash = _require_sha256(metadata.get("code_sha256"), field="code_sha256")

    if metadata.get("synthetic") is not False or metadata.get("smoke_test") is True:
        raise ValueError("Synthetic or smoke prediction artifacts cannot be finally evaluated")
    if metadata.get("research_metrics_reported") is not False:
        raise ValueError(
            "Prediction provenance research_metrics_reported must be exactly false"
        )

    calibration_identity: str | None = None
    checkpoint_identity: str | None = None
    checkpoint_fit_lineage: ValidatedCheckpointFitLineage | None = None
    inference_config_sha256: str | None = None
    run_mode = metadata.get("run_mode")
    input_data_sha256 = _require_sha256(
        metadata.get("input_data_sha256"), field="input_data_sha256"
    )
    if model_name in {"tfidf", "sbert"}:
        if metadata.get("smoke_test") is not False:
            raise ValueError("Final baseline smoke_test must be exactly false")
        if run_mode != "final":
            raise ValueError("Calibrated baseline prediction run_mode must be 'final'")
        if metadata.get("data_source") != "human_gold_corpus":
            raise ValueError("Final baseline is not traceable to the human gold corpus")
        if metadata.get("research_use") != "research":
            raise ValueError("Final baseline provenance must declare research use")
        calibration_identity = _validate_calibration(metadata.get("calibration"))
        if model_name == "sbert":
            _validate_loading_policy(
                metadata.get("loading_policy"), context="Final Sentence-BERT provenance"
            )
            model_identity = metadata.get("model_identity")
            encoder_runtime = metadata.get("encoder_runtime")
            if not isinstance(model_identity, Mapping) or not isinstance(
                model_identity.get("model_name"), str
            ) or not model_identity["model_name"].strip():
                raise ValueError("Sentence-BERT provenance is missing model_identity")
            if (
                not isinstance(encoder_runtime, Mapping)
                or encoder_runtime.get("configured_model_weights_used") is not True
                or encoder_runtime.get("source")
                != "configured_sentence_transformer"
            ):
                raise ValueError(
                    "Final Sentence-BERT provenance must use configured model weights"
                )
    else:
        for field in ("synthetic", "smoke_test", "checkpoint_synthetic"):
            if metadata.get(field) is not False:
                raise ValueError(
                    f"Final JUDGEBERT provenance {field} must be exactly false"
                )
        if run_mode != "inference":
            raise ValueError("JUDGEBERT prediction run_mode must be 'inference'")
        if metadata.get("checkpoint_run_mode") != "final":
            raise ValueError("JUDGEBERT checkpoint provenance must come from a final run")
        if metadata.get("research_use") != "prediction-only":
            raise ValueError("JUDGEBERT inference provenance has invalid research_use")
        checkpoint_sha256 = _require_sha256(
            metadata.get("checkpoint_sha256"), field="checkpoint_sha256"
        )
        checkpoint_manifest_sha256 = _require_sha256(
            metadata.get("checkpoint_manifest_sha256"),
            field="checkpoint_manifest_sha256",
        )
        checkpoint_identity = _stable_sha256(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            }
        )
        model_identity = metadata.get("model_identity")
        if not isinstance(model_identity, str) or not model_identity.strip():
            raise ValueError("JUDGEBERT provenance is missing model_identity")
        loading_policy = _validate_loading_policy(
            metadata.get("loading_policy"), context="Final JUDGEBERT provenance"
        )
        if (
            loading_policy.get("model_source") != "configured_muril"
            or loading_policy.get("tokenizer_source") != "configured_muril"
        ):
            raise ValueError(
                "Final JUDGEBERT provenance must use the configured checkpoint "
                "model and tokenizer"
            )

        checkpoint_path, referenced_checkpoint_sha256 = _validated_artifact_reference(
            metadata.get("checkpoint_artifact"), field="checkpoint"
        )
        manifest_path, referenced_manifest_sha256 = _validated_artifact_reference(
            metadata.get("checkpoint_manifest_artifact"),
            field="checkpoint_manifest",
        )
        from src.models.predict_judgebert import (
            _load_final_checkpoint_manifest,
            load_judgebert_checkpoint,
        )

        checkpoint = load_judgebert_checkpoint(checkpoint_path, map_location="cpu")
        manifest_lineage = _load_final_checkpoint_manifest(manifest_path, checkpoint)
        if (
            checkpoint.sha256 != referenced_checkpoint_sha256
            or checkpoint.sha256 != checkpoint_sha256
        ):
            raise ValueError(
                "Prediction provenance checkpoint artifact SHA-256 does not match "
                "the safely loaded checkpoint bytes"
            )
        if (
            manifest_lineage.sha256 != referenced_manifest_sha256
            or manifest_lineage.sha256 != checkpoint_manifest_sha256
        ):
            raise ValueError(
                "Prediction provenance checkpoint manifest artifact SHA-256 does "
                "not match the validated manifest bytes"
            )
        checkpoint_format_version = metadata.get("checkpoint_format_version")
        if isinstance(checkpoint_format_version, bool) or not isinstance(
            checkpoint_format_version, int
        ):
            raise ValueError(
                "Prediction provenance checkpoint_format_version must be an integer"
            )
        if (
            checkpoint.model_identity != model_identity
            or metadata.get("backbone") != checkpoint.backbone
            or checkpoint_format_version != checkpoint.format_version
            or metadata.get("checkpoint_run_mode") != checkpoint.run_mode
            or metadata.get("checkpoint_synthetic") is not checkpoint.synthetic
        ):
            raise ValueError(
                "Prediction provenance checkpoint identity does not match the "
                "validated checkpoint artifact"
            )
        if (
            provenance.get("config_fingerprint") != checkpoint.config_fingerprint
            or seed != checkpoint.effective_seed
        ):
            raise ValueError(
                "Prediction provenance config or seed does not match the validated "
                "checkpoint artifact"
            )
        inference_config = metadata.get("inference_config_identity")
        if (
            not isinstance(inference_config, Mapping)
            or set(inference_config)
            != {"config_fingerprint", "config_sha256", "effective_seed"}
        ):
            raise ValueError(
                "Prediction provenance inference config identity must contain the "
                "complete checkpoint-bound config identity"
            )
        inference_fingerprint = inference_config.get("config_fingerprint")
        if not isinstance(inference_fingerprint, str) or not inference_fingerprint.strip():
            raise ValueError(
                "Prediction provenance inference config identity fingerprint is invalid"
            )
        inference_config_sha256 = _require_sha256(
            inference_config.get("config_sha256"),
            field="inference config identity config_sha256",
        )
        inference_seed = inference_config.get("effective_seed")
        if isinstance(inference_seed, bool) or not isinstance(inference_seed, int):
            raise ValueError(
                "Prediction provenance inference config identity effective_seed must "
                "be an integer"
            )
        if (
            inference_fingerprint != checkpoint.config_fingerprint
            or inference_config_sha256 != checkpoint.config_sha256
            or inference_seed != checkpoint.effective_seed
        ):
            raise ValueError(
                "Prediction provenance inference config identity does not match the "
                "validated checkpoint artifact"
            )
        if checkpoint.fit is None:
            raise ValueError("Validated final checkpoint is missing fit lineage")
        checkpoint_fit_lineage = ValidatedCheckpointFitLineage(
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint.sha256,
            manifest_path=manifest_lineage.path,
            manifest_sha256=manifest_lineage.sha256,
            input_data_columns=checkpoint.input_data_columns,
            input_data_sha256=checkpoint.input_data_sha256,
            train_ids=tuple(checkpoint.fit["ids_by_split"]["train"]),
            validation_ids=tuple(checkpoint.fit["ids_by_split"]["validation"]),
        )

    for column, expected_value in (
        ("run_mode", run_mode),
        ("data_source", metadata.get("data_source")),
        ("research_use", metadata.get("research_use")),
        ("synthetic", False),
    ):
        values = _all_column_values(frame, column)
        if values and values != {expected_value}:
            raise ValueError(
                f"Prediction artifact {column} values do not match its sidecar"
            )

    run_identity = _stable_sha256(
        {
            "model_name": model_name,
            "model_identity": metadata.get("model_identity", model_name),
            "run_mode": run_mode,
            "seed": seed,
            "config_fingerprint": provenance["config_fingerprint"],
            **(
                {"config_sha256": inference_config_sha256}
                if inference_config_sha256 is not None
                else {}
            ),
            "code_sha256": code_hash,
            "input_data_sha256": input_data_sha256,
            "calibration_identity": calibration_identity,
            "checkpoint_identity": checkpoint_identity,
        }
    )
    return ValidatedPredictionArtifact(
        path=artifact_path,
        sidecar_path=sidecar_path.resolve(),
        frame=frame,
        metadata=metadata,
        predictions_sha256=actual_hash,
        sidecar_sha256=sidecar_hash,
        model_name=model_name,
        run_identity=run_identity,
        calibration_identity=calibration_identity,
        checkpoint_identity=checkpoint_identity,
        checkpoint_fit_lineage=checkpoint_fit_lineage,
    )


def _require_result_boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Evaluation result {field} must be a boolean")
    return value


def _require_result_integer(
    value: Any, *, field: str, minimum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Evaluation result {field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(
            f"Evaluation result {field} must be at least {minimum}"
        )
    return value


def _validate_regression_summary(
    value: Any, *, field: str, expected_rows: int | None = None
) -> int:
    if not isinstance(value, Mapping):
        raise ValueError(f"Evaluation result {field} must be an object")
    expected_fields = {"n_rows", *_REGRESSION_METRICS}
    if set(value) != expected_fields:
        raise ValueError(
            f"Evaluation result {field} must have the exact regression metric schema"
        )
    n_rows = _require_result_integer(
        value.get("n_rows"), field=f"{field}.n_rows", minimum=1
    )
    if expected_rows is not None and n_rows != expected_rows:
        raise ValueError(f"Evaluation result {field}.n_rows does not match evaluated gold")
    for metric in _REGRESSION_METRICS:
        metric_value = value.get(metric)
        if not isinstance(metric_value, float) or not np.isfinite(metric_value):
            raise ValueError(
                f"Evaluation result {field}.{metric} must be a finite float"
            )
    return n_rows


def _validate_regression_report(
    value: Any, *, model_name: str, evaluated_rows: int
) -> None:
    field = f"models.{model_name}.regression"
    expected_fields = {"overall", "by_language", "by_legal_domain", "by_split"}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(
            f"Evaluation result {field} must have the exact stratified schema"
        )
    _validate_regression_summary(
        value["overall"], field=f"{field}.overall", expected_rows=evaluated_rows
    )
    for grouping in ("by_language", "by_legal_domain", "by_split"):
        groups = value[grouping]
        if not isinstance(groups, Mapping) or not groups:
            raise ValueError(f"Evaluation result {field}.{grouping} must be nonempty")
        grouped_rows = 0
        for group_name, summary in groups.items():
            if not isinstance(group_name, str) or not group_name:
                raise ValueError(
                    f"Evaluation result {field}.{grouping} has an invalid group name"
                )
            grouped_rows += _validate_regression_summary(
                summary, field=f"{field}.{grouping}.{group_name}"
            )
        if grouped_rows != evaluated_rows:
            raise ValueError(
                f"Evaluation result {field}.{grouping} row counts do not match "
                "evaluated gold"
            )


def _require_result_float(value: Any, *, field: str) -> float:
    if not isinstance(value, float) or not np.isfinite(value):
        raise ValueError(f"Evaluation result {field} must be a finite float")
    return value


def _validate_decision_metrics(value: Any, *, field: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"accuracy", "macro_f1"}:
        raise ValueError(f"Evaluation result {field} has an invalid metric schema")
    for metric in ("accuracy", "macro_f1"):
        metric_value = _require_result_float(value.get(metric), field=f"{field}.{metric}")
        if not 0.0 <= metric_value <= 1.0:
            raise ValueError(f"Evaluation result {field}.{metric} must be within 0-1")


def _validate_decision_report(value: Any, *, model_name: str) -> None:
    field = f"models.{model_name}.decision"
    if not isinstance(value, Mapping) or set(value) != {"thresholds", "metrics"}:
        raise ValueError(f"Evaluation result {field} must have the exact decision schema")
    _validate_decision_metrics(value["metrics"], field=f"{field}.metrics")

    thresholds = value["thresholds"]
    expected_threshold_fields = {
        "schema_version",
        "threshold_type",
        "partial_min",
        "preserved_min",
        "score_range",
        "fit_split",
        "fit_records",
        "selection_objective",
        "validation_metrics",
        "provenance",
    }
    if not isinstance(thresholds, Mapping) or set(thresholds) != expected_threshold_fields:
        raise ValueError(
            f"Evaluation result {field}.thresholds must have the exact schema"
        )
    schema_version = thresholds.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError(
            f"Evaluation result {field}.thresholds.schema_version must be integer 1"
        )
    if (
        thresholds.get("threshold_type") != "validation_grid_search"
        or thresholds.get("fit_split") != "validation"
        or thresholds.get("selection_objective") != ["macro_f1", "accuracy"]
    ):
        raise ValueError(
            f"Evaluation result {field}.thresholds policy is not validation-only"
        )
    score_range = thresholds.get("score_range")
    if (
        not isinstance(score_range, list)
        or len(score_range) != 2
        or any(not isinstance(bound, float) for bound in score_range)
        or score_range != [float(schema.SCORE_MIN), float(schema.SCORE_MAX)]
    ):
        raise ValueError(
            f"Evaluation result {field}.thresholds score_range is invalid"
        )
    partial_min = _require_result_float(
        thresholds.get("partial_min"), field=f"{field}.thresholds.partial_min"
    )
    preserved_min = _require_result_float(
        thresholds.get("preserved_min"), field=f"{field}.thresholds.preserved_min"
    )
    if not score_range[0] <= partial_min < preserved_min <= score_range[1]:
        raise ValueError(
            f"Evaluation result {field}.thresholds are not strictly ordered"
        )

    fit_records = thresholds.get("fit_records")
    if not isinstance(fit_records, list) or not fit_records:
        raise ValueError(
            f"Evaluation result {field}.thresholds.fit_records must be nonempty"
        )
    fit_ids: set[str] = set()
    for record in fit_records:
        if not isinstance(record, Mapping) or set(record) != {
            "pair_id",
            "source_group",
            "split",
        }:
            raise ValueError(
                f"Evaluation result {field}.thresholds.fit_records are malformed"
            )
        pair_id = record.get("pair_id")
        source_group = record.get("source_group")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or pair_id in fit_ids
            or not isinstance(source_group, str)
            or not source_group
            or record.get("split") != "validation"
        ):
            raise ValueError(
                f"Evaluation result {field}.thresholds.fit_records are invalid"
            )
        fit_ids.add(pair_id)

    _validate_decision_metrics(
        thresholds.get("validation_metrics"),
        field=f"{field}.thresholds.validation_metrics",
    )
    provenance = thresholds.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance)
        != {"prediction_run_identity", "prediction_file_sha256", "fit_policy"}
        or provenance.get("fit_policy") != "validation_only"
    ):
        raise ValueError(
            f"Evaluation result {field}.thresholds provenance is malformed"
        )
    _require_sha256(
        provenance.get("prediction_run_identity"),
        field=f"{field}.thresholds.prediction_run_identity",
    )
    _require_sha256(
        provenance.get("prediction_file_sha256"),
        field=f"{field}.thresholds.prediction_file_sha256",
    )


def _validate_gold_projection_envelope(
    value: Any, *, name: str, require_pair_ids: bool
) -> None:
    label = f"{name} gold projection"
    if not isinstance(value, Mapping):
        raise ValueError(f"Evaluation result {label} must be an object")
    if value.get("columns") != list(_GOLD_PROJECTION_COLUMNS):
        raise ValueError(
            f"Evaluation result {label} columns must match the canonical schema exactly"
        )
    _require_sha256(value.get("sha256"), field=f"{label} sha256")
    pair_digest = _require_sha256(
        value.get("pair_id_sha256"), field=f"{label} pair_id_sha256"
    )
    n_rows = value.get("n_rows")
    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 0:
        raise ValueError(f"Evaluation result {label} n_rows must be a nonnegative integer")
    if not require_pair_ids:
        return
    pair_ids = value.get("pair_ids")
    if (
        not isinstance(pair_ids, list)
        or len(pair_ids) != n_rows
        or any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids)
        or len(pair_ids) != len(set(pair_ids))
    ):
        raise ValueError(
            f"Evaluation result {label} pair_ids must be unique nonempty strings "
            "matching n_rows"
        )
    if pair_id_sha256(pair_ids) != pair_digest:
        raise ValueError(
            f"Evaluation result {label} pair_id_sha256 does not match pair_ids"
        )


def load_evaluation_result(path: str | Path) -> dict[str, Any]:
    """Load an evaluation result with its adjacent provenance envelope."""
    result_path = Path(path).expanduser().resolve()
    if not result_path.is_file():
        raise FileNotFoundError(f"Evaluation result not found: {result_path}")
    sidecar_path = result_path.with_suffix(".metadata.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Evaluation result provenance sidecar not found: {sidecar_path}"
        )
    result_bytes = result_path.read_bytes()
    sidecar_bytes = sidecar_path.read_bytes()
    actual_result_hash = hashlib.sha256(result_bytes).hexdigest()
    # The sidecar has no parent envelope, but its exact bytes are still hashed
    # before parsing so validation never reopens a changed file.
    _sidecar_sha256 = hashlib.sha256(sidecar_bytes).hexdigest()
    result = _parse_json_bytes(result_bytes, path=result_path)
    sidecar = _parse_json_bytes(sidecar_bytes, path=sidecar_path)
    if not isinstance(result, Mapping) or not isinstance(sidecar, Mapping):
        raise ValueError("Evaluation result and provenance roots must be mappings")
    if sidecar.get("artifact") != "final_evaluation_result":
        raise ValueError(
            "Evaluation result sidecar artifact must be 'final_evaluation_result'"
        )
    if sidecar.get("result_file") != result_path.name:
        raise ValueError(
            "Evaluation result provenance does not name the exact result artifact"
        )
    expected_hash = _require_sha256(
        sidecar.get("result_sha256"), field="result_sha256"
    )
    if expected_hash != actual_result_hash:
        raise ValueError(
            "Evaluation result provenance is stale or tampered: "
            "result_sha256 does not match the exact artifact"
        )
    for label, payload in (("result", result), ("sidecar", sidecar)):
        schema_version = payload.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            raise ValueError(
                f"Evaluation {label} schema_version must be the exact integer 1"
            )
    try:
        result_provenance = result["provenance"]
        canonical_gold = result["canonical_gold"]
        evaluated_gold = result["evaluated_gold"]
        model_reports = result["models"]
        output_linkage = result["output_linkage"]
        if not all(
            isinstance(value, Mapping)
            for value in (
                result_provenance,
                canonical_gold,
                evaluated_gold,
                model_reports,
                output_linkage,
            )
        ):
            raise TypeError
        model_run_identities = {
            str(name): report["artifact"]["run_identity"]
            for name, report in model_reports.items()
        }
        result_seed = result_provenance["seed"]
        if isinstance(result_seed, bool) or not isinstance(result_seed, int):
            raise TypeError
        expected_sidecar = {
            "canonical_pair_id_sha256": canonical_gold["pair_id_sha256"],
            "canonical_gold_sha256": canonical_gold["sha256"],
            "config_fingerprint": result_provenance["config_fingerprint"],
            "seed": result_seed,
            "evaluation_split": result["evaluation_split"],
            "model_run_identities": model_run_identities,
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Evaluation result envelope is malformed") from exc
    _validate_gold_projection_envelope(
        canonical_gold, name="canonical", require_pair_ids=False
    )
    _validate_gold_projection_envelope(
        evaluated_gold, name="evaluated", require_pair_ids=True
    )

    evaluation_split = result.get("evaluation_split")
    if not isinstance(evaluation_split, str) or evaluation_split not in schema.SPLITS:
        raise ValueError("Evaluation result evaluation_split is invalid")
    allow_fit_data = _require_result_boolean(
        result.get("allow_fit_data"), field="allow_fit_data"
    )
    includes_fit_data = _require_result_boolean(
        result.get("includes_fit_data"), field="includes_fit_data"
    )
    explicitly_allowed = _require_result_boolean(
        result.get("fit_data_evaluation_explicitly_allowed"),
        field="fit_data_evaluation_explicitly_allowed",
    )
    report_decisions = _require_result_boolean(
        result.get("report_decision_metrics"), field="report_decision_metrics"
    )
    expected_includes_fit_data = evaluation_split in {"train", "validation"}
    if includes_fit_data != expected_includes_fit_data:
        raise ValueError(
            "Evaluation result includes_fit_data does not match evaluation_split"
        )
    if explicitly_allowed != (includes_fit_data and allow_fit_data):
        raise ValueError(
            "Evaluation result fit-data permission fields are semantically inconsistent"
        )
    if includes_fit_data and not allow_fit_data:
        raise ValueError("Evaluation result reports fit data without explicit permission")

    score_range = result.get("score_range")
    if (
        not isinstance(score_range, list)
        or len(score_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in score_range)
        or score_range != [schema.SCORE_MIN, schema.SCORE_MAX]
    ):
        raise ValueError("Evaluation result score_range must be the exact canonical range")

    label_policy = result.get("label_policy")
    if not isinstance(label_policy, Mapping):
        raise ValueError("Evaluation result label_policy must be an object")
    if label_policy.get("score") != "human_lmp_score":
        raise ValueError(
            "Evaluation result label_policy.score must be authoritative human_lmp_score"
        )
    if label_policy.get("decision") != "canonical_human_gold_decision":
        raise ValueError(
            "Evaluation result label_policy.decision must name canonical human gold"
        )
    accepted_statuses = label_policy.get("accepted_annotation_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or not accepted_statuses
        or any(not isinstance(value, str) for value in accepted_statuses)
        or accepted_statuses != sorted(set(accepted_statuses))
        or not set(accepted_statuses) <= schema.GOLD_STATUSES
    ):
        raise ValueError(
            "Evaluation result accepted annotation statuses must be unique sorted "
            "human-gold statuses"
        )
    draft_labels_used = _require_result_boolean(
        label_policy.get("draft_labels_used"), field="label_policy.draft_labels_used"
    )
    if draft_labels_used:
        raise ValueError("Evaluation result must not report draft labels as used")

    bootstrap_policy = result.get("bootstrap_policy")
    if not isinstance(bootstrap_policy, Mapping):
        raise ValueError("Evaluation result bootstrap_policy must be an object")
    if bootstrap_policy.get("unit") != "source_group_cluster":
        raise ValueError(
            "Evaluation result bootstrap_policy.unit must be source_group_cluster"
        )
    bootstrap_enabled = _require_result_boolean(
        bootstrap_policy.get("enabled"), field="bootstrap_policy.enabled"
    )
    paired_resamples = _require_result_boolean(
        bootstrap_policy.get("paired_models_use_identical_resamples"),
        field="bootstrap_policy.paired_models_use_identical_resamples",
    )
    if not paired_resamples:
        raise ValueError("Evaluation result must use identical paired-model resamples")
    bootstrap_resamples = _require_result_integer(
        bootstrap_policy.get("n_resamples"),
        field="bootstrap_policy.n_resamples",
        minimum=0,
    )
    bootstrap_seed = _require_result_integer(
        bootstrap_policy.get("seed"), field="bootstrap_policy.seed"
    )
    if bootstrap_seed != result_seed:
        raise ValueError("Evaluation result bootstrap seed does not match provenance seed")
    bootstrap_confidence = bootstrap_policy.get("confidence")
    if bootstrap_enabled:
        if bootstrap_resamples < 1:
            raise ValueError(
                "Evaluation result enabled bootstrap must use positive n_resamples"
            )
        if (
            not isinstance(bootstrap_confidence, float)
            or not np.isfinite(bootstrap_confidence)
            or not 0.0 < bootstrap_confidence < 1.0
        ):
            raise ValueError(
                "Evaluation result bootstrap confidence must be a finite float between 0 and 1"
            )
    elif bootstrap_resamples != 0 or bootstrap_confidence is not None:
        raise ValueError(
            "Evaluation result disabled bootstrap must record zero resamples and null confidence"
        )

    canonical_rows = _require_result_integer(
        canonical_gold.get("n_rows"), field="canonical gold projection n_rows", minimum=1
    )
    evaluated_rows = _require_result_integer(
        evaluated_gold.get("n_rows"), field="evaluated gold projection n_rows", minimum=1
    )
    if evaluated_rows > canonical_rows:
        raise ValueError(
            "Evaluation result evaluated gold row count exceeds canonical gold row count"
        )
    bootstrap_metric_names: set[str] | None = None
    for model_name, report in model_reports.items():
        if not isinstance(model_name, str) or model_name not in _KNOWN_MODELS:
            raise ValueError(
                f"Evaluation result contains unsupported model identity {model_name!r}"
            )
        if not isinstance(report, Mapping):
            raise ValueError(
                f"Evaluation result model report {model_name!r} must be an object"
            )
        artifact_report = report.get("artifact")
        if (
            not isinstance(artifact_report, Mapping)
            or artifact_report.get("model_name") != model_name
        ):
            raise ValueError(
                f"Evaluation result model key {model_name!r} does not match "
                "artifact.model_name"
            )
        model_score_range = report.get("score_range")
        if (
            not isinstance(model_score_range, list)
            or len(model_score_range) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in model_score_range
            )
            or model_score_range != [schema.SCORE_MIN, schema.SCORE_MAX]
        ):
            raise ValueError(
                f"Evaluation result model report {model_name!r} has an invalid score range"
            )
        _validate_regression_report(
            report.get("regression"),
            model_name=str(model_name),
            evaluated_rows=evaluated_rows,
        )

        model_bootstrap = report.get("bootstrap")
        if not isinstance(model_bootstrap, Mapping):
            raise ValueError(
                f"Evaluation result model report {model_name!r} bootstrap must be an object"
            )
        if bootstrap_enabled and not model_bootstrap:
            raise ValueError(
                f"Evaluation result enabled bootstrap is missing for model {model_name!r}"
            )
        if not bootstrap_enabled and model_bootstrap:
            raise ValueError(
                f"Evaluation result disabled bootstrap retains model {model_name!r} reports"
            )
        if bootstrap_enabled:
            current_metric_names = set(model_bootstrap)
            if (
                not current_metric_names
                or not current_metric_names <= _REGRESSION_METRICS
            ):
                unknown_metrics = sorted(current_metric_names - _REGRESSION_METRICS)
                raise ValueError(
                    "Evaluation result contains unsupported bootstrap metric "
                    f"identities: {unknown_metrics}"
                )
            if bootstrap_metric_names is None:
                bootstrap_metric_names = current_metric_names
            elif current_metric_names != bootstrap_metric_names:
                raise ValueError(
                    "Evaluation result model bootstrap metric sets must match exactly"
                )
            expected_interval_fields = {
                "metric",
                "estimate",
                "lower",
                "upper",
                "confidence",
                "seed",
                "requested_replicates",
                "effective_replicates",
                "n_clusters",
            }
            for metric_name, interval in model_bootstrap.items():
                if (
                    not isinstance(metric_name, str)
                    or not metric_name
                    or not isinstance(interval, Mapping)
                    or set(interval) != expected_interval_fields
                    or interval.get("metric") != metric_name
                ):
                    raise ValueError(
                        f"Evaluation result model {model_name!r} bootstrap metric is malformed"
                    )
                interval_seed = _require_result_integer(
                    interval.get("seed"),
                    field=f"models.{model_name}.bootstrap.{metric_name}.seed",
                )
                requested = _require_result_integer(
                    interval.get("requested_replicates"),
                    field=(
                        f"models.{model_name}.bootstrap.{metric_name}.requested_replicates"
                    ),
                    minimum=1,
                )
                effective = _require_result_integer(
                    interval.get("effective_replicates"),
                    field=(
                        f"models.{model_name}.bootstrap.{metric_name}.effective_replicates"
                    ),
                    minimum=1,
                )
                _require_result_integer(
                    interval.get("n_clusters"),
                    field=f"models.{model_name}.bootstrap.{metric_name}.n_clusters",
                    minimum=1,
                )
                if (
                    interval_seed != bootstrap_seed
                    or requested != bootstrap_resamples
                    or effective != requested
                    or interval.get("confidence") != bootstrap_confidence
                ):
                    raise ValueError(
                        f"Evaluation result model {model_name!r} bootstrap settings "
                        "do not match bootstrap_policy"
                    )
                for numeric_field in ("estimate", "lower", "upper"):
                    numeric_value = interval.get(numeric_field)
                    if (
                        not isinstance(numeric_value, float)
                        or not np.isfinite(numeric_value)
                    ):
                        raise ValueError(
                            f"Evaluation result model {model_name!r} bootstrap "
                            f"{numeric_field} must be finite"
                        )

        decision_report = report.get("decision")
        if report_decisions:
            _validate_decision_report(decision_report, model_name=str(model_name))
        elif decision_report is not None:
            raise ValueError(
                "Evaluation result report_decision_metrics is false but a decision report exists"
            )

    paired_reports = result.get("paired_comparisons")
    if not isinstance(paired_reports, Mapping):
        raise ValueError("Evaluation result paired_comparisons must be an object")
    if not bootstrap_enabled:
        if paired_reports:
            raise ValueError(
                "Evaluation result disabled bootstrap retains paired comparisons"
            )
    else:
        model_names = list(model_reports)
        expected_pairs = {
            f"{model_a}_minus_{model_b}": (model_a, model_b)
            for model_a, model_b in combinations(model_names, 2)
        }
        if set(paired_reports) != set(expected_pairs):
            raise ValueError(
                "Evaluation result paired comparisons do not match the exact model pairs"
            )
        expected_metrics = bootstrap_metric_names or set()
        expected_delta_fields = {
            "metric",
            "delta_definition",
            "delta_estimate",
            "lower",
            "upper",
            "confidence",
            "seed",
            "requested_replicates",
            "effective_replicates",
            "n_clusters",
        }
        for pair_name, (model_a, model_b) in expected_pairs.items():
            comparison = paired_reports[pair_name]
            if (
                not isinstance(comparison, Mapping)
                or set(comparison) != {"model_a", "model_b", "metrics"}
                or comparison.get("model_a") != model_a
                or comparison.get("model_b") != model_b
            ):
                raise ValueError(
                    f"Evaluation result paired comparison {pair_name!r} is malformed"
                )
            paired_metrics = comparison.get("metrics")
            if (
                not isinstance(paired_metrics, Mapping)
                or set(paired_metrics) != expected_metrics
            ):
                raise ValueError(
                    f"Evaluation result paired comparison {pair_name!r} metrics "
                    "do not match model bootstrap metrics"
                )
            for metric_name, interval in paired_metrics.items():
                if (
                    not isinstance(interval, Mapping)
                    or set(interval) != expected_delta_fields
                    or interval.get("metric") != metric_name
                    or interval.get("delta_definition")
                    != "model_a_minus_model_b"
                ):
                    raise ValueError(
                        f"Evaluation result paired comparison {pair_name!r} metric "
                        f"{metric_name!r} is malformed"
                    )
                interval_seed = _require_result_integer(
                    interval.get("seed"),
                    field=f"paired_comparisons.{pair_name}.{metric_name}.seed",
                )
                requested = _require_result_integer(
                    interval.get("requested_replicates"),
                    field=(
                        f"paired_comparisons.{pair_name}.{metric_name}.requested_replicates"
                    ),
                    minimum=1,
                )
                effective = _require_result_integer(
                    interval.get("effective_replicates"),
                    field=(
                        f"paired_comparisons.{pair_name}.{metric_name}.effective_replicates"
                    ),
                    minimum=1,
                )
                _require_result_integer(
                    interval.get("n_clusters"),
                    field=f"paired_comparisons.{pair_name}.{metric_name}.n_clusters",
                    minimum=1,
                )
                if (
                    interval_seed != bootstrap_seed
                    or requested != bootstrap_resamples
                    or effective != requested
                    or interval.get("confidence") != bootstrap_confidence
                ):
                    raise ValueError(
                        f"Evaluation result paired comparison {pair_name!r} settings "
                        "do not match bootstrap_policy"
                    )
                for numeric_field in ("delta_estimate", "lower", "upper"):
                    numeric_value = interval.get(numeric_field)
                    if (
                        not isinstance(numeric_value, float)
                        or not np.isfinite(numeric_value)
                    ):
                        raise ValueError(
                            f"Evaluation result paired comparison {pair_name!r} "
                            f"{numeric_field} must be a finite float"
                        )

    sidecar_seed = sidecar.get("seed")
    if isinstance(sidecar_seed, bool) or not isinstance(sidecar_seed, int):
        raise ValueError(
            "Evaluation result sidecar field 'seed' does not match the result envelope"
        )
    for field, expected_value in expected_sidecar.items():
        if sidecar.get(field) != expected_value:
            raise ValueError(
                f"Evaluation result sidecar field {field!r} does not match "
                "the result envelope"
            )
    if (
        output_linkage.get("metrics_file") != result_path.name
        or output_linkage.get("provenance_file") != sidecar_path.name
    ):
        raise ValueError(
            "Evaluation result output linkage does not match the exact artifacts"
        )
    return dict(result)


def validate_prediction_provenance(path: str | Path) -> ValidatedPredictionArtifact:
    """Compatibility alias for the public artifact validator."""
    return load_prediction_artifact(path)


__all__ = [
    "ValidatedCheckpointFitLineage",
    "ValidatedPredictionArtifact",
    "load_evaluation_result",
    "load_prediction_artifact",
    "pair_id_sha256",
    "validate_prediction_provenance",
]
