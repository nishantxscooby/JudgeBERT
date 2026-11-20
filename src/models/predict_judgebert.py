"""Safe, label-free inference for JUDGEBERT-inspired checkpoints."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
import yaml

from src.data import schema
from src.models.common import (
    load_model_config,
    select_device,
    setup_model_seed,
    validate_predictions,
)
from src.models.judgebert_model import (
    MODEL_IDENTITY,
    JudgeBertConfig,
    JudgeBertModel,
)
from src.models.provenance import (
    ResultPaths,
    build_result_metadata,
    dataframe_sha256,
    file_sha256,
)
from src.models import train_judgebert as _training
from src.models.train_judgebert import encode_text_pairs
from src.utils import io as io_utils
from src.utils.config import config_fingerprint

MURIL_BACKBONE = "google/muril-base-cased"

_REQUIRED_INPUT_COLUMNS = ("pair_id", "original_text", "simplified_text")
_IDENTIFYING_METADATA_COLUMNS = (
    "pair_id",
    "language",
    "legal_domain",
    "source_group",
    "split",
)


@dataclass(frozen=True)
class JudgeBertCheckpoint:
    """Validated portable checkpoint ready for strict state restoration."""

    path: Path
    format_version: int
    model_identity: str
    backbone: str
    project_config: dict[str, Any]
    settings: JudgeBertConfig
    state_dict: Mapping[str, torch.Tensor]
    sha256: str
    run_mode: str
    synthetic: bool
    fit: dict[str, Any] | None
    config_fingerprint: str
    config_sha256: str
    effective_seed: int
    code_sha256: str
    input_data_columns: tuple[str, ...]
    input_data_sha256: str


@dataclass(frozen=True)
class CheckpointManifestLineage:
    """Validated training manifest bound to one exact final checkpoint."""

    path: Path
    sha256: str
    checkpoint_path: Path
    effective_config_path: Path
    training_predictions_path: Path


def _required_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"JUDGEBERT {field} must be a lowercase SHA-256 digest")
    return value


def _validated_fit_lineage(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} fit lineage must be a mapping")
    if value.get("splits") != ["train", "validation"]:
        raise ValueError(
            f"{context} fit splits must be exactly train and validation"
        )
    ids_raw = value.get("ids_by_split")
    if not isinstance(ids_raw, Mapping) or set(ids_raw) != {"train", "validation"}:
        raise ValueError(f"{context} fit ids_by_split is malformed")
    ids_by_split: dict[str, list[str]] = {}
    seen: set[str] = set()
    for split in ("train", "validation"):
        values = ids_raw.get(split)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{context} fit {split} IDs must be nonempty")
        normalized: list[str] = []
        for value_id in values:
            if isinstance(value_id, bool) or not isinstance(value_id, str):
                raise ValueError(f"{context} fit IDs must be strings")
            pair_id = value_id.strip()
            if not pair_id or pair_id in seen:
                raise ValueError(f"{context} fit IDs must be nonblank and unique")
            seen.add(pair_id)
            normalized.append(pair_id)
        ids_by_split[split] = normalized
    expected_records = [
        {"pair_id": pair_id, "split": split}
        for split in ("train", "validation")
        for pair_id in ids_by_split[split]
    ]
    if value.get("records") != expected_records:
        raise ValueError(
            f"{context} fit records must exactly match train/validation IDs"
        )
    return {
        "splits": ["train", "validation"],
        "ids_by_split": ids_by_split,
        "records": expected_records,
    }


def load_judgebert_checkpoint(
    path: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
) -> JudgeBertCheckpoint:
    """Read and fully validate the version-1 portable checkpoint safely."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"JUDGEBERT checkpoint not found: {checkpoint_path}")
    try:
        checkpoint_bytes = checkpoint_path.read_bytes()
        payload = _training._read_checkpoint(
            io.BytesIO(checkpoint_bytes), map_location=map_location
        )
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(
            f"Malformed or unsafe JUDGEBERT checkpoint {checkpoint_path}: {exc}"
        ) from exc

    format_version = payload.get("format_version")
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version != 1
    ):
        raise ValueError(
            "JUDGEBERT checkpoint format_version must be the exact integer 1"
        )
    if payload.get("model_identity") != MODEL_IDENTITY:
        raise ValueError(
            f"JUDGEBERT checkpoint model_identity must be exactly {MODEL_IDENTITY!r}"
        )
    backbone = payload.get("backbone")
    if backbone != MURIL_BACKBONE:
        raise ValueError(
            f"JUDGEBERT checkpoint backbone must be exactly {MURIL_BACKBONE!r}"
        )
    project_config = payload.get("project_config")
    if not isinstance(project_config, Mapping):
        raise ValueError("JUDGEBERT checkpoint is missing a valid project_config")
    try:
        checkpoint_settings = JudgeBertConfig.from_project_config(project_config)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"JUDGEBERT checkpoint project_config is invalid: {exc}"
        ) from exc
    if checkpoint_settings.backbone != backbone:
        raise ValueError(
            "JUDGEBERT checkpoint backbone disagrees with its project_config"
        )

    if config is not None:
        explicit_settings = JudgeBertConfig.from_project_config(config)
        if explicit_settings != checkpoint_settings:
            raise ValueError(
                "Explicit config is incompatible with the JUDGEBERT checkpoint config"
            )

    state = payload.get("model_state_dict")
    if (
        not isinstance(state, Mapping)
        or not state
        or any(not isinstance(name, str) for name in state)
        or any(not isinstance(value, torch.Tensor) for value in state.values())
    ):
        raise ValueError(
            "JUDGEBERT checkpoint model_state_dict must be a non-empty tensor mapping"
        )
    run_mode = payload.get("run_mode")
    if run_mode not in {"smoke_test", "final"}:
        raise ValueError(
            "JUDGEBERT checkpoint run_mode must be 'smoke_test' or 'final'"
        )
    synthetic = payload.get("synthetic")
    if not isinstance(synthetic, bool):
        raise ValueError("JUDGEBERT checkpoint synthetic marker must be a boolean")
    if (run_mode == "smoke_test") != synthetic:
        raise ValueError(
            "JUDGEBERT checkpoint run_mode and synthetic markers are inconsistent"
        )
    fit: dict[str, Any] | None = None
    if run_mode == "final":
        fit = _validated_fit_lineage(
            payload.get("fit"), context="JUDGEBERT checkpoint"
        )
        if payload.get("fit_ids") != fit["ids_by_split"]:
            raise ValueError("JUDGEBERT checkpoint fit_ids do not match fit lineage")
        if payload.get("fit_splits") != fit["splits"]:
            raise ValueError("JUDGEBERT checkpoint fit_splits do not match fit lineage")
        if payload.get("train_ids") != fit["ids_by_split"]["train"]:
            raise ValueError("JUDGEBERT checkpoint train_ids do not match fit lineage")
        if payload.get("validation_ids") != fit["ids_by_split"]["validation"]:
            raise ValueError(
                "JUDGEBERT checkpoint validation_ids do not match fit lineage"
            )
    expected_config_fingerprint = config_fingerprint(project_config)
    if payload.get("config_fingerprint") != expected_config_fingerprint:
        raise ValueError(
            "JUDGEBERT checkpoint config_fingerprint does not match project_config"
        )
    expected_config_sha256 = _training._config_sha256(project_config)
    config_sha256 = _required_sha256(
        payload.get("config_sha256"), field="checkpoint config_sha256"
    )
    if config_sha256 != expected_config_sha256:
        raise ValueError("JUDGEBERT checkpoint config_sha256 does not match project_config")
    expected_seed = _training._effective_model_seed(project_config)
    effective_seed = payload.get("effective_seed")
    if isinstance(effective_seed, bool) or not isinstance(effective_seed, int):
        raise ValueError("JUDGEBERT checkpoint effective_seed must be an integer")
    if effective_seed != expected_seed:
        raise ValueError("JUDGEBERT checkpoint effective_seed does not match project_config")
    if config is not None:
        explicit_fingerprint = config_fingerprint(config)
        if explicit_fingerprint != expected_config_fingerprint:
            raise ValueError(
                "Explicit config fingerprint does not match the JUDGEBERT checkpoint"
            )
        explicit_config_sha256 = _training._config_sha256(config)
        if explicit_config_sha256 != config_sha256:
            raise ValueError(
                "Explicit config SHA-256 does not match the JUDGEBERT checkpoint"
            )
        explicit_seed = _training._effective_model_seed(config)
        if explicit_seed != effective_seed:
            raise ValueError(
                "Explicit config effective seed does not match the JUDGEBERT checkpoint"
            )
    code_sha256 = _required_sha256(
        payload.get("code_sha256"), field="checkpoint code_sha256"
    )
    raw_input_data_columns = payload.get("input_data_columns")
    if (
        not isinstance(raw_input_data_columns, list)
        or not raw_input_data_columns
        or any(
            not isinstance(column, str) or not column
            for column in raw_input_data_columns
        )
        or len(raw_input_data_columns) != len(set(raw_input_data_columns))
    ):
        raise ValueError("JUDGEBERT checkpoint input_data_columns is malformed")
    input_data_columns = tuple(raw_input_data_columns)
    if (
        run_mode == "final"
        and input_data_columns != _training.FINAL_TRAINING_INPUT_COLUMNS
    ):
        raise ValueError(
            "JUDGEBERT final checkpoint input_data_columns do not match the "
            "authoritative training projection"
        )
    input_data_sha256 = _required_sha256(
        payload.get("input_data_sha256"), field="checkpoint input_data_sha256"
    )
    expected_lineage = {
        "config_fingerprint": expected_config_fingerprint,
        "config_sha256": config_sha256,
        "effective_seed": effective_seed,
        "code_sha256": code_sha256,
        "input_data_columns": list(input_data_columns),
        "input_data_sha256": input_data_sha256,
        **({"fit": fit} if fit is not None else {}),
    }
    if payload.get("lineage") != expected_lineage:
        raise ValueError("JUDGEBERT checkpoint lineage is internally inconsistent")
    if run_mode == "final":
        for field, expected in (
            ("smoke_test", False),
            ("data_source", "human_gold_corpus"),
            ("research_use", "research-training-only"),
            ("research_metrics_reported", False),
        ):
            value = payload.get(field)
            if (isinstance(expected, bool) and value is not expected) or (
                not isinstance(expected, bool) and value != expected
            ):
                raise ValueError(
                    f"JUDGEBERT final checkpoint {field} must be exactly {expected!r}"
                )

    return JudgeBertCheckpoint(
        path=checkpoint_path,
        format_version=format_version,
        model_identity=MODEL_IDENTITY,
        backbone=backbone,
        project_config=dict(project_config),
        settings=checkpoint_settings,
        state_dict=state,
        sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        run_mode=run_mode,
        synthetic=synthetic,
        fit=fit,
        config_fingerprint=expected_config_fingerprint,
        config_sha256=config_sha256,
        effective_seed=effective_seed,
        code_sha256=code_sha256,
        input_data_columns=input_data_columns,
        input_data_sha256=input_data_sha256,
    )


def _load_final_checkpoint_manifest(
    path: Path,
    checkpoint: JudgeBertCheckpoint,
) -> CheckpointManifestLineage:
    """Bind a final checkpoint to its exact training-manifest bytes."""
    if not path.is_file():
        raise FileNotFoundError(f"JUDGEBERT checkpoint manifest not found: {path}")
    manifest_bytes = path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    try:
        manifest_raw = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Malformed JUDGEBERT checkpoint manifest {path}: {exc}") from exc
    if not isinstance(manifest_raw, Mapping):
        raise ValueError("JUDGEBERT checkpoint manifest root must be a mapping")
    manifest = dict(manifest_raw)
    required_fields = {
        "artifacts",
        "backbone",
        "checkpoint_sha256",
        "code_sha256",
        "config_sha256",
        "data_source",
        "effective_config_sha256",
        "effective_seed",
        "fit",
        "fit_ids",
        "fit_splits",
        "hashes",
        "input_data_columns",
        "input_data_sha256",
        "loading_policy",
        "model_identity",
        "predictions_sha256",
        "provenance",
        "research_metrics_reported",
        "research_use",
        "run_mode",
        "smoke_test",
        "synthetic",
    }
    missing_fields = sorted(required_fields - set(manifest))
    if missing_fields:
        raise ValueError(
            "JUDGEBERT checkpoint manifest must be a complete final training "
            f"manifest; missing fields: {missing_fields}"
        )

    for field, expected in (
        ("model_identity", checkpoint.model_identity),
        ("backbone", checkpoint.backbone),
        ("run_mode", checkpoint.run_mode),
    ):
        if manifest.get(field) != expected:
            raise ValueError(
                f"JUDGEBERT checkpoint manifest {field} does not match the checkpoint"
            )
    for field, expected in (
        ("synthetic", checkpoint.synthetic),
        ("smoke_test", checkpoint.run_mode == "smoke_test"),
    ):
        value = manifest.get(field)
        if not isinstance(value, bool) or value is not expected:
            raise ValueError(
                f"JUDGEBERT checkpoint manifest {field} does not match the checkpoint"
            )
    for field, expected in (
        ("data_source", "human_gold_corpus"),
        ("research_use", "research-training-only"),
    ):
        if manifest.get(field) != expected:
            raise ValueError(
                f"JUDGEBERT checkpoint manifest {field} must be exactly {expected!r}"
            )
    if manifest.get("research_metrics_reported") is not False:
        raise ValueError(
            "JUDGEBERT checkpoint manifest research_metrics_reported must be exactly False"
        )
    loading_policy = manifest.get("loading_policy")
    if not isinstance(loading_policy, Mapping):
        raise ValueError("JUDGEBERT checkpoint manifest loading_policy is invalid")
    for field in ("tokenizer_source", "model_source"):
        if loading_policy.get(field) != "configured_muril":
            raise ValueError(
                f"JUDGEBERT checkpoint manifest {field} must be configured_muril"
            )
    allow_download = loading_policy.get("allow_download")
    local_files_only = loading_policy.get("local_files_only")
    if (
        not isinstance(allow_download, bool)
        or not isinstance(local_files_only, bool)
        or local_files_only is allow_download
        or loading_policy.get("trust_remote_code") is not False
    ):
        raise ValueError(
            "JUDGEBERT checkpoint manifest loading_policy is internally inconsistent"
        )
    manifest_fit = _validated_fit_lineage(
        manifest.get("fit"), context="JUDGEBERT checkpoint manifest"
    )
    if manifest_fit != checkpoint.fit:
        raise ValueError(
            "JUDGEBERT checkpoint manifest fit lineage does not match the checkpoint"
        )
    if manifest.get("fit_ids") != manifest_fit["ids_by_split"]:
        raise ValueError("JUDGEBERT checkpoint manifest fit_ids are inconsistent")
    if manifest.get("fit_splits") != manifest_fit["splits"]:
        raise ValueError("JUDGEBERT checkpoint manifest fit_splits are inconsistent")
    if manifest.get("input_data_columns") != list(checkpoint.input_data_columns):
        raise ValueError(
            "JUDGEBERT checkpoint manifest input_data_columns do not match checkpoint lineage"
        )
    training_provenance = manifest.get("provenance")
    if not isinstance(training_provenance, Mapping):
        raise ValueError("JUDGEBERT checkpoint manifest provenance is invalid")
    if training_provenance.get("artifact") != "judgebert_training":
        raise ValueError(
            "JUDGEBERT checkpoint manifest provenance artifact is invalid"
        )
    if training_provenance.get("config_fingerprint") != checkpoint.config_fingerprint:
        raise ValueError(
            "JUDGEBERT checkpoint manifest config_fingerprint does not match checkpoint config"
        )
    provenance_seed = training_provenance.get("seed")
    if (
        isinstance(provenance_seed, bool)
        or not isinstance(provenance_seed, int)
        or provenance_seed != checkpoint.effective_seed
    ):
        raise ValueError(
            "JUDGEBERT checkpoint manifest provenance seed must be an integer "
            "matching checkpoint config"
        )
    effective_seed = manifest.get("effective_seed")
    if (
        isinstance(effective_seed, bool)
        or not isinstance(effective_seed, int)
        or effective_seed != checkpoint.effective_seed
    ):
        raise ValueError(
            "JUDGEBERT checkpoint manifest effective_seed must be an integer "
            "matching checkpoint config"
        )
    hashes = manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("JUDGEBERT checkpoint manifest hashes must be a mapping")
    checkpoint_lineage_hashes = {
        "config_sha256": checkpoint.config_sha256,
        "code_sha256": checkpoint.code_sha256,
        "input_data_sha256": checkpoint.input_data_sha256,
    }
    for field, expected in checkpoint_lineage_hashes.items():
        top_level = _required_sha256(
            manifest.get(field), field=f"checkpoint manifest {field}"
        )
        nested = _required_sha256(
            hashes.get(field), field=f"checkpoint manifest hashes.{field}"
        )
        if top_level != expected or nested != expected:
            raise ValueError(
                f"JUDGEBERT checkpoint manifest {field} does not match checkpoint lineage"
            )

    recorded_hash = _required_sha256(
        manifest.get("checkpoint_sha256"),
        field="checkpoint manifest checkpoint_sha256",
    )
    nested_hash = _required_sha256(
        hashes.get("checkpoint_sha256"),
        field="checkpoint manifest hashes.checkpoint_sha256",
    )
    if recorded_hash != checkpoint.sha256 or nested_hash != checkpoint.sha256:
        raise ValueError(
            "JUDGEBERT checkpoint manifest checkpoint_sha256 does not match "
            "the loaded checkpoint"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "checkpoint",
        "effective_config",
        "predictions",
    }:
        raise ValueError("JUDGEBERT checkpoint manifest artifacts mapping is invalid")
    artifact_paths: dict[str, Path] = {}
    for name, raw_path in artifacts.items():
        if (
            not isinstance(raw_path, str)
            or not raw_path.strip()
            or Path(raw_path).is_absolute()
            or Path(raw_path).name != raw_path
        ):
            raise ValueError(
                f"JUDGEBERT checkpoint manifest artifact path {name!r} is unsafe"
            )
        artifact_paths[name] = path.parent / raw_path
    if artifact_paths["checkpoint"].resolve() != checkpoint.path.resolve():
        raise ValueError(
            "JUDGEBERT checkpoint manifest checkpoint artifact path is inconsistent"
        )

    try:
        effective_config_bytes = artifact_paths["effective_config"].read_bytes()
        prediction_bytes = artifact_paths["predictions"].read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"JUDGEBERT checkpoint manifest artifact not found: {exc.filename}"
        ) from exc
    artifact_payloads = {
        "effective_config_sha256": effective_config_bytes,
        "predictions_sha256": prediction_bytes,
    }
    for field, payload_bytes in artifact_payloads.items():
        expected = hashlib.sha256(payload_bytes).hexdigest()
        top_level = _required_sha256(
            manifest.get(field), field=f"checkpoint manifest {field}"
        )
        nested = _required_sha256(
            hashes.get(field), field=f"checkpoint manifest hashes.{field}"
        )
        if top_level != expected or nested != expected:
            raise ValueError(
                f"JUDGEBERT checkpoint manifest {field} does not match artifact bytes"
            )

    try:
        effective_config_raw = yaml.safe_load(effective_config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(
            "JUDGEBERT checkpoint effective config artifact is malformed"
        ) from exc
    if not isinstance(effective_config_raw, Mapping):
        raise ValueError("JUDGEBERT checkpoint effective config must be a mapping")
    if config_fingerprint(effective_config_raw) != checkpoint.config_fingerprint:
        raise ValueError(
            "JUDGEBERT checkpoint effective config fingerprint does not match checkpoint"
        )
    if _training._config_sha256(effective_config_raw) != checkpoint.config_sha256:
        raise ValueError(
            "JUDGEBERT checkpoint effective config config_sha256 does not match checkpoint"
        )
    if _training._effective_model_seed(effective_config_raw) != checkpoint.effective_seed:
        raise ValueError(
            "JUDGEBERT checkpoint effective config seed does not match checkpoint"
        )

    try:
        training_predictions = pd.read_csv(
            io.BytesIO(prediction_bytes),
            encoding="utf-8",
            keep_default_na=False,
            na_values=[""],
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(
            "JUDGEBERT checkpoint prediction artifact is malformed"
        ) from exc
    if not {"pair_id", "split"}.issubset(training_predictions.columns):
        raise ValueError(
            "JUDGEBERT checkpoint prediction artifact is missing fit identity"
        )
    prediction_records = [
        {"pair_id": str(row.pair_id).strip(), "split": str(row.split).strip()}
        for row in training_predictions[["pair_id", "split"]].itertuples(index=False)
    ]
    if checkpoint.fit is None or prediction_records != checkpoint.fit["records"]:
        raise ValueError(
            "JUDGEBERT checkpoint prediction artifact does not match fit lineage"
        )
    return CheckpointManifestLineage(
        path=path.resolve(),
        sha256=manifest_sha256,
        checkpoint_path=artifact_paths["checkpoint"].resolve(),
        effective_config_path=artifact_paths["effective_config"].resolve(),
        training_predictions_path=artifact_paths["predictions"].resolve(),
    )


def restore_judgebert_model(
    checkpoint: JudgeBertCheckpoint,
    model: JudgeBertModel,
) -> JudgeBertModel:
    """Strictly restore a validated checkpoint into a compatible initialized model."""
    if not isinstance(checkpoint, JudgeBertCheckpoint):
        raise TypeError("checkpoint must be a validated JudgeBertCheckpoint")
    if not isinstance(model, JudgeBertModel):
        raise TypeError("model must be an initialized JudgeBertModel")
    if model.encoder is None or model.regression_head is None:
        raise ValueError(
            "A checkpoint requires a compatible initialized JudgeBertModel"
        )
    if model.settings != checkpoint.settings:
        raise ValueError(
            "Injected JUDGEBERT model config is incompatible with the checkpoint"
        )
    model.load_state_dict(checkpoint.state_dict, strict=True)
    return model


def _validated_prediction_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a defensive copy with validated IDs and exact paired text."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("JUDGEBERT prediction input must be a pandas DataFrame")
    missing = sorted(set(_REQUIRED_INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"JUDGEBERT prediction input is missing columns: {missing}")
    if frame.empty:
        raise ValueError("JUDGEBERT prediction input must contain at least one row")

    work = frame.copy(deep=True).reset_index(drop=True)
    ids = work["pair_id"]
    invalid_ids = ids.map(
        lambda value: isinstance(value, (bool, np.bool_))
        or not pd.api.types.is_scalar(value)
    )
    if invalid_ids.any():
        raise ValueError("JUDGEBERT prediction input contains invalid pair_id values")
    blank_ids = ids.isna() | ids.astype("string").str.strip().eq("").fillna(True)
    if blank_ids.any():
        raise ValueError("JUDGEBERT prediction input contains null or blank pair_id values")
    normalised_ids = ids.astype(str).str.strip()
    duplicates = normalised_ids[normalised_ids.duplicated(keep=False)]
    if not duplicates.empty:
        raise ValueError(
            "Duplicate pair_id value(s) in JUDGEBERT prediction input: "
            f"{sorted(duplicates.unique())[:5]}"
        )
    work["pair_id"] = normalised_ids

    for column in ("original_text", "simplified_text"):
        values = work[column]
        invalid_type = ~values.map(lambda value: isinstance(value, str))
        blank = values.astype("string").str.strip().eq("").fillna(True)
        bad = (values.isna() | invalid_type | blank).fillna(True)
        if bad.any():
            raise ValueError(
                f"JUDGEBERT prediction input {column} contains "
                f"{int(bad.sum())} null, blank, or non-string value(s)"
            )

    if "split" in work.columns:
        splits = work["split"]
        blank = splits.isna() | splits.astype("string").str.strip().eq("").fillna(True)
        normalised_splits = splits.astype("string").str.strip()
        unknown = sorted(set(normalised_splits[~blank].astype(str)) - set(schema.SPLITS))
        if blank.any() or unknown:
            detail = unknown or ["<null-or-blank>"]
            raise ValueError(
                f"Unknown split value(s) in JUDGEBERT prediction input: {detail}"
            )
        work["split"] = normalised_splits.astype(str)
    return work


def _eval_batch_size(cfg: Mapping[str, Any]) -> int:
    training = cfg.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Project config is missing the 'training' mapping")
    value = training.get("eval_batch_size")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("training.eval_batch_size must be a positive integer")
    return value


def predict_judgebert(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    tokenizer: Any,
    model: JudgeBertModel,
    device: str | None = None,
) -> pd.DataFrame:
    """Predict finite 1--10 scores for exact text pairs in input row order."""
    work = _validated_prediction_input(frame)
    settings = JudgeBertConfig.from_project_config(cfg)
    if not isinstance(model, JudgeBertModel):
        raise TypeError("model must be an initialized JudgeBertModel")
    if model.settings != settings:
        raise ValueError("JUDGEBERT model settings do not match the effective config")
    if model.encoder is None or model.regression_head is None:
        raise ValueError("JUDGEBERT model must be initialized before prediction")

    requested_device = device or str((cfg.get("training") or {}).get("device", "auto"))
    resolved_device = select_device(requested_device)
    batch_size = _eval_batch_size(cfg)
    model.to(resolved_device)

    score_batches: list[torch.Tensor] = []
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(work), batch_size):
                part = work.iloc[start : start + batch_size]
                encoded = encode_text_pairs(
                    tokenizer,
                    part["original_text"].tolist(),
                    part["simplified_text"].tolist(),
                    max_length=settings.max_seq_length,
                    padding=True,
                )
                batch = {name: values.to(resolved_device) for name, values in encoded.items()}
                score_batches.append(model.predict_scores(**batch).detach().cpu())
    finally:
        model.train(was_training)

    combined = torch.cat(score_batches).to(torch.float64).numpy()
    checked = validate_predictions(
        pd.DataFrame(
            {"pair_id": work["pair_id"].tolist(), "prediction": combined}
        ),
        work["pair_id"].tolist(),
        clip=False,
    )
    metadata_columns = [
        column for column in _IDENTIFYING_METADATA_COLUMNS if column in work.columns
    ]
    output = work[metadata_columns].copy().reset_index(drop=True)
    output["prediction"] = checked["prediction"].to_numpy(dtype=float)
    return output


def _validate_output_paths(
    output_path: str | Path,
    *,
    checkpoint_path: Path,
    checkpoint_manifest_path: Path | None,
    input_path: Path | None,
    manifest_lineage: CheckpointManifestLineage | None = None,
) -> tuple[Path, Path]:
    destination = Path(output_path)
    if destination.suffix.casefold() != ".csv" or not destination.name:
        raise ValueError("JUDGEBERT --output must be a .csv file path")
    metadata_path = destination.with_suffix(".metadata.json")
    protected: dict[Path, str] = {
        checkpoint_path.resolve(): "checkpoint",
    }
    if checkpoint_manifest_path is not None:
        protected[checkpoint_manifest_path.resolve()] = "training manifest"
    if input_path is not None:
        protected[input_path.resolve()] = "inference input"
    if manifest_lineage is not None:
        protected.update(
            {
                manifest_lineage.path: "training manifest",
                manifest_lineage.checkpoint_path: "checkpoint",
                manifest_lineage.effective_config_path: "effective config",
                manifest_lineage.training_predictions_path: "training predictions",
            }
        )
    for path in (destination, metadata_path):
        resolved = path.resolve()
        protected_name = None
        for protected_path, candidate_name in protected.items():
            same_path = resolved == protected_path.resolve()
            same_file = False
            if not same_path and path.exists() and protected_path.exists():
                try:
                    same_file = os.path.samefile(path, protected_path)
                except OSError:
                    same_file = False
            if same_path or same_file:
                protected_name = candidate_name
                break
        if protected_name is not None:
            qualifier = "manifest-bound " if manifest_lineage is not None else ""
            raise ValueError(
                "JUDGEBERT output paths must not overwrite or alias "
                f"{qualifier}{protected_name} artifacts"
            )
        if path.exists() and path.is_dir():
            raise IsADirectoryError(f"JUDGEBERT output path is a directory: {path}")
    return destination, metadata_path


def _materialize_configured_model(
    model: JudgeBertModel,
    tokenizer: Any,
    frame: pd.DataFrame,
    settings: JudgeBertConfig,
) -> None:
    """Materialize lazy production weights/head using one real input pair."""
    if model.encoder is not None and model.regression_head is not None:
        return
    sample = frame.iloc[:1]
    encoded = encode_text_pairs(
        tokenizer,
        sample["original_text"].tolist(),
        sample["simplified_text"].tolist(),
        max_length=settings.max_seq_length,
        padding=True,
    )
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            model(**encoded)
    finally:
        model.train(was_training)
    if model.encoder is None or model.regression_head is None:
        raise RuntimeError("JUDGEBERT model initialization did not materialize all weights")


def _input_hash_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Project input to paired text plus present identifying metadata only.

    The fixed column order deliberately excludes every target, draft, decision,
    annotation, and error field so inference lineage never hashes labels.
    """
    columns = list(_REQUIRED_INPUT_COLUMNS)
    columns.extend(
        column
        for column in _IDENTIFYING_METADATA_COLUMNS
        if column != "pair_id" and column in frame.columns
    )
    return frame[columns].copy()


def _parse_input_csv_bytes(payload: bytes, *, path: Path) -> pd.DataFrame:
    """Parse captured CSV bytes with the project's exact UTF-8/NA contract."""
    try:
        return pd.read_csv(
            io.BytesIO(payload),
            encoding="utf-8",
            keep_default_na=False,
            na_values=[""],
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Malformed JUDGEBERT input CSV {path}: {exc}") from exc


def run_judgebert_prediction(
    frame: pd.DataFrame,
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    checkpoint_manifest_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
    input_path: str | Path | None = None,
    tokenizer: Any | None = None,
    model: JudgeBertModel | None = None,
    allow_download: bool = False,
    device: str | None = None,
) -> ResultPaths:
    """Validate, restore, predict, then write a provenance-complete bundle."""
    if not isinstance(allow_download, bool):
        raise ValueError("allow_download must be a boolean")
    caller_work = _validated_prediction_input(frame)
    checkpoint_file = Path(checkpoint_path)
    manifest_file = (
        Path(checkpoint_manifest_path)
        if checkpoint_manifest_path is not None
        else checkpoint_file.with_name("manifest.json")
    )
    source_path = Path(input_path) if input_path is not None else None
    if source_path is not None and not source_path.is_file():
        raise FileNotFoundError(f"JUDGEBERT input CSV not found: {source_path}")
    if source_path is not None:
        source_bytes = source_path.read_bytes()
        source_work = _validated_prediction_input(
            _parse_input_csv_bytes(source_bytes, path=source_path)
        )
        caller_projection = _input_hash_frame(caller_work)
        source_projection = _input_hash_frame(source_work)
        if (
            caller_projection.columns.tolist()
            != source_projection.columns.tolist()
            or not caller_projection.equals(source_projection)
        ):
            raise ValueError(
                "JUDGEBERT caller frame label-free projection differs from input_path"
            )
        work = source_work
    else:
        source_bytes = None
        work = caller_work
    destination, metadata_path = _validate_output_paths(
        output_path,
        checkpoint_path=checkpoint_file,
        checkpoint_manifest_path=manifest_file,
        input_path=source_path,
    )
    checkpoint = load_judgebert_checkpoint(
        checkpoint_file,
        config=config,
        map_location="cpu",
    )
    checkpoint_manifest = (
        _load_final_checkpoint_manifest(manifest_file, checkpoint)
        if checkpoint.run_mode == "final"
        else None
    )
    if checkpoint_manifest is not None:
        destination, metadata_path = _validate_output_paths(
            output_path,
            checkpoint_path=checkpoint_file,
            checkpoint_manifest_path=manifest_file,
            input_path=source_path,
            manifest_lineage=checkpoint_manifest,
        )
    effective_cfg = copy.deepcopy(checkpoint.project_config)
    settings = JudgeBertConfig.from_project_config(effective_cfg)
    seed = setup_model_seed(effective_cfg)
    requested_device = device or str(
        (effective_cfg.get("training") or {}).get("device", "auto")
    )
    resolved_device = select_device(requested_device)
    _eval_batch_size(effective_cfg)

    injected_tokenizer = tokenizer is not None
    injected_model = model is not None
    if injected_tokenizer != injected_model:
        raise ValueError(
            "Injected prediction components require both tokenizer and initialized model"
        )
    if tokenizer is None:
        tokenizer = _training.load_judgebert_tokenizer(
            effective_cfg, allow_download=allow_download
        )
    if model is None:
        model = JudgeBertModel.from_config(
            effective_cfg, allow_download=allow_download
        )
        _materialize_configured_model(model, tokenizer, work, settings)
    restore_judgebert_model(checkpoint, model)

    predictions = predict_judgebert(
        work,
        effective_cfg,
        tokenizer=tokenizer,
        model=model,
        device=str(resolved_device),
    )
    artifact_run_mode = "smoke_test" if checkpoint.synthetic else "inference"
    research_use = "non-research" if checkpoint.synthetic else "prediction-only"
    predictions["run_mode"] = artifact_run_mode
    predictions["synthetic"] = checkpoint.synthetic
    predictions["research_use"] = research_use
    pair_ids = predictions["pair_id"].astype(str).tolist()
    splits = (
        predictions["split"].astype(str).tolist()
        if "split" in predictions.columns
        else []
    )
    metadata = build_result_metadata(
        effective_cfg,
        model_name="judgebert",
        run_mode=artifact_run_mode,
        pair_ids=pair_ids,
        splits=splits,
        seed=seed,
        device=str(resolved_device),
    )
    records = [
        {
            "pair_id": pair_id,
            **({"split": splits[index]} if splits else {}),
        }
        for index, pair_id in enumerate(pair_ids)
    ]
    input_data_frame = _input_hash_frame(work)
    input_data_sha256 = dataframe_sha256(input_data_frame)
    input_sha256 = (
        hashlib.sha256(source_bytes).hexdigest()
        if source_bytes is not None
        else input_data_sha256
    )
    metadata.update(
        {
            "model_identity": MODEL_IDENTITY,
            "backbone": settings.backbone,
            "checkpoint_format_version": checkpoint.format_version,
            "checkpoint_run_mode": checkpoint.run_mode,
            "checkpoint_synthetic": checkpoint.synthetic,
            "smoke_test": checkpoint.synthetic,
            "synthetic": checkpoint.synthetic,
            "research_use": research_use,
            "research_metrics_reported": False,
            "checkpoint_artifact": {
                "path": str(checkpoint.path.resolve()),
                "sha256": checkpoint.sha256,
            },
            "checkpoint_manifest_artifact": (
                {
                    "path": str(checkpoint_manifest.path),
                    "sha256": checkpoint_manifest.sha256,
                }
                if checkpoint_manifest is not None
                else None
            ),
            "checkpoint_sha256": checkpoint.sha256,
            "checkpoint_manifest_sha256": (
                checkpoint_manifest.sha256
                if checkpoint_manifest is not None
                else None
            ),
            "inference_config_identity": {
                "config_fingerprint": checkpoint.config_fingerprint,
                "config_sha256": checkpoint.config_sha256,
                "effective_seed": checkpoint.effective_seed,
            },
            "input_data_columns": input_data_frame.columns.tolist(),
            "input_data_sha256": input_data_sha256,
            "input_sha256": input_sha256,
            "input_hash_kind": (
                "file_sha256" if source_path is not None else "dataframe_sha256"
            ),
            "pair_ids": pair_ids,
            "records": records,
            "score_range": [settings.score_min, settings.score_max],
            "effective_config": {
                "judgebert": asdict(settings),
                "eval_batch_size": _eval_batch_size(effective_cfg),
                "seed": seed,
                "device": str(resolved_device),
            },
            "loading_policy": {
                "allow_download": allow_download,
                "local_files_only": not allow_download,
                "trust_remote_code": False,
                "tokenizer_source": (
                    "injected_test_tokenizer"
                    if injected_tokenizer
                    else "configured_muril"
                ),
                "model_source": (
                    "injected_test_model" if injected_model else "configured_muril"
                ),
            },
        }
    )

    prediction_temp = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    metadata_temp = metadata_path.with_name(
        f".{metadata_path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        io_utils.write_csv(predictions, prediction_temp)
        output_sha256 = file_sha256(prediction_temp)
        metadata["output_sha256"] = output_sha256
        metadata["predictions_sha256"] = output_sha256
        io_utils.write_json(metadata, metadata_temp)
        os.replace(prediction_temp, destination)
        os.replace(metadata_temp, metadata_path)
    finally:
        prediction_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    return ResultPaths(destination, metadata_path)


def main(
    argv: list[str] | None = None,
    *,
    tokenizer: Any | None = None,
    model: JudgeBertModel | None = None,
) -> int:
    """Run checkpoint inference without reading or deriving any target field."""
    parser = argparse.ArgumentParser(
        description="Run label-free JUDGEBERT-inspired MuRIL checkpoint inference."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Portable JUDGEBERT checkpoint.pt path",
    )
    parser.add_argument(
        "--checkpoint-manifest",
        default=None,
        help="Training manifest path (defaults to manifest.json beside checkpoint)",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Pair CSV with pair_id, original_text, and simplified_text",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Label-free prediction CSV path (provenance is written beside it)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional project config YAML; model settings must match checkpoint",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow downloading the exact google/muril-base-cased assets",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="Inference device (defaults to training.device in the effective config)",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_model_config(args.config) if args.config is not None else None
        frame = io_utils.read_csv(args.input)
        paths = run_judgebert_prediction(
            frame,
            checkpoint_path=args.checkpoint,
            checkpoint_manifest_path=args.checkpoint_manifest,
            output_path=args.output,
            input_path=args.input,
            config=cfg,
            tokenizer=tokenizer,
            model=model,
            allow_download=args.allow_download,
            device=args.device,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote label-free JUDGEBERT predictions to {paths.predictions_csv}")
    print(f"Wrote JUDGEBERT provenance metadata to {paths.metadata_json}")
    return 0


__all__ = [
    "JudgeBertCheckpoint",
    "load_judgebert_checkpoint",
    "main",
    "predict_judgebert",
    "restore_judgebert_model",
    "run_judgebert_prediction",
]


if __name__ == "__main__":
    sys.exit(main())
