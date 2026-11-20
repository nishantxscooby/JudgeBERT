"""Training and checkpointing for the JUDGEBERT-inspired MuRIL model.

The production path is deliberately offline-first and never substitutes a toy
encoder or tokenizer for the configured MuRIL assets.  Tiny components may be
injected only through the Python API for isolated unit tests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from io import BytesIO
from math import ceil, cos, isfinite, pi, sqrt
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.models.common import (
    load_model_config,
    load_model_corpus,
    require_gold,
    select_device,
    setup_model_seed,
    validate_model_corpus,
)
from src.models.provenance import (
    dataframe_sha256,
    ensure_output_directory,
    file_sha256,
    source_tree_sha256,
)
from src.utils import io as io_utils
from src.utils.config import config_fingerprint, dump_config
from src.utils.io import provenance

from src.models.judgebert_model import (
    MODEL_IDENTITY,
    JudgeBertConfig,
    JudgeBertModel,
    _huggingface_offline_guard,
)


# Stable, human-gold-only bytes bound to every final training checkpoint.
# Draft, AI-generated, synthetic, and derived score fields are deliberately absent.
FINAL_TRAINING_INPUT_COLUMNS = (
    "pair_id",
    "original_text",
    "simplified_text",
    "language",
    "legal_domain",
    "source_group",
    "split",
    "human_lmp_score",
    "annotation_status",
)


def _load_auto_tokenizer_class() -> Any:
    """Import transformers only when the configured tokenizer is requested."""
    from transformers import AutoTokenizer

    return AutoTokenizer


def load_judgebert_tokenizer(
    cfg: Mapping[str, Any], *, allow_download: bool = False
) -> Any:
    """Load the exact configured tokenizer without network access by default."""
    settings = JudgeBertConfig.from_project_config(cfg)
    if not isinstance(allow_download, bool):
        raise ValueError("allow_download must be a boolean")
    local_files_only = not allow_download
    try:
        with _huggingface_offline_guard(local_files_only):
            return _load_auto_tokenizer_class().from_pretrained(
                settings.backbone,
                local_files_only=local_files_only,
                trust_remote_code=False,
            )
    except Exception as exc:
        if local_files_only:
            policy = (
                "local_files_only=True; cache the exact tokenizer locally or pass "
                "--allow-download explicitly"
            )
        else:
            policy = "local_files_only=False"
        raise RuntimeError(
            f"{MODEL_IDENTITY} tokenizer {settings.backbone!r} is unavailable with "
            f"{policy}. No fallback tokenizer will be used. Original error: {exc}"
        ) from exc


def encode_text_pairs(
    tokenizer: Any,
    originals: Sequence[str],
    simplified: Sequence[str],
    *,
    max_length: int,
    padding: bool | str = True,
) -> dict[str, Tensor]:
    """Encode ``[CLS] original [SEP] simplified [SEP]`` with a HF tokenizer."""
    encoded = tokenizer(
        list(originals),
        list(simplified),
        truncation="longest_first",
        padding=padding,
        max_length=max_length,
        return_tensors="pt",
    )
    keys = ("input_ids", "attention_mask", "token_type_ids")
    return {key: encoded[key] for key in keys if key in encoded}


class _EncodedPairDataset(Dataset):
    def __init__(self, encodings: Mapping[str, Tensor], labels: Tensor) -> None:
        self.encodings = dict(encodings)
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


@dataclass(frozen=True)
class TrainingData:
    """Encoded train/validation data and their exact fit identities."""

    train_loader: DataLoader
    validation_loader: DataLoader
    train_ids: list[str]
    validation_ids: list[str]
    fit_frame: pd.DataFrame


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    adam_epsilon: float
    warmup_ratio: float
    max_grad_norm: float
    scheduler: str
    gradient_accumulation_steps: int
    freeze_encoder_epochs: int
    early_stopping_enabled: bool
    early_stopping_patience: int
    early_stopping_min_delta: float

    @classmethod
    def from_project_config(
        cls, cfg: Mapping[str, Any], *, epochs: int | None = None
    ) -> "TrainingSettings":
        training = cfg.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("Project config is missing the 'training' mapping")
        effective_epochs = training.get("epochs") if epochs is None else epochs
        values = dict(training)
        values["epochs"] = effective_epochs
        integer_names = (
            "epochs",
            "batch_size",
            "eval_batch_size",
            "gradient_accumulation_steps",
        )
        parsed_ints = {name: _required_positive_int(values, name) for name in integer_names}
        freeze_epochs = values.get("freeze_encoder_epochs")
        if (
            isinstance(freeze_epochs, bool)
            or not isinstance(freeze_epochs, int)
            or freeze_epochs < 0
        ):
            raise ValueError("training.freeze_encoder_epochs must be a non-negative integer")

        numeric_rules = {
            "learning_rate": (0.0, False),
            "weight_decay": (0.0, True),
            "adam_epsilon": (0.0, False),
            "warmup_ratio": (0.0, True),
            "max_grad_norm": (0.0, False),
        }
        parsed_numbers: dict[str, float] = {}
        for name, (minimum, inclusive) in numeric_rules.items():
            value = values.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"training.{name} must be a finite number")
            number = float(value)
            valid_lower = number >= minimum if inclusive else number > minimum
            if not isfinite(number) or not valid_lower:
                qualifier = "non-negative" if inclusive else "positive"
                raise ValueError(f"training.{name} must be finite and {qualifier}")
            parsed_numbers[name] = number
        if parsed_numbers["warmup_ratio"] > 1.0:
            raise ValueError("training.warmup_ratio must be within 0-1")
        scheduler = values.get("scheduler")
        if scheduler not in {"linear", "cosine", "constant"}:
            raise ValueError("training.scheduler must be linear, cosine, or constant")

        early = values.get("early_stopping")
        if not isinstance(early, Mapping):
            raise ValueError("training.early_stopping must be a mapping")
        enabled = early.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("training.early_stopping.enabled must be a boolean")
        patience = early.get("patience")
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError("training.early_stopping.patience must be positive")
        if early.get("metric") != "val_rmse" or early.get("mode") != "min":
            raise ValueError("JUDGEBERT early stopping must minimize val_rmse")
        min_delta = early.get("min_delta")
        if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)):
            raise ValueError("training.early_stopping.min_delta must be non-negative")
        min_delta = float(min_delta)
        if not isfinite(min_delta) or min_delta < 0.0:
            raise ValueError("training.early_stopping.min_delta must be non-negative")

        return cls(
            epochs=parsed_ints["epochs"],
            batch_size=parsed_ints["batch_size"],
            eval_batch_size=parsed_ints["eval_batch_size"],
            learning_rate=parsed_numbers["learning_rate"],
            weight_decay=parsed_numbers["weight_decay"],
            adam_epsilon=parsed_numbers["adam_epsilon"],
            warmup_ratio=parsed_numbers["warmup_ratio"],
            max_grad_norm=parsed_numbers["max_grad_norm"],
            scheduler=str(scheduler),
            gradient_accumulation_steps=parsed_ints["gradient_accumulation_steps"],
            freeze_encoder_epochs=freeze_epochs,
            early_stopping_enabled=enabled,
            early_stopping_patience=patience,
            early_stopping_min_delta=min_delta,
        )


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, Any]]
    best_epoch: int
    best_validation_rmse: float
    stopped_early: bool
    total_optimizer_steps: int
    validation_predictions: pd.DataFrame
    train_ids: list[str]
    validation_ids: list[str]
    fit_frame: pd.DataFrame


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint: Path
    predictions: Path
    manifest: Path
    effective_config: Path

    def __iter__(self):
        return iter(
            (self.checkpoint, self.predictions, self.manifest, self.effective_config)
        )


def _required_positive_int(mapping: Mapping[str, Any], name: str) -> int:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"training.{name} must be a positive integer")
    return value


def _validated_fit_split(
    frame: pd.DataFrame, split: str, *, label_column: str
) -> pd.DataFrame:
    required = {
        "pair_id",
        "split",
        "original_text",
        "simplified_text",
        label_column,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{split} data is missing columns: {missing}")
    selected = frame.loc[frame["split"].astype(str).str.strip().eq(split)].copy()
    if selected.empty:
        raise ValueError(f"JUDGEBERT training requires at least one {split} row")
    for column in ("original_text", "simplified_text"):
        invalid = selected[column].isna() | selected[column].astype("string").str.strip().eq("")
        if invalid.any():
            raise ValueError(f"{split} {column} contains null or blank values")
    labels = pd.to_numeric(selected[label_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or ((labels < 1.0) | (labels > 10.0)).any():
        raise ValueError(f"{split} {label_column} values must be finite scores within 1-10")
    return selected.reset_index(drop=True)


def build_training_dataloaders(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    tokenizer: Any,
    *,
    label_column: str,
    seed: int,
) -> TrainingData:
    """Build deterministic Windows-safe loaders without touching test rows."""
    validate_model_corpus(frame)
    if "draft" in label_column.casefold():
        raise ValueError("Draft fields must never be used as training labels")
    train = _validated_fit_split(frame, "train", label_column=label_column)
    validation = _validated_fit_split(frame, "validation", label_column=label_column)
    model_settings = JudgeBertConfig.from_project_config(cfg)
    training = cfg.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("Project config is missing the 'training' mapping")
    batch_size = _required_positive_int(training, "batch_size")
    eval_batch_size = _required_positive_int(training, "eval_batch_size")

    def dataset_for(part: pd.DataFrame) -> _EncodedPairDataset:
        encoded = encode_text_pairs(
            tokenizer,
            part["original_text"].tolist(),
            part["simplified_text"].tolist(),
            max_length=model_settings.max_seq_length,
            padding=True,
        )
        labels = torch.tensor(
            pd.to_numeric(part[label_column]).to_numpy(dtype=float),
            dtype=torch.float32,
        )
        return _EncodedPairDataset(encoded, labels)

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        dataset_for(train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        dataset_for(validation),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0,
    )
    fit_frame = pd.concat([train, validation], ignore_index=True)
    return TrainingData(
        train_loader=train_loader,
        validation_loader=validation_loader,
        train_ids=train["pair_id"].astype(str).tolist(),
        validation_ids=validation["pair_id"].astype(str).tolist(),
        fit_frame=fit_frame,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if name == "constant":
            return 1.0
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        if name == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + cos(pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)


def _set_encoder_trainable(model: JudgeBertModel, trainable: bool) -> None:
    if model.encoder is None:
        raise RuntimeError("JUDGEBERT model must be initialized before training")
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(trainable)


def _batch_model_kwargs(
    batch: Mapping[str, Tensor], device: torch.device, *, include_labels: bool
) -> dict[str, Tensor]:
    keys = ["input_ids", "attention_mask", "token_type_ids"]
    if include_labels:
        keys.append("labels")
    return {key: batch[key].to(device) for key in keys if key in batch}


@torch.no_grad()
def _evaluate_model(
    model: JudgeBertModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, list[float]]:
    was_training = model.training
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    try:
        for batch in loader:
            kwargs = _batch_model_kwargs(batch, device, include_labels=False)
            scores = model.predict_scores(**kwargs)
            predictions.append(scores.detach().cpu())
            targets.append(batch["labels"].detach().cpu())
    finally:
        model.train(was_training)
    predicted = torch.cat(predictions).to(torch.float64)
    expected = torch.cat(targets).to(torch.float64)
    if not bool(torch.isfinite(predicted).all()):
        raise RuntimeError(f"{MODEL_IDENTITY} produced non-finite validation predictions")
    if bool(((predicted < 1.0) | (predicted > 10.0)).any()):
        raise RuntimeError(f"{MODEL_IDENTITY} produced validation predictions outside 1-10")
    rmse = sqrt(float(torch.mean((predicted - expected) ** 2).item()))
    if not isfinite(rmse):
        raise RuntimeError(f"{MODEL_IDENTITY} produced non-finite validation RMSE")
    return rmse, predicted.tolist()


def train_judgebert(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    tokenizer: Any,
    model: JudgeBertModel,
    label_column: str,
    epochs: int | None = None,
    device: str | torch.device = "cpu",
) -> TrainingResult:
    """Fine-tune an initialized model using train and validation rows only."""
    settings = TrainingSettings.from_project_config(cfg, epochs=epochs)
    seed = setup_model_seed(cfg)
    data = build_training_dataloaders(
        frame,
        cfg,
        tokenizer,
        label_column=label_column,
        seed=seed,
    )
    resolved_device = torch.device(device)
    model.to(resolved_device)
    if model.encoder is None or model.regression_head is None:
        raise RuntimeError("JUDGEBERT model must be initialized before training")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
        eps=settings.adam_epsilon,
    )
    steps_per_epoch = ceil(
        len(data.train_loader) / settings.gradient_accumulation_steps
    )
    total_planned_steps = steps_per_epoch * settings.epochs
    scheduler = _build_scheduler(
        optimizer,
        name=settings.scheduler,
        total_steps=total_planned_steps,
        warmup_ratio=settings.warmup_ratio,
    )

    history: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_rmse = float("inf")
    epochs_without_improvement = 0
    total_optimizer_steps = 0
    stopped_early = False

    for epoch_index in range(settings.epochs):
        encoder_frozen = epoch_index < settings.freeze_encoder_epochs
        _set_encoder_trainable(model, not encoder_frozen)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        example_count = 0
        epoch_optimizer_steps = 0
        final_group_size = len(data.train_loader) % settings.gradient_accumulation_steps

        for batch_index, batch in enumerate(data.train_loader):
            kwargs = _batch_model_kwargs(batch, resolved_device, include_labels=True)
            output = model(**kwargs)
            if output.loss is None or not bool(torch.isfinite(output.loss)):
                raise RuntimeError(f"{MODEL_IDENTITY} produced a non-finite training loss")
            batch_size = int(batch["labels"].shape[0])
            loss_sum += float(output.loss.detach().item()) * batch_size
            example_count += batch_size
            in_partial_final_group = (
                final_group_size > 0
                and batch_index >= len(data.train_loader) - final_group_size
            )
            accumulation_divisor = (
                final_group_size
                if in_partial_final_group
                else settings.gradient_accumulation_steps
            )
            (output.loss / accumulation_divisor).backward()

            boundary = (batch_index + 1) % settings.gradient_accumulation_steps == 0
            final_batch = batch_index + 1 == len(data.train_loader)
            if boundary or final_batch:
                parameters_with_grad = [
                    parameter for parameter in model.parameters() if parameter.grad is not None
                ]
                torch.nn.utils.clip_grad_norm_(
                    parameters_with_grad, settings.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_optimizer_steps += 1
                total_optimizer_steps += 1

        train_loss = loss_sum / example_count
        validation_rmse, _ = _evaluate_model(
            model, data.validation_loader, resolved_device
        )
        improved = validation_rmse < best_rmse - settings.early_stopping_min_delta
        if improved:
            best_rmse = validation_rmse
            best_epoch = epoch_index + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch_index + 1,
                "train_loss": float(train_loss),
                "validation_rmse": float(validation_rmse),
                "optimizer_steps": epoch_optimizer_steps,
                "encoder_frozen": encoder_frozen,
            }
        )
        if (
            settings.early_stopping_enabled
            and epochs_without_improvement >= settings.early_stopping_patience
        ):
            stopped_early = True
            break

    if best_state is None:
        raise RuntimeError("JUDGEBERT training completed without a finite best checkpoint")
    model.load_state_dict(best_state, strict=True)
    _set_encoder_trainable(model, True)
    restored_rmse, restored_scores = _evaluate_model(
        model, data.validation_loader, resolved_device
    )
    if abs(restored_rmse - best_rmse) > 1e-7:
        raise RuntimeError("Restored best JUDGEBERT state does not match its validation score")
    predictions = pd.DataFrame(
        {"pair_id": data.validation_ids, "prediction": restored_scores}
    )
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_rmse=float(best_rmse),
        stopped_early=stopped_early,
        total_optimizer_steps=total_optimizer_steps,
        validation_predictions=predictions,
        train_ids=data.train_ids,
        validation_ids=data.validation_ids,
        fit_frame=data.fit_frame,
    )


def build_synthetic_smoke_fixture(
    *, max_train_rows: int | None = None
) -> pd.DataFrame:
    """Return isolated bilingual smoke rows with only a generic test label."""
    train_rows = [
        ("JB_SMOKE_T01", "en", "A buyer shall pay within 30 days.", "The buyer must pay in 30 days.", 9.0),
        ("JB_SMOKE_T02", "en", "The seller may cancel after notice.", "The seller can cancel.", 6.0),
        ("JB_SMOKE_T03", "en", "No person may disclose the key.", "A person may disclose the key.", 2.0),
        ("JB_SMOKE_T04", "en", "Goods must be delivered on Monday.", "Goods must be delivered on Monday.", 10.0),
        ("JB_SMOKE_T05", "hi", "क्रेता तीस दिन में भुगतान करेगा।", "खरीदार 30 दिन में भुगतान करेगा।", 9.0),
        ("JB_SMOKE_T06", "hi", "कोई व्यक्ति कुंजी साझा नहीं करेगा।", "व्यक्ति कुंजी साझा कर सकता है।", 2.0),
        ("JB_SMOKE_T07", "hi", "विक्रेता सूचना देकर अनुबंध रद्द कर सकता है।", "विक्रेता अनुबंध रद्द कर सकता है।", 6.0),
        ("JB_SMOKE_T08", "hi", "माल सोमवार को दिया जाएगा।", "माल सोमवार को दिया जाएगा।", 10.0),
    ]
    if max_train_rows is not None:
        if (
            isinstance(max_train_rows, bool)
            or not isinstance(max_train_rows, int)
            or max_train_rows < 1
        ):
            raise ValueError("max_train_rows must be a positive integer")
        train_rows = train_rows[:max_train_rows]
    validation_rows = [
        ("JB_SMOKE_V01", "en", "A password must remain confidential.", "The password stays confidential.", 8.0),
        ("JB_SMOKE_V02", "hi", "विक्रेता माल नहीं रोकेगा।", "विक्रेता माल रोकेगा।", 2.0),
    ]
    records = []
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for pair_id, language, original, simplified, label in rows:
            records.append(
                {
                    "pair_id": pair_id,
                    "language": language,
                    "legal_domain": "contract",
                    "source_group": f"judgebert-smoke::{pair_id}",
                    "split": split,
                    "original_text": original,
                    "simplified_text": simplified,
                    "label": label,
                }
            )
    return pd.DataFrame.from_records(records)


def final_training_input_projection(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact ordered human-gold projection hashed for final training."""
    missing = sorted(set(FINAL_TRAINING_INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            "JUDGEBERT final training input is missing canonical columns: "
            f"{missing}"
        )
    return frame.loc[:, FINAL_TRAINING_INPUT_COLUMNS].copy().reset_index(drop=True)


def _validated_checkpoint_fit_lineage(result: TrainingResult) -> dict[str, Any]:
    """Return canonical train/validation fit lineage consistent with result IDs."""
    frame = result.fit_frame.reset_index(drop=True)
    missing = sorted({"pair_id", "split"} - set(frame.columns))
    if missing:
        raise ValueError(f"JUDGEBERT checkpoint fit rows are missing columns: {missing}")
    if frame.empty:
        raise ValueError("JUDGEBERT checkpoint fit rows must be nonempty")
    raw_ids = frame["pair_id"]
    pair_ids = raw_ids.astype("string").str.strip()
    if raw_ids.isna().any() or pair_ids.eq("").fillna(True).any():
        raise ValueError("JUDGEBERT checkpoint fit rows contain blank pair_id values")
    if pair_ids.duplicated().any():
        raise ValueError("JUDGEBERT checkpoint fit rows contain duplicate pair_id values")
    splits = frame["split"].astype("string").str.strip()
    if frame["split"].isna().any() or not splits.isin(["train", "validation"]).all():
        raise ValueError(
            "JUDGEBERT checkpoint fit rows must contain only train/validation splits"
        )
    ids_by_split = {
        split: pair_ids.loc[splits.eq(split)].astype(str).tolist()
        for split in ("train", "validation")
    }
    declared_ids = {
        "train": [str(value).strip() for value in result.train_ids],
        "validation": [str(value).strip() for value in result.validation_ids],
    }
    if any(not values or len(values) != len(set(values)) for values in declared_ids.values()):
        raise ValueError("JUDGEBERT checkpoint fit IDs must be nonempty and unique")
    if declared_ids != ids_by_split:
        raise ValueError(
            "JUDGEBERT checkpoint fit IDs do not match the ordered fit rows"
        )
    records = [
        {"pair_id": pair_id, "split": split}
        for pair_id, split in zip(
            pair_ids.astype(str).tolist(), splits.astype(str).tolist(), strict=True
        )
    ]
    return {
        "splits": ["train", "validation"],
        "ids_by_split": ids_by_split,
        "records": records,
    }


def _config_sha256(cfg: Mapping[str, Any]) -> str:
    def portable(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: portable(nested)
                for key, nested in value.items()
                if not str(key).endswith("_abs")
            }
        if isinstance(value, (list, tuple)):
            return [portable(nested) for nested in value]
        return value

    payload = json.dumps(
        portable(cfg),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _effective_model_seed(cfg: Mapping[str, Any]) -> int:
    training = cfg.get("training") or {}
    project = cfg.get("project") or {}
    value = training.get("seed", project.get("seed"))
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Effective JUDGEBERT model seed must be an integer")
    return value


def save_judgebert_checkpoint(
    path: str | Path,
    model: JudgeBertModel,
    cfg: Mapping[str, Any],
    result: TrainingResult,
    *,
    run_mode: str,
    synthetic: bool,
) -> Path:
    """Save a portable best-state checkpoint with its effective configuration."""
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Checkpoint parent directory does not exist: {destination.parent}"
        )
    research_use = "non-research" if synthetic else "research-training-only"
    if run_mode == "final":
        require_gold(result.fit_frame, context="JUDGEBERT final checkpoint")
        fit_lineage = _validated_checkpoint_fit_lineage(result)
        input_projection = final_training_input_projection(result.fit_frame)
    else:
        fit_lineage = None
        input_projection = result.fit_frame.reset_index(drop=True)
    input_data_columns = input_projection.columns.astype(str).tolist()
    config_identity = config_fingerprint(cfg)
    config_sha256 = _config_sha256(cfg)
    effective_seed = _effective_model_seed(cfg)
    code_sha256 = source_tree_sha256()
    input_data_sha256 = dataframe_sha256(input_projection)
    lineage = {
        "config_fingerprint": config_identity,
        "config_sha256": config_sha256,
        "effective_seed": effective_seed,
        "code_sha256": code_sha256,
        "input_data_columns": input_data_columns,
        "input_data_sha256": input_data_sha256,
        **({"fit": fit_lineage} if fit_lineage is not None else {}),
    }
    payload = {
        "format_version": 1,
        "model_identity": MODEL_IDENTITY,
        "backbone": model.settings.backbone,
        "model_state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "project_config": copy.deepcopy(dict(cfg)),
        "run_mode": str(run_mode),
        "smoke_test": run_mode == "smoke_test",
        "synthetic": bool(synthetic),
        "data_source": (
            "in_memory_synthetic_fixture" if synthetic else "human_gold_corpus"
        ),
        "research_use": research_use,
        "research_metrics_reported": False,
        "config_fingerprint": config_identity,
        "config_sha256": config_sha256,
        "effective_seed": effective_seed,
        "code_sha256": code_sha256,
        "input_data_columns": input_data_columns,
        "input_data_sha256": input_data_sha256,
        "lineage": lineage,
        "best_epoch": int(result.best_epoch),
        "best_validation_rmse": float(result.best_validation_rmse),
        "history": [
            {**copy.deepcopy(row), "research_use": research_use}
            for row in result.history
        ],
        "train_ids": list(result.train_ids),
        "validation_ids": list(result.validation_ids),
        **(
            {
                "fit": fit_lineage,
                "fit_ids": fit_lineage["ids_by_split"],
                "fit_splits": fit_lineage["splits"],
            }
            if fit_lineage is not None
            else {}
        ),
    }
    torch.save(payload, destination)
    return destination


def _read_checkpoint(
    path: str | Path | BinaryIO, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    checkpoint_source = path if hasattr(path, "read") else Path(path)
    payload = torch.load(
        checkpoint_source, map_location=map_location, weights_only=True
    )
    if not isinstance(payload, dict):
        raise ValueError("JUDGEBERT checkpoint payload must be a mapping")
    return payload


def load_checkpoint_into_model(
    path: str | Path | BinaryIO,
    model: JudgeBertModel,
    *,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint state into an injected compatible initialized model."""
    payload = _read_checkpoint(path, map_location=map_location)
    if payload.get("model_identity") != MODEL_IDENTITY:
        raise ValueError("Checkpoint model identity is not JUDGEBERT-inspired MuRIL")
    if payload.get("backbone") != model.settings.backbone:
        raise ValueError(
            "Checkpoint backbone does not match the injected compatible model"
        )
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint is missing model_state_dict")
    model.load_state_dict(state, strict=strict)
    return payload


def initialize_judgebert_model(
    model: JudgeBertModel,
    tokenizer: Any,
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    label_column: str | None = None,
) -> JudgeBertModel:
    """Materialize the configured encoder and head before any output is created."""
    effective_label = label_column or (
        "label" if "label" in frame else "human_lmp_score"
    )
    sample = _validated_fit_split(
        frame, "train", label_column=effective_label
    ).iloc[:1]
    settings = JudgeBertConfig.from_project_config(cfg)
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
    return model


@torch.no_grad()
def _predict_fit_frame(
    model: JudgeBertModel,
    tokenizer: Any,
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    device: torch.device,
) -> list[float]:
    settings = JudgeBertConfig.from_project_config(cfg)
    training = TrainingSettings.from_project_config(cfg)
    encoded = encode_text_pairs(
        tokenizer,
        frame["original_text"].tolist(),
        frame["simplified_text"].tolist(),
        max_length=settings.max_seq_length,
        padding=True,
    )
    scores: list[Tensor] = []
    for start in range(0, len(frame), training.eval_batch_size):
        batch = {
            key: value[start : start + training.eval_batch_size].to(device)
            for key, value in encoded.items()
        }
        scores.append(model.predict_scores(**batch).detach().cpu())
    combined = torch.cat(scores).to(torch.float64)
    if not bool(torch.isfinite(combined).all()):
        raise RuntimeError(f"{MODEL_IDENTITY} produced non-finite artifact predictions")
    if bool(((combined < 1.0) | (combined > 10.0)).any()):
        raise RuntimeError(f"{MODEL_IDENTITY} produced artifact predictions outside 1-10")
    return combined.tolist()


def run_judgebert_training(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    output_dir: str | Path,
    run_mode: str,
    synthetic: bool,
    label_column: str,
    tokenizer: Any | None = None,
    model: JudgeBertModel | None = None,
    epochs: int | None = None,
    allow_download: bool = False,
    device: str | None = None,
) -> TrainingArtifacts:
    """Initialize, train, reload, predict, and write an auditable artifact bundle."""
    if not isinstance(allow_download, bool):
        raise ValueError("allow_download must be a boolean")
    if run_mode not in {"smoke_test", "final"}:
        raise ValueError("run_mode must be 'smoke_test' or 'final'")
    if run_mode == "smoke_test":
        if not synthetic or label_column != "label":
            raise ValueError(
                "Smoke mode requires synthetic=True and the generic test-only 'label' column"
            )
        forbidden = {
            "human_lmp_score",
            "draft_lmp_score",
            "annotation_status",
            "decision",
            "error_type",
        }
        present = sorted(forbidden & set(frame.columns))
        if present:
            raise ValueError(f"Smoke fixture contains forbidden corpus fields: {present}")
        research_use = "non-research"
        data_source = "in_memory_synthetic_fixture"
    else:
        require_gold(frame, context="JUDGEBERT final training")
        if synthetic or label_column != "human_lmp_score":
            raise ValueError(
                "Final mode requires synthetic=False and human_lmp_score labels"
            )
        research_use = "research-training-only"
        data_source = "human_gold_corpus"

    effective_cfg = copy.deepcopy(dict(cfg))
    if epochs is not None:
        effective_cfg.setdefault("training", {})["epochs"] = epochs
    TrainingSettings.from_project_config(effective_cfg)
    effective_seed = setup_model_seed(effective_cfg)
    requested_device = device or str(
        (effective_cfg.get("training") or {}).get("device", "auto")
    )
    resolved_device = select_device(requested_device)

    injected_tokenizer = tokenizer is not None
    injected_model = model is not None
    if tokenizer is None:
        tokenizer = load_judgebert_tokenizer(
            effective_cfg, allow_download=allow_download
        )
    if model is None:
        model = JudgeBertModel.from_config(
            effective_cfg, allow_download=allow_download
        )
    initialize_judgebert_model(
        model,
        tokenizer,
        frame,
        effective_cfg,
        label_column=label_column,
    )
    model.to(resolved_device)

    destination = ensure_output_directory(output_dir)
    result = train_judgebert(
        frame,
        effective_cfg,
        tokenizer=tokenizer,
        model=model,
        label_column=label_column,
        device=resolved_device,
    )

    checkpoint_path = destination / "checkpoint.pt"
    save_judgebert_checkpoint(
        checkpoint_path,
        model,
        effective_cfg,
        result,
        run_mode=run_mode,
        synthetic=synthetic,
    )
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    restored_checkpoint = load_checkpoint_into_model(
        BytesIO(checkpoint_bytes), model, map_location=resolved_device, strict=True
    )

    fit_frame = result.fit_frame.reset_index(drop=True)
    prediction_values = _predict_fit_frame(
        model, tokenizer, fit_frame, effective_cfg, resolved_device
    )
    prediction_name = (
        "smoke_predictions.csv"
        if run_mode == "smoke_test"
        else "validation_predictions.csv"
    )
    prediction_path = destination / prediction_name
    predictions = pd.DataFrame(
        {
            "pair_id": fit_frame["pair_id"].astype(str).tolist(),
            "split": fit_frame["split"].astype(str).tolist(),
            "prediction": prediction_values,
            "run_mode": run_mode,
            "synthetic": bool(synthetic),
            "research_use": research_use,
        }
    )
    io_utils.write_csv(predictions, prediction_path)

    effective_config_path = destination / "effective_config.yaml"
    dump_config(effective_cfg, effective_config_path)
    history = [
        {
            **copy.deepcopy(row),
            "research_use": research_use,
            "purpose": "training_diagnostic_and_validation_selection",
        }
        for row in result.history
    ]
    fit_records = [
        {"pair_id": pair_id, "split": "train"}
        for pair_id in result.train_ids
    ] + [
        {"pair_id": pair_id, "split": "validation"}
        for pair_id in result.validation_ids
    ]
    fit_ids = {
        "train": list(result.train_ids),
        "validation": list(result.validation_ids),
    }
    if run_mode == "final":
        checkpoint_fit = restored_checkpoint["fit"]
        fit_records = copy.deepcopy(checkpoint_fit["records"])
        fit_ids = copy.deepcopy(checkpoint_fit["ids_by_split"])
    hashes = {
        "config_sha256": restored_checkpoint["config_sha256"],
        "code_sha256": restored_checkpoint["code_sha256"],
        "input_data_sha256": restored_checkpoint["input_data_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "predictions_sha256": file_sha256(prediction_path),
        "effective_config_sha256": file_sha256(effective_config_path),
    }
    manifest = {
        "provenance": provenance(
            effective_cfg,
            artifact="judgebert_training",
            seed=effective_seed,
        ),
        "model_identity": MODEL_IDENTITY,
        "backbone": model.settings.backbone,
        "run_mode": run_mode,
        "smoke_test": run_mode == "smoke_test",
        "synthetic": bool(synthetic),
        "data_source": data_source,
        "research_use": research_use,
        "research_metrics_reported": False,
        "effective_seed": effective_seed,
        "input_data_columns": copy.deepcopy(
            restored_checkpoint["input_data_columns"]
        ),
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "effective_config": effective_config_path.name,
            "predictions": prediction_path.name,
        },
        "loading_policy": {
            "allow_download": bool(allow_download),
            "local_files_only": not bool(allow_download),
            "trust_remote_code": False,
            "tokenizer_source": (
                "injected_test_tokenizer" if injected_tokenizer else "configured_muril"
            ),
            "model_source": (
                "injected_test_model" if injected_model else "configured_muril"
            ),
        },
        "fit": {
            "splits": ["train", "validation"],
            "ids_by_split": fit_ids,
            "records": fit_records,
        },
        "fit_ids": fit_ids,
        "fit_splits": ["train", "validation"],
        "best_epoch": result.best_epoch,
        "selection": {
            "metric": "validation_rmse",
            "value": result.best_validation_rmse,
            "research_use": research_use,
        },
        "history": history,
        "stopped_early": result.stopped_early,
        "checkpoint_reloaded_before_prediction": True,
        **hashes,
        "hashes": hashes,
    }
    manifest_path = destination / "manifest.json"
    io_utils.write_json(manifest, manifest_path)
    return TrainingArtifacts(
        checkpoint=checkpoint_path,
        predictions=prediction_path,
        manifest=manifest_path,
        effective_config=effective_config_path,
    )


def main(
    argv: list[str] | None = None,
    *,
    tokenizer: Any | None = None,
    model: JudgeBertModel | None = None,
) -> int:
    """Run gold-gated final training or an isolated synthetic smoke test."""
    parser = argparse.ArgumentParser(
        description="Train the JUDGEBERT-inspired MuRIL paired-text regressor."
    )
    parser.add_argument("--config", default=None, help="Path to project config YAML")
    parser.add_argument(
        "--output-dir",
        default="models/checkpoints/judgebert",
        help="Directory for checkpoint and provenance artifacts",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use only the in-memory synthetic non-research fixture",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training.epochs (use 1 for a smoke run)",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=8,
        help="Maximum in-memory train rows in smoke mode",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow downloading the exact configured MuRIL assets",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_model_config(args.config)
        if args.smoke_test:
            frame = build_synthetic_smoke_fixture(
                max_train_rows=args.max_train_rows
            )
            run_mode = "smoke_test"
            synthetic = True
            label_column = "label"
        else:
            frame = load_model_corpus(config_path=args.config, mode="final")
            run_mode = "final"
            synthetic = False
            label_column = "human_lmp_score"
        artifacts = run_judgebert_training(
            frame,
            cfg,
            output_dir=args.output_dir,
            run_mode=run_mode,
            synthetic=synthetic,
            label_column=label_column,
            tokenizer=tokenizer,
            model=model,
            epochs=args.epochs,
            allow_download=args.allow_download,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    qualifier = "synthetic non-research smoke" if args.smoke_test else "final training"
    print(f"Wrote {qualifier} checkpoint to {artifacts.checkpoint}")
    print(f"Wrote label-free predictions to {artifacts.predictions}")
    print(f"Wrote provenance manifest to {artifacts.manifest}")
    return 0


__all__ = [
    "FINAL_TRAINING_INPUT_COLUMNS",
    "TrainingArtifacts",
    "TrainingData",
    "TrainingResult",
    "TrainingSettings",
    "build_synthetic_smoke_fixture",
    "build_training_dataloaders",
    "encode_text_pairs",
    "final_training_input_projection",
    "initialize_judgebert_model",
    "load_checkpoint_into_model",
    "load_judgebert_tokenizer",
    "main",
    "run_judgebert_training",
    "save_judgebert_checkpoint",
    "train_judgebert",
]


if __name__ == "__main__":
    sys.exit(main())
