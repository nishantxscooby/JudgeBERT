"""Core JUDGEBERT-inspired MuRIL cross-encoder regression model.

This module contains only the neural model and score transformations.  It does
not load project data, train a model, or write predictions.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODEL_IDENTITY = "JUDGEBERT-inspired MuRIL cross-encoder"
SCORE_MIN = 1.0
SCORE_MAX = 10.0


@contextmanager
def _huggingface_offline_guard(enabled: bool):
    """Prevent Hub traffic while local-only AutoModel discovery is active."""
    if not enabled:
        yield
        return

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"
    constants = sys.modules.get("huggingface_hub.constants")
    missing = object()
    previous_constant = (
        getattr(constants, "HF_HUB_OFFLINE", missing)
        if constants is not None
        else missing
    )
    if constants is not None:
        constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        current_constants = sys.modules.get("huggingface_hub.constants")
        if current_constants is not None:
            if previous_constant is missing:
                truthy = {"1", "ON", "TRUE", "YES"}
                current_constants.HF_HUB_OFFLINE = any(
                    (value or "").upper() in truthy for value in previous.values()
                )
            else:
                current_constants.HF_HUB_OFFLINE = previous_constant


def _load_auto_model_class() -> Any:
    """Import transformers only when configured weights are first requested."""
    from transformers import AutoModel

    return AutoModel


def _floating_tensor(values: Tensor | Any, *, name: str) -> Tensor:
    tensor = values if isinstance(values, Tensor) else torch.as_tensor(values)
    if not (tensor.is_floating_point() or tensor.is_complex()):
        tensor = tensor.to(torch.get_default_dtype())
    if tensor.is_complex():
        raise ValueError(f"{name} must be real-valued")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must be finite")
    return tensor


def scale_targets(targets: Tensor | Any) -> Tensor:
    """Map finite 1--10 LMP targets exactly onto the unit interval."""
    tensor = _floating_tensor(targets, name="Targets")
    if bool(((tensor < SCORE_MIN) | (tensor > SCORE_MAX)).any()):
        raise ValueError("Targets must be within the finite 1-10 LMP score range")
    return (tensor - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)


def unscale_targets(scaled_targets: Tensor | Any) -> Tensor:
    """Map finite unit-interval targets exactly back onto the 1--10 scale."""
    tensor = _floating_tensor(scaled_targets, name="Scaled targets")
    if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
        raise ValueError("Scaled targets must be within 0-1")
    return tensor * (SCORE_MAX - SCORE_MIN) + SCORE_MIN


@dataclass(frozen=True)
class JudgeBertConfig:
    """Validated settings from ``models.judgebert`` and ``dataset``."""

    backbone: str
    max_seq_length: int
    dropout: float
    head_hidden_size: int
    pooling: str
    target_scaling: str
    loss: str
    smooth_l1_beta: float
    clip_predictions: bool
    score_min: float = SCORE_MIN
    score_max: float = SCORE_MAX

    @classmethod
    def from_project_config(cls, cfg: Mapping[str, Any]) -> "JudgeBertConfig":
        models = cfg.get("models")
        if not isinstance(models, Mapping):
            raise ValueError("Project config is missing the 'models' mapping")
        model_cfg = models.get("judgebert")
        if not isinstance(model_cfg, Mapping):
            raise ValueError("Project config is missing the 'models.judgebert' mapping")
        required = {
            "enabled",
            "backbone",
            "max_seq_length",
            "dropout",
            "head_hidden_size",
            "pooling",
            "target_scaling",
            "loss",
            "smooth_l1_beta",
            "clip_predictions",
        }
        missing = sorted(required - set(model_cfg))
        if missing:
            raise ValueError(f"models.judgebert is missing setting(s): {missing}")
        if model_cfg["enabled"] is not True:
            raise ValueError("models.judgebert.enabled must be true")

        backbone = model_cfg["backbone"]
        if not isinstance(backbone, str) or not backbone.strip():
            raise ValueError("models.judgebert.backbone must be a non-empty string")
        max_seq_length = model_cfg["max_seq_length"]
        if isinstance(max_seq_length, bool) or not isinstance(max_seq_length, int) or max_seq_length < 1:
            raise ValueError("models.judgebert.max_seq_length must be a positive integer")
        dropout = model_cfg["dropout"]
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise ValueError("models.judgebert.dropout must be a number in [0, 1)")
        dropout = float(dropout)
        if not isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("models.judgebert.dropout must be a finite number in [0, 1)")
        hidden = model_cfg["head_hidden_size"]
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 0:
            raise ValueError("models.judgebert.head_hidden_size must be a non-negative integer")
        pooling = model_cfg["pooling"]
        if pooling not in {"cls", "mean"}:
            raise ValueError("models.judgebert.pooling must be 'cls' or 'mean'")
        scaling = model_cfg["target_scaling"]
        if scaling not in {"minmax", "none"}:
            raise ValueError("models.judgebert.target_scaling must be 'minmax' or 'none'")
        loss = model_cfg["loss"]
        if loss not in {"smooth_l1", "mse"}:
            raise ValueError("models.judgebert.loss must be 'smooth_l1' or 'mse'")
        beta = model_cfg["smooth_l1_beta"]
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise ValueError("models.judgebert.smooth_l1_beta must be positive")
        beta = float(beta)
        if not isfinite(beta) or beta <= 0.0:
            raise ValueError("models.judgebert.smooth_l1_beta must be finite and positive")
        clip = model_cfg["clip_predictions"]
        if not isinstance(clip, bool):
            raise ValueError("models.judgebert.clip_predictions must be a boolean")

        dataset = cfg.get("dataset")
        if not isinstance(dataset, Mapping):
            raise ValueError("Project config is missing the 'dataset' mapping")
        score_min = dataset.get("score_min")
        score_max = dataset.get("score_max")
        for field, value in (("score_min", score_min), ("score_max", score_max)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"dataset.{field} must be a number, not a boolean")
            if not isfinite(float(value)):
                raise ValueError(f"dataset.{field} must be finite")
        if score_min != SCORE_MIN or score_max != SCORE_MAX:
            raise ValueError("JUDGEBERT targets require dataset score bounds 1-10")

        return cls(
            backbone=backbone.strip(),
            max_seq_length=max_seq_length,
            dropout=dropout,
            head_hidden_size=hidden,
            pooling=pooling,
            target_scaling=scaling,
            loss=loss,
            smooth_l1_beta=beta,
            clip_predictions=clip,
        )


@dataclass
class JudgeBertOutput:
    """Structured result from one cross-encoder forward pass."""

    scores: Tensor
    logits: Tensor
    pooled_output: Tensor
    loss: Tensor | None = None
    attentions: Any | None = None

    @property
    def predictions(self) -> Tensor:
        """Alias for externally meaningful 1--10 scores."""
        return self.scores


class JudgeBertModel(nn.Module):
    """Regression head over a paired-text transformer encoder."""

    def __init__(
        self,
        settings: JudgeBertConfig,
        *,
        encoder: nn.Module | None = None,
        allow_download: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(allow_download, bool):
            raise ValueError("allow_download must be a boolean")
        self.settings = settings
        self.allow_download = allow_download
        self.encoder = encoder
        self.regression_head: nn.Sequential | None = None
        if encoder is not None:
            self.regression_head = self._build_regression_head(
                self._encoder_hidden_size(encoder)
            )

    @classmethod
    def from_config(
        cls,
        cfg: Mapping[str, Any],
        *,
        encoder: nn.Module | None = None,
        allow_download: bool = False,
    ) -> "JudgeBertModel":
        """Construct from project config; injected encoders keep tests offline."""
        settings = JudgeBertConfig.from_project_config(cfg)
        return cls(
            settings,
            encoder=encoder,
            allow_download=allow_download,
        )

    @staticmethod
    def _encoder_hidden_size(encoder: nn.Module) -> int:
        hidden_size = getattr(getattr(encoder, "config", None), "hidden_size", None)
        if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError("Encoder config.hidden_size must be a positive integer")
        return hidden_size

    def _ensure_encoder(self) -> nn.Module:
        if self.encoder is not None:
            return self.encoder

        local_files_only = not self.allow_download
        try:
            with _huggingface_offline_guard(local_files_only):
                auto_model = _load_auto_model_class()
                loaded = auto_model.from_pretrained(
                    self.settings.backbone,
                    local_files_only=local_files_only,
                    trust_remote_code=False,
                )
            if not isinstance(loaded, nn.Module):
                raise TypeError("AutoModel.from_pretrained did not return a torch module")
            self._encoder_hidden_size(loaded)
        except Exception as exc:
            if local_files_only:
                message = (
                    f"{MODEL_IDENTITY} backbone {self.settings.backbone!r} is unavailable "
                    "with local_files_only=True; cache the exact model locally or pass "
                    "allow_download=True explicitly. No fallback model will be used. "
                    f"Original error: {exc}"
                )
            else:
                message = (
                    f"{MODEL_IDENTITY} backbone {self.settings.backbone!r} is unavailable "
                    "with local_files_only=False. No fallback model will be used. "
                    f"Original error: {exc}"
                )
            raise RuntimeError(message) from exc

        loaded.train(self.training)
        self.encoder = loaded
        return loaded

    def _ensure_regression_head(self, encoder: nn.Module) -> nn.Sequential:
        if self.regression_head is None:
            self.regression_head = self._build_regression_head(
                self._encoder_hidden_size(encoder)
            )
            self.regression_head.train(self.training)
        return self.regression_head

    def _build_regression_head(self, hidden_size: int) -> nn.Sequential:
        if self.settings.head_hidden_size == 0:
            return nn.Sequential(
                nn.Dropout(self.settings.dropout),
                nn.Linear(hidden_size, 1),
            )
        return nn.Sequential(
            nn.Dropout(self.settings.dropout),
            nn.Linear(hidden_size, self.settings.head_hidden_size),
            nn.GELU(),
            nn.Dropout(self.settings.dropout),
            nn.Linear(self.settings.head_hidden_size, 1),
        )

    def _validate_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None,
    ) -> None:
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise ValueError("input_ids must be a two-dimensional tensor")
        if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must contain a non-empty batch and sequence")
        if input_ids.shape[1] > self.settings.max_seq_length:
            raise ValueError(
                "input_ids sequence length exceeds models.judgebert.max_seq_length "
                f"({self.settings.max_seq_length})"
            )
        if input_ids.is_floating_point() or input_ids.is_complex():
            raise ValueError("input_ids must contain integer token IDs")
        if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 2:
            raise ValueError("attention_mask must be a two-dimensional tensor")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        binary_mask = (attention_mask == 0) | (attention_mask == 1)
        if not bool(binary_mask.all()):
            raise ValueError("attention_mask must contain only 0 and 1 values")
        if bool((attention_mask.sum(dim=1) == 0).any()):
            raise ValueError("attention_mask must retain at least one token per row")
        if token_type_ids is not None:
            if not isinstance(token_type_ids, Tensor) or token_type_ids.ndim != 2:
                raise ValueError("token_type_ids must be a two-dimensional tensor")
            if token_type_ids.shape != input_ids.shape:
                raise ValueError("token_type_ids must have the same shape as input_ids")
            if token_type_ids.is_floating_point() or token_type_ids.is_complex():
                raise ValueError("token_type_ids must contain integer segment IDs")

    @staticmethod
    def _validated_labels(labels: Tensor | Any, batch_size: int) -> Tensor:
        if isinstance(labels, Tensor) and labels.dtype == torch.bool:
            raise ValueError("Labels must be real-valued 1-10 scores, not booleans")
        tensor = _floating_tensor(labels, name="Labels")
        valid_shape = tensor.shape == (batch_size,) or tensor.shape == (batch_size, 1)
        if not valid_shape:
            raise ValueError(
                "Labels must have shape (batch_size,) or (batch_size, 1); "
                f"got {tuple(tensor.shape)} for batch size {batch_size}"
            )
        tensor = tensor.reshape(batch_size)
        if bool(((tensor < SCORE_MIN) | (tensor > SCORE_MAX)).any()):
            raise ValueError("Labels must be within the finite 1-10 LMP score range")
        return tensor

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        output_attentions: bool = False,
    ) -> JudgeBertOutput:
        self._validate_inputs(input_ids, attention_mask, token_type_ids)
        validated_labels = (
            self._validated_labels(labels, input_ids.shape[0])
            if labels is not None
            else None
        )
        encoder_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_attentions": bool(output_attentions),
            "return_dict": True,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        encoder = self._ensure_encoder()
        encoded = encoder(**encoder_kwargs)
        hidden = encoded.last_hidden_state
        if self.settings.pooling == "cls":
            pooled = hidden[:, 0, :]
        else:
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        regression_head = self._ensure_regression_head(encoder)
        logits = regression_head(pooled).squeeze(-1)
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(f"{MODEL_IDENTITY} produced non-finite regression output")
        if self.settings.target_scaling == "minmax":
            scores = logits * (SCORE_MAX - SCORE_MIN) + SCORE_MIN
        else:
            scores = logits
        if self.settings.clip_predictions:
            scores = scores.clamp(min=SCORE_MIN, max=SCORE_MAX)

        loss = None
        if validated_labels is not None:
            targets = (
                scale_targets(validated_labels)
                if self.settings.target_scaling == "minmax"
                else validated_labels
            ).to(device=logits.device, dtype=logits.dtype)
            if self.settings.loss == "smooth_l1":
                loss = F.smooth_l1_loss(logits, targets, beta=self.settings.smooth_l1_beta)
            else:
                loss = F.mse_loss(logits, targets)

        attentions = getattr(encoded, "attentions", None) if output_attentions else None
        return JudgeBertOutput(
            scores=scores,
            logits=logits,
            pooled_output=pooled,
            loss=loss,
            attentions=attentions,
        )

    @torch.no_grad()
    def predict_scores(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
    ) -> Tensor:
        """Return detached finite scores, using eval mode for deterministic dropout."""
        was_training = self.training
        try:
            self.eval()
            scores = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).scores.detach()
        finally:
            self.train(was_training)
        if not bool(torch.isfinite(scores).all()):  # defensive; forward checks logits
            raise RuntimeError(f"{MODEL_IDENTITY} produced non-finite score predictions")
        return scores


JudgeBERTModel = JudgeBertModel

__all__ = [
    "MODEL_IDENTITY",
    "JudgeBERTModel",
    "JudgeBertConfig",
    "JudgeBertModel",
    "JudgeBertOutput",
    "scale_targets",
    "unscale_targets",
]
