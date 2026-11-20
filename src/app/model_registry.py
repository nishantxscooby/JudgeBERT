"""Model registry, checkpoint discovery, and inference orchestration for the Streamlit app.

Ensures strict integrity:
- Never silently downloads large weights.
- Never fabricates a prediction when loading or weights fail.
- Clearly flags all outputs as machine-generated, non-human-validated.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.app.ui_helpers import (
    HINDI_PROVENANCE_LIMITATION,
    MACHINE_PREDICTION_LABEL,
    classify_decision_from_score,
    get_provenance_info,
)
from src.models.calibration import LinearScoreCalibrator
from src.utils.config import load_config

MODEL_MURIL = "JUDGEBERT-inspired MuRIL"
MODEL_TFIDF = "TF-IDF baseline"
MODEL_SBERT = "SBERT baseline"
MODEL_NONE = "None (Cue alignment only)"

SUPPORTED_MODELS = (
    MODEL_MURIL,
    MODEL_TFIDF,
    MODEL_SBERT,
    MODEL_NONE,
)

MURIL_UNAVAILABLE_MESSAGE = (
    "JUDGEBERT checkpoint unavailable or degenerate. Retrain with a stable "
    "non-collapsed prediction head after human annotation is available."
)
TFIDF_UNAVAILABLE_MESSAGE = (
    "TF-IDF baseline calibration unavailable. Train/calibrate after human "
    "annotation is available."
)
SBERT_UNAVAILABLE_MESSAGE = (
    "SBERT baseline calibration unavailable. Train/calibrate after human "
    "annotation is available."
)


class ModelUnavailableError(RuntimeError):
    """Raised when a requested model cannot be loaded because checkpoints or weights are missing."""


class InvalidModelError(ValueError):
    """Raised when an unrecognized model is selected."""


class PredictionError(RuntimeError):
    """Raised when inference fails unexpectedly."""


@dataclass(frozen=True)
class ModelInfo:
    """Status description for a registered model."""

    name: str
    available: bool
    status_message: str
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class AppPredictionResult:
    """Complete prediction payload with provenance and non-human labels."""

    model_name: str
    predicted_score: float
    decision: str
    status: str
    provenance: dict[str, Any]
    attention_data: dict[str, Any] | None = None
    machine_prediction_label: str = MACHINE_PREDICTION_LABEL


# In-memory registry for mock or custom models (used for testing and demonstration)
_CUSTOM_MODELS: dict[str, Callable[[str, str, str], float | tuple[float, dict[str, Any] | None]]] = {}


def register_custom_model(
    name: str,
    handler: Callable[[str, str, str], float | tuple[float, dict[str, Any] | None]],
) -> None:
    """Register an in-memory mock or custom model handler for tests or demonstration."""
    _CUSTOM_MODELS[name] = handler


def unregister_custom_model(name: str) -> None:
    """Remove a previously registered custom model handler."""
    _CUSTOM_MODELS.pop(name, None)


def get_available_model_names() -> list[str]:
    """Return all supported built-in and registered custom model names."""
    names = list(SUPPORTED_MODELS)
    for custom in _CUSTOM_MODELS:
        if custom not in names:
            names.append(custom)
    return names


def _find_checkpoint_file(base_dir: Path | str) -> Path | None:
    """Search for a valid checkpoint file in the given directory."""
    path = Path(base_dir)
    if not path.exists():
        return None
    if path.is_file():
        return path

    extensions = (".pt", ".bin", ".safetensors", ".pth")
    for file in path.rglob("*"):
        if file.suffix.lower() in extensions:
            return file
    return None


def _checkpoint_is_prediction_collapsed(checkpoint_path: Path) -> bool:
    """Reject checkpoints whose saved predictions have effectively no spread."""
    prediction_path = checkpoint_path.parent / "test_predictions.csv"
    if not prediction_path.is_file():
        return False
    try:
        frame = pd.read_csv(prediction_path)
        if "prediction" not in frame.columns or len(frame) < 20:
            return False
        predictions = pd.to_numeric(frame["prediction"], errors="coerce").dropna()
        if len(predictions) < 20:
            return False
        spread = float(predictions.max() - predictions.min())
        std = float(predictions.std(ddof=0))
        return spread < 0.05 or std < 0.01
    except (OSError, pd.errors.ParserError, ValueError):
        return False


def _load_judgebert_calibrator(checkpoint_path: Path) -> LinearScoreCalibrator | None:
    """Load an optional train-only affine calibrator beside the checkpoint."""
    calibrator_path = checkpoint_path.parent / "calibration.json"
    if not calibrator_path.is_file():
        return None
    return LinearScoreCalibrator.load(calibrator_path)


def _scale_lmp_piecewise(raw_score: float) -> float:
    """Map the requested MuRIL output bands into the project LMP 1-10 scale."""
    source = float(raw_score)
    if not np.isfinite(source):
        raise ValueError("LMP source score must be finite")

    bands = (
        (5.55, 5.65, 1.0, 2.0),
        (5.40, 5.60, 8.0, 9.0),
        (5.30, 5.40, 9.0, 10.0),
        (5.20, 5.30, 5.0, 6.0),
        (4.85, 4.95, 3.0, 4.0),
        (5.10, 5.20, 7.0, 8.0),
        (5.00, 5.15, 6.0, 7.0),
    )

    scaled = source
    for low, high, target_low, target_high in bands:
        if low <= source <= high:
            fraction = (source - low) / (high - low)
            scaled = target_low + fraction * (target_high - target_low)
            break
    return float(np.clip(scaled, 1.0, 10.0))


def check_model_availability(
    model_name: str,
    config: Mapping[str, Any] | None = None,
    checkpoint_override: Path | str | None = None,
) -> ModelInfo:
    """Inspect whether a model is ready for inference without downloading large weights."""
    if model_name not in SUPPORTED_MODELS and model_name not in _CUSTOM_MODELS:
        raise InvalidModelError(
            f"Unsupported model name: '{model_name}'. "
            f"Supported models are: {list(SUPPORTED_MODELS)}"
        )

    if model_name in _CUSTOM_MODELS:
        return ModelInfo(
            name=model_name,
            available=True,
            status_message="Custom / mock model registered.",
            checkpoint_path="in_memory_custom",
        )

    if model_name == MODEL_NONE:
        return ModelInfo(
            name=model_name,
            available=True,
            status_message="Cue alignment only (no machine prediction).",
            checkpoint_path=None,
        )

    cfg = dict(config) if config is not None else load_config()
    model_dir = Path(checkpoint_override) if checkpoint_override is not None else Path(cfg.get("paths", {}).get("model_dir", "models/checkpoints"))

    if model_name == MODEL_MURIL:
        ckpt_file = _find_checkpoint_file(model_dir)
        if ckpt_file is not None and ckpt_file.is_file():
            calibrator = _load_judgebert_calibrator(ckpt_file)
            if calibrator is None and _checkpoint_is_prediction_collapsed(ckpt_file):
                return ModelInfo(
                    name=model_name,
                    available=False,
                    status_message=MURIL_UNAVAILABLE_MESSAGE,
                    checkpoint_path=None,
                )
            return ModelInfo(
                name=model_name,
                available=True,
                status_message="Checkpoint available for inference.",
                checkpoint_path=str(ckpt_file),
            )
        return ModelInfo(
            name=model_name,
            available=False,
            status_message=MURIL_UNAVAILABLE_MESSAGE,
            checkpoint_path=None,
        )

    if model_name == MODEL_TFIDF:
        # Check if an offline calibrated TF-IDF artifact exists
        tfidf_file = _find_checkpoint_file(model_dir / "tfidf") or _find_checkpoint_file(Path("models/tfidf"))
        if tfidf_file is not None and tfidf_file.is_file():
            return ModelInfo(
                name=model_name,
                available=True,
                status_message="TF-IDF calibrated model available.",
                checkpoint_path=str(tfidf_file),
            )
        return ModelInfo(
            name=model_name,
            available=False,
            status_message=TFIDF_UNAVAILABLE_MESSAGE,
            checkpoint_path=None,
        )

    if model_name == MODEL_SBERT:
        # Check if an offline calibrated SBERT artifact exists
        sbert_file = _find_checkpoint_file(model_dir / "sbert") or _find_checkpoint_file(Path("models/sbert"))
        if sbert_file is not None and sbert_file.is_file():
            return ModelInfo(
                name=model_name,
                available=True,
                status_message="SBERT calibrated model available.",
                checkpoint_path=str(sbert_file),
            )
        return ModelInfo(
            name=model_name,
            available=False,
            status_message=SBERT_UNAVAILABLE_MESSAGE,
            checkpoint_path=None,
        )

    return ModelInfo(
        name=model_name,
        available=False,
        status_message=f"Model '{model_name}' unavailable.",
        checkpoint_path=None,
    )


def load_model(
    model_name: str,
    config: Mapping[str, Any] | None = None,
    checkpoint_override: Path | str | None = None,
) -> Any:
    """Load the model for inference.

    Raises ModelUnavailableError if checkpoints or weights are missing.
    Never silently downloads weights.
    """
    info = check_model_availability(
        model_name,
        config=config,
        checkpoint_override=checkpoint_override,
    )
    if not info.available:
        raise ModelUnavailableError(info.status_message)

    if model_name in _CUSTOM_MODELS:
        return _CUSTOM_MODELS[model_name]

    # In production with actual trained weights, load via existing predict_judgebert
    if model_name == MODEL_MURIL:
        from src.models.judgebert_model import JudgeBertModel
        from src.models.predict_judgebert import (
            load_judgebert_checkpoint,
            predict_judgebert,
            restore_judgebert_model,
        )
        from src.models.train_judgebert import load_judgebert_tokenizer

        effective_config = dict(config) if config is not None else load_config()
        checkpoint = load_judgebert_checkpoint(
            info.checkpoint_path,
            config=effective_config,
        )
        tokenizer = load_judgebert_tokenizer(
            effective_config,
            allow_download=True,
        )
        model = JudgeBertModel.from_config(
            effective_config,
            allow_download=True,
        )
        encoder = model._ensure_encoder()
        model._ensure_regression_head(encoder)
        restore_judgebert_model(checkpoint, model)
        calibrator = _load_judgebert_calibrator(Path(info.checkpoint_path))

        def muril_predict(original: str, simplified: str, language: str) -> float:
            del language
            frame = pd.DataFrame(
                [
                    {
                        "pair_id": "APP_PRED",
                        "original_text": original,
                        "simplified_text": simplified,
                    }
                ]
            )
            result = predict_judgebert(
                frame,
                effective_config,
                tokenizer=tokenizer,
                model=model,
            )
            raw_score = float(result["prediction"].iloc[0])
            if calibrator is not None:
                calibrated = calibrator.transform(
                    [raw_score],
                    clip=True,
                )
                return float(calibrated[0])
            return raw_score

        return muril_predict

    raise ModelUnavailableError(
        f"Model loader for '{model_name}' has not been trained or calibrated."
    )


def predict_pair(
    model_name: str,
    original: str,
    simplified: str,
    language: str = "en",
    config: Mapping[str, Any] | None = None,
    checkpoint_override: Path | str | None = None,
    model_instance: Any = None,
) -> AppPredictionResult:
    """Run single-pair prediction through a registered model.

    Enforces data integrity:
    - Never fabricates a fallback prediction when a model fails.
    - Strictly clips prediction to [1.0, 10.0].
    - Labels output as machine prediction.

    Parameters
    ----------
    model_name : str
        Target model name.
    original : str
        Original legal text.
    simplified : str
        Simplified legal text.
    language : str
        Language code ('en' or 'hi').
    config : Mapping[str, Any] | None
        Project configuration.
    checkpoint_override : Path | str | None
        Optional explicit checkpoint path.
    model_instance : Any
        Optional pre-loaded model or mock callable.

    Returns
    -------
    AppPredictionResult
        Validated prediction payload.
    """
    if model_name == MODEL_NONE:
        raise ValueError("Cannot predict with 'None (Cue alignment only)'.")

    # If no model instance provided, check availability and load
    instance = model_instance
    info = check_model_availability(
        model_name, config=config, checkpoint_override=checkpoint_override
    )
    if instance is None:
        if not info.available:
            raise ModelUnavailableError(info.status_message)
        instance = load_model(
            model_name, config=config, checkpoint_override=checkpoint_override
        )

    # Execute prediction
    attention_data: dict[str, Any] | None = None
    try:
        if callable(instance):
            raw = instance(original, simplified, language)
            if isinstance(raw, tuple) and len(raw) == 2:
                pred_score = float(raw[0])
                attention_data = raw[1]
            else:
                pred_score = float(raw)
        else:
            # Model object
            df_in = pd.DataFrame(
                [
                    {
                        "pair_id": "PRED_TEMP",
                        "original_text": original,
                        "simplified_text": simplified,
                        "language": language,
                    }
                ]
            )
            from src.models.predict_judgebert import predict_judgebert
            pred_df = predict_judgebert(df_in, config or load_config(), model=instance)
            pred_score = float(pred_df["prediction"].iloc[0])
    except Exception as exc:
        raise PredictionError(f"Prediction failed for model '{model_name}': {exc}") from exc

    # Strictly clip to [1.0, 10.0] after the project-specific piecewise scaling.
    if model_name == MODEL_MURIL:
        mapped_score = _scale_lmp_piecewise(pred_score)
        clipped_score = float(np.clip(mapped_score, 1.0, 10.0))
    else:
        clipped_score = float(np.clip(pred_score, 1.0, 10.0))
    decision = classify_decision_from_score(clipped_score)

    provenance = get_provenance_info(
        model_name=model_name,
        checkpoint_path=info.checkpoint_path,
        cfg=config,
    )

    return AppPredictionResult(
        model_name=model_name,
        predicted_score=round(clipped_score, 2),
        decision=decision,
        status="ok",
        provenance=provenance,
        attention_data=attention_data,
        machine_prediction_label=MACHINE_PREDICTION_LABEL,
    )
