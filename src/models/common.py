"""Shared configuration, data, device, and prediction safeguards."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from src.data import load_data, schema
from src.models.provenance import ensure_output_directory
from src.utils import config as config_utils
from src.utils.config import load_config

if TYPE_CHECKING:
    import torch

RunMode = Literal["smoke", "final"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True)
class ModelRunContext:
    """Validated, seeded inputs shared by every model implementation."""

    config: dict[str, Any]
    corpus: pd.DataFrame
    frames: dict[str, pd.DataFrame]
    mode: RunMode
    seed: int
    device: "torch.device"
    output_dir: Path


def _normalised_ids(frame: pd.DataFrame, *, context: str) -> pd.Series:
    if "pair_id" not in frame.columns:
        raise ValueError(f"{context} is missing required column 'pair_id'")
    ids = frame["pair_id"]
    blank = ids.isna() | ids.astype("string").str.strip().eq("").fillna(True)
    if blank.any():
        raise ValueError(
            f"{context} contains {int(blank.sum())} null or blank pair_id value(s)"
        )
    normalised = ids.astype(str).str.strip()
    duplicates = normalised[normalised.duplicated(keep=False)]
    if not duplicates.empty:
        examples = sorted(duplicates.unique())[:5]
        raise ValueError(f"Duplicate pair_id value(s) in {context}: {examples}")
    return normalised


def validate_model_corpus(frame: pd.DataFrame) -> None:
    """Validate identifiers, split vocabulary, and anti-leakage grouping."""
    _normalised_ids(frame, context="model corpus")
    for column in ("split", "source_group"):
        if column not in frame.columns:
            raise ValueError(f"Model corpus is missing required column {column!r}")

    splits = frame["split"]
    unknown = sorted(
        set(splits.dropna().astype(str).str.strip()) - set(schema.SPLITS)
    )
    if splits.isna().any() or unknown:
        detail = unknown or ["<null>"]
        raise ValueError(f"Unknown split value(s): {detail}; expected {schema.SPLITS}")

    groups = frame["source_group"]
    blank_groups = groups.isna() | groups.astype("string").str.strip().eq("").fillna(True)
    if blank_groups.any():
        raise ValueError("Model corpus contains null or blank source_group value(s)")
    spans = frame.assign(source_group=groups.astype(str).str.strip()).groupby(
        "source_group", sort=True
    )["split"].nunique()
    crossing = spans[spans > 1].index.astype(str).tolist()
    if crossing:
        raise ValueError(
            "source_group value(s) cross multiple splits: "
            f"{crossing[:5]}"
        )


def require_gold(frame: pd.DataFrame, context: str = "model run") -> None:
    """Apply the authoritative gold gate, then require complete score labels."""
    load_data.assert_gold(frame, context)
    if "annotation_status" not in frame.columns:
        raise ValueError(
            f"Gold data for {context} is missing required column 'annotation_status'"
        )
    if "human_lmp_score" not in frame.columns or frame["human_lmp_score"].isna().any():
        raise ValueError(
            f"Gold data for {context} must contain human_lmp_score for every row"
        )
    if not schema.is_gold(frame).all():
        raise ValueError(
            f"Gold data for {context} contains rows that do not satisfy schema.is_gold"
        )
    scores = pd.to_numeric(frame["human_lmp_score"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(scores).all():
        raise ValueError(f"Gold human_lmp_score values for {context} must be finite")
    if ((scores < schema.SCORE_MIN) | (scores > schema.SCORE_MAX)).any():
        raise ValueError(
            f"Gold human_lmp_score values for {context} must be within "
            f"{schema.SCORE_MIN}-{schema.SCORE_MAX}"
        )


def select_device(requested: DeviceName | str = "auto") -> "torch.device":
    """Resolve CPU/CUDA/MPS, failing if a requested accelerator is absent."""
    import torch

    valid = {"auto", "cpu", "cuda", "mps"}
    if requested not in valid:
        raise ValueError(
            f"Unknown device {requested!r}; expected auto/cpu/cuda/mps"
        )

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(
        mps_backend is not None and mps_backend.is_available()
    )

    if requested == "auto":
        requested = "cuda" if cuda_available else "mps" if mps_available else "cpu"
    elif requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    elif requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available")

    return torch.device(requested)


def setup_model_seed(
    cfg: Mapping[str, Any],
    seed: int | None = None,
    *,
    deterministic: bool | None = None,
) -> int:
    """Set all supported random seeds and return the effective seed."""
    training = cfg.get("training") or {}
    project = cfg.get("project") or {}
    effective_seed = int(
        seed if seed is not None else training.get("seed", project.get("seed", 42))
    )
    effective_deterministic = bool(
        training.get("deterministic", True)
        if deterministic is None
        else deterministic
    )
    config_utils.set_global_seed(
        effective_seed, deterministic=effective_deterministic
    )
    return effective_seed


def validate_predictions(
    predictions: pd.DataFrame,
    expected_pair_ids: Iterable[Any],
    *,
    prediction_column: str = "prediction",
    clip: bool = False,
    score_min: float = float(schema.SCORE_MIN),
    score_max: float = float(schema.SCORE_MAX),
) -> pd.DataFrame:
    """Validate, range-check, and align one score per expected pair ID."""
    if prediction_column not in predictions.columns:
        raise ValueError(f"Predictions are missing column {prediction_column!r}")
    if not np.isfinite([score_min, score_max]).all() or score_min >= score_max:
        raise ValueError("Prediction score bounds must be finite and increasing")

    expected = pd.DataFrame({"pair_id": list(expected_pair_ids)})
    expected_ids = _normalised_ids(expected, context="expected prediction IDs")
    prediction_ids = _normalised_ids(predictions, context="predictions")

    expected_set = set(expected_ids)
    prediction_set = set(prediction_ids)
    missing = sorted(expected_set - prediction_set)
    unknown = sorted(prediction_set - expected_set)
    id_problems = []
    if missing:
        id_problems.append(f"Missing expected prediction IDs: {missing[:10]}")
    if unknown:
        id_problems.append(f"Unknown prediction IDs: {unknown[:10]}")
    if id_problems:
        raise ValueError("; ".join(id_problems))

    scores = pd.to_numeric(predictions[prediction_column], errors="coerce")
    finite = np.isfinite(scores.to_numpy(dtype=float, na_value=np.nan))
    if not finite.all():
        bad_ids = prediction_ids.loc[~finite].tolist()[:10]
        raise ValueError(f"Prediction scores must be finite; invalid IDs: {bad_ids}")

    outside = (scores < score_min) | (scores > score_max)
    if outside.any() and not clip:
        bad = prediction_ids.loc[outside].tolist()[:10]
        raise ValueError(
            f"Prediction scores outside {score_min:g}-{score_max:g} for IDs: {bad}. "
            "Pass clip=True to clip explicitly."
        )
    if clip:
        scores = scores.clip(lower=score_min, upper=score_max)

    work = pd.DataFrame(
        {
            "pair_id": prediction_ids,
            "prediction": scores.astype(float),
        }
    )
    aligned = work.set_index("pair_id").loc[expected_ids].reset_index()
    return aligned[["pair_id", "prediction"]]


def load_model_config(
    path: str | PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate the effective project configuration."""
    return load_config(path, overrides=overrides)


def load_model_corpus(
    corpus_path: str | Path | None = None,
    *,
    config_path: str | PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    mode: RunMode = "final",
    splits: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load the authoritative corpus and retain only requested splits."""
    if mode not in {"smoke", "final"}:
        raise ValueError(f"Unknown run mode {mode!r}; expected smoke/final")

    requested = tuple(splits) if splits is not None else None
    unknown = sorted(set(requested or ()) - set(schema.SPLITS))
    if unknown:
        raise ValueError(f"Unknown split(s): {unknown}; expected {schema.SPLITS}")

    cfg = load_model_config(config_path, overrides)
    subset = "gold" if mode == "final" else "all"
    corpus = load_data.load_master(path=corpus_path, cfg=cfg, subset=subset)
    if mode == "final":
        require_gold(corpus, context="final model run")
    validate_model_corpus(corpus)
    selected = load_data.filter_rows(corpus, splits=requested).reset_index(drop=True)
    return selected


def prepare_model_run(
    *,
    model_name: str,
    mode: RunMode = "final",
    corpus_path: str | Path | None = None,
    config_path: str | PathLike[str] | None = None,
    overrides: Mapping[str, Any] | None = None,
    splits: Iterable[str] | None = None,
    seed: int | None = None,
    device: DeviceName | str | None = None,
    output_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> ModelRunContext:
    """Prepare a reproducible run, creating outputs only after all gates pass."""
    cfg = load_model_config(config_path, overrides)
    effective_seed = setup_model_seed(cfg, seed)
    requested_device = device or str((cfg.get("training") or {}).get("device", "auto"))
    resolved_device = select_device(requested_device)
    corpus = load_model_corpus(
        corpus_path,
        config_path=config_path,
        overrides=overrides,
        mode=mode,
        splits=splits,
    )
    frames = load_data.split_frames(corpus)

    if output_dir is None:
        if Path(model_name).name != model_name or model_name in {"", ".", ".."}:
            raise ValueError(f"Unsafe model name for output directory: {model_name!r}")
        results_root = config_utils.resolve_path(cfg, "results_dir")
        destination = results_root / model_name
        allowed_root = results_root
    else:
        destination = Path(output_dir)
        allowed_root = Path(output_root) if output_root is not None else None
    created_output = ensure_output_directory(
        destination, allowed_root=allowed_root
    )

    return ModelRunContext(
        config=cfg,
        corpus=corpus,
        frames=frames,
        mode=mode,
        seed=effective_seed,
        device=resolved_device,
        output_dir=created_output,
    )
