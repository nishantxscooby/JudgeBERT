"""Provenance and safe artifact-writing helpers for model results."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.data import load_data, schema
from src.utils import io as io_utils
from src.utils.io import provenance


@dataclass(frozen=True)
class ResultPaths:
    """Paths written for one model-result bundle."""

    predictions_csv: Path
    metadata_json: Path


def _validated_pair_ids(pair_ids: Iterable[Any], *, context: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in pair_ids:
        if value is None or pd.isna(value):
            raise ValueError(f"{context} contains a null or blank pair_id")
        pair_id = str(value).strip()
        if not pair_id:
            raise ValueError(f"{context} contains a null or blank pair_id")
        if pair_id in seen:
            duplicates.add(pair_id)
        seen.add(pair_id)
        ids.append(pair_id)
    if duplicates:
        raise ValueError(
            f"Duplicate pair_id value(s) in {context}: {sorted(duplicates)[:5]}"
        )
    return ids


def _pair_id_digest(ids: Iterable[str]) -> str:
    payload = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of an existing artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash an ordered frame using a stable UTF-8 CSV representation."""
    payload = frame.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_csv(frame: pd.DataFrame, destination: str | Path) -> Path:
    """Publish a CSV via same-directory replacement without following hard links."""
    target = Path(destination)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        io_utils.write_csv(frame, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_json(value: Mapping[str, Any], destination: str | Path) -> Path:
    """Publish JSON via same-directory replacement without following hard links."""
    target = Path(destination)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        io_utils.write_json(dict(value), temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def source_tree_sha256(root: str | Path | None = None) -> str:
    """Hash Python source paths and bytes when no Git revision is available."""
    source_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else Path(__file__).resolve().parents[1]
    )
    files = sorted(source_root.rglob("*.py"), key=lambda path: path.as_posix())
    if not files:
        raise ValueError(f"No Python source files found under {source_root}")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_output_directory(
    path: str | Path,
    *,
    allowed_root: str | Path | None = None,
) -> Path:
    """Create an output directory, optionally confined beneath a root."""
    target = Path(path).expanduser().resolve()
    if allowed_root is not None:
        root = Path(allowed_root).expanduser().resolve()
        if not target.is_relative_to(root):
            raise ValueError(
                f"Output directory {target} is outside allowed root {root}"
            )
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Output path exists and is not a directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def validate_prediction_output_paths(
    output_path: str | Path,
    cfg: Mapping[str, Any],
    *,
    model_name: str,
    input_paths: Iterable[str | Path] = (),
) -> tuple[Path, Path]:
    """Reject prediction or sidecar paths that alias authoritative inputs."""
    destination = Path(output_path)
    metadata_path = destination.with_suffix(".metadata.json")
    protected: dict[Path, str] = {
        load_data.master_path(dict(cfg)).expanduser().resolve(): (
            "authoritative dataset master"
        )
    }
    for input_path in input_paths:
        protected[Path(input_path).expanduser().resolve()] = "input artifact"

    for kind, candidate in (
        ("prediction", destination),
        ("sidecar", metadata_path),
    ):
        resolved = candidate.expanduser().resolve()
        protected_kind = protected.get(resolved)
        if protected_kind is not None:
            raise ValueError(
                f"{model_name} {kind} output path must not overwrite "
                f"{protected_kind}: {resolved}"
            )
        for protected_path, protected_kind in protected.items():
            if (
                resolved.exists()
                and protected_path.exists()
                and resolved.samefile(protected_path)
            ):
                raise ValueError(
                    f"{model_name} {kind} output must not be a hard-link alias of "
                    f"{protected_kind}: {protected_path}"
                )
        if candidate.exists() and candidate.is_dir():
            raise IsADirectoryError(f"{model_name} output path is a directory: {candidate}")
    return destination, metadata_path


def write_result_bundle(
    predictions: pd.DataFrame,
    metadata: Mapping[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "model",
    allowed_root: str | Path | None = None,
) -> ResultPaths:
    """Write label-free predictions and their metadata with project IO helpers."""
    if Path(stem).name != stem or stem in {"", ".", ".."}:
        raise ValueError(f"Unsafe result filename stem: {stem!r}")
    required = {"pair_id", "prediction"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction output is missing columns: {missing}")

    ids = _validated_pair_ids(
        predictions["pair_id"].tolist(), context="prediction output"
    )
    scores = pd.to_numeric(predictions["prediction"], errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(scores).all():
        raise ValueError("Prediction output scores must all be finite")
    if ((scores < schema.SCORE_MIN) | (scores > schema.SCORE_MAX)).any():
        raise ValueError(
            f"Prediction output scores must be within "
            f"{schema.SCORE_MIN}-{schema.SCORE_MAX}"
        )
    expected_count = metadata.get("n_predictions")
    expected_digest = metadata.get("pair_id_sha256")
    if expected_count is not None and (
        isinstance(expected_count, bool) or not isinstance(expected_count, int)
    ):
        raise ValueError("Result metadata has an invalid prediction count")
    count_mismatch = expected_count is not None and expected_count != len(ids)
    digest_mismatch = (
        expected_digest is not None
        and str(expected_digest) != _pair_id_digest(ids)
    )
    if count_mismatch or digest_mismatch:
        raise ValueError("Result metadata does not match prediction pair-ID provenance")

    clean = pd.DataFrame({"pair_id": ids, "prediction": scores})
    directory = ensure_output_directory(output_dir, allowed_root=allowed_root)
    prediction_path = directory / f"{stem}_predictions.csv"
    metadata_path = directory / f"{stem}_metadata.json"
    atomic_write_csv(clean, prediction_path)
    atomic_write_json(metadata, metadata_path)
    return ResultPaths(prediction_path, metadata_path)


def build_result_metadata(
    cfg: Mapping[str, Any],
    *,
    model_name: str,
    run_mode: str,
    pair_ids: Iterable[str],
    splits: Iterable[str],
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Build reproducibility metadata for a prediction artifact."""
    ids = _validated_pair_ids(pair_ids, context="result provenance")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Result metadata seed must be a non-bool integer")
    effective_seed = seed
    return {
        "provenance": provenance(
            cfg,
            artifact="model_predictions",
            seed=effective_seed,
        ),
        "model_name": model_name,
        "run_mode": run_mode,
        "seed": effective_seed,
        "device": str(device),
        "n_predictions": len(ids),
        "splits": list(dict.fromkeys(str(split_name) for split_name in splits)),
        "pair_id_sha256": _pair_id_digest(ids),
        "code_sha256": source_tree_sha256(),
    }
