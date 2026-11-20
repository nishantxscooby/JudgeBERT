"""Strict pair-ID alignment between canonical gold rows and predictions."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.data import schema

_METADATA_COLUMNS = ("language", "legal_domain", "source_group", "split")
_REQUIRED_PREDICTION_METADATA = (
    "language",
    "legal_domain",
    "split",
)
_OUTPUT_COLUMNS = (
    "pair_id",
    "language",
    "legal_domain",
    "source_group",
    "split",
    "human_lmp_score",
    "decision",
    "annotation_status",
)
_FORBIDDEN_PREDICTION_COLUMNS = frozenset(
    {
        "human_lmp_score",
        "draft_lmp_score",
        "decision",
        "error_type",
        "draft_error_type",
        "annotation_status",
        "label",
    }
)


def _normalized_ids(frame: pd.DataFrame, *, context: str) -> pd.Series:
    if "pair_id" not in frame.columns:
        raise ValueError(f"{context} is missing required column 'pair_id'")
    values = frame["pair_id"]
    normalized = values.astype("string").str.strip()
    blank = values.isna() | normalized.eq("").fillna(True)
    if blank.any():
        raise ValueError(f"{context} contains null or blank pair_id values")
    normalized = normalized.astype(str)
    duplicated = normalized.duplicated(keep=False)
    if duplicated.any():
        examples = sorted(normalized.loc[duplicated].unique())[:5]
        raise ValueError(f"Duplicate pair_id values in {context}: {examples}")
    return normalized


def validate_source_group_splits(frame: pd.DataFrame) -> None:
    """Reject blank groups and any source group assigned to multiple splits."""
    missing = sorted({"source_group", "split"} - set(frame.columns))
    if missing:
        raise ValueError(f"Canonical pairs are missing columns: {missing}")
    groups = frame["source_group"]
    normalized_groups = groups.astype("string").str.strip()
    blank_groups = groups.isna() | normalized_groups.eq("").fillna(True)
    if blank_groups.any():
        raise ValueError("Canonical source_group values contain null or blank IDs")
    splits = frame["split"]
    normalized_splits = splits.astype("string").str.strip()
    bad_split = (
        splits.isna()
        | normalized_splits.eq("").fillna(True)
        | ~normalized_splits.isin(schema.SPLITS)
    )
    if bad_split.any():
        found = sorted(
            {"<null>" if pd.isna(value) else str(value).strip() for value in splits}
        )
        raise ValueError(f"Canonical pairs contain unknown split values: {found}")
    spans = pd.DataFrame(
        {"source_group": normalized_groups, "split": normalized_splits}
    ).groupby("source_group", sort=True)["split"].nunique()
    crossing = spans[spans > 1].index.astype(str).tolist()
    if crossing:
        raise ValueError(f"source_group values cross splits: {crossing[:10]}")


def align_predictions(
    canonical: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    prediction_column: str = "prediction",
    score_min: float = float(schema.SCORE_MIN),
    score_max: float = float(schema.SCORE_MAX),
    require_metadata: bool = True,
) -> pd.DataFrame:
    """Validate an exact prediction set and align it to canonical pair order."""
    if not isinstance(canonical, pd.DataFrame) or not isinstance(
        predictions, pd.DataFrame
    ):
        raise TypeError("canonical and predictions must be pandas DataFrames")
    if canonical.empty:
        raise ValueError("Canonical pairs must be nonempty")
    if predictions.empty:
        raise ValueError("Predictions must be nonempty")
    required_canonical = set(_OUTPUT_COLUMNS)
    missing_canonical = sorted(required_canonical - set(canonical.columns))
    if missing_canonical:
        raise ValueError(f"Canonical pairs are missing columns: {missing_canonical}")
    if prediction_column not in predictions.columns:
        raise ValueError(f"Predictions are missing required column {prediction_column!r}")
    leaked = sorted(
        str(column)
        for column in predictions.columns
        if (
            str(column).casefold() in _FORBIDDEN_PREDICTION_COLUMNS
            or str(column).casefold().startswith("draft_")
            or str(column).casefold().endswith("_label")
        )
    )
    if leaked:
        raise ValueError(f"Prediction artifact contains forbidden label columns: {leaked}")
    if require_metadata:
        missing_metadata = sorted(
            set(_REQUIRED_PREDICTION_METADATA) - set(predictions.columns)
        )
        if missing_metadata:
            raise ValueError(
                f"Predictions are missing required metadata columns: {missing_metadata}"
            )
    if not np.isfinite([score_min, score_max]).all() or score_min >= score_max:
        raise ValueError("Prediction score bounds must be finite and increasing")

    human_scores = pd.to_numeric(
        canonical["human_lmp_score"], errors="coerce"
    ).to_numpy(dtype=float, na_value=np.nan)
    invalid_human_score = (
        ~np.isfinite(human_scores)
        | (human_scores < schema.SCORE_MIN)
        | (human_scores > schema.SCORE_MAX)
    )
    if invalid_human_score.any():
        raise ValueError(
            "Canonical human_lmp_score values must be finite and within "
            f"{schema.SCORE_MIN}-{schema.SCORE_MAX}"
        )

    canonical_ids = _normalized_ids(canonical, context="canonical pairs")
    prediction_ids = _normalized_ids(predictions, context="predictions")
    expected = set(canonical_ids)
    observed = set(prediction_ids)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"Missing prediction IDs: {missing[:10]}")
        if unknown:
            details.append(f"Unknown/extra prediction IDs: {unknown[:10]}")
        raise ValueError("; ".join(details))

    validate_source_group_splits(canonical)
    canonical_work = canonical.copy().reset_index(drop=True)
    canonical_work["pair_id"] = canonical_ids.to_numpy()
    prediction_work = predictions.copy().reset_index(drop=True)
    prediction_work["pair_id"] = prediction_ids.to_numpy()
    prediction_work = prediction_work.set_index("pair_id").loc[canonical_ids].reset_index()

    for column in _METADATA_COLUMNS:
        if column not in prediction_work.columns:
            continue
        canonical_values = canonical_work[column].astype("string").str.strip()
        prediction_values = prediction_work[column].astype("string").str.strip()
        mismatch = (
            canonical_work[column].isna()
            | prediction_work[column].isna()
            | canonical_values.ne(prediction_values).fillna(True)
        )
        if mismatch.any():
            bad_ids = canonical_ids.loc[mismatch].tolist()[:10]
            raise ValueError(
                f"Prediction {column} metadata mismatch for pair IDs: {bad_ids}"
            )

    scores = pd.to_numeric(prediction_work[prediction_column], errors="coerce")
    numeric = scores.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(numeric).all():
        bad_ids = canonical_ids.loc[~np.isfinite(numeric)].tolist()[:10]
        raise ValueError(f"Prediction scores must be finite; invalid IDs: {bad_ids}")
    outside = (numeric < score_min) | (numeric > score_max)
    if outside.any():
        bad_ids = canonical_ids.loc[outside].tolist()[:10]
        raise ValueError(
            f"Prediction scores must be within {score_min:g}-{score_max:g}; "
            f"invalid IDs: {bad_ids}"
        )

    output = canonical_work[list(_OUTPUT_COLUMNS)].copy()
    output["prediction"] = numeric
    return output


__all__ = ["align_predictions", "validate_source_group_splits"]
