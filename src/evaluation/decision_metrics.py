"""Three-way legal decision metrics and threshold calibration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import DECISIONS


def _validated_labels(
    truth: Iterable[str], predictions: Iterable[str]
) -> tuple[np.ndarray, np.ndarray]:
    expected_values = list(truth)
    actual_values = list(predictions)
    if not expected_values or not actual_values:
        raise ValueError("Decision truth and predictions must be nonempty")
    if len(expected_values) != len(actual_values):
        raise ValueError("Decision truth and predictions must have equal length")
    legal = frozenset(DECISIONS)
    invalid = []
    for value in [*expected_values, *actual_values]:
        try:
            valid = value in legal
        except TypeError:
            valid = False
        if not valid:
            invalid.append(repr(value))
    if invalid:
        raise ValueError(
            f"Found illegal decision label(s): {sorted(set(invalid))}; "
            f"expected {DECISIONS}"
        )
    return (
        np.asarray(expected_values, dtype=object),
        np.asarray(actual_values, dtype=object),
    )


def _macro_f1_arrays(expected: np.ndarray, actual: np.ndarray) -> float:
    f1_values: list[float] = []
    for label in DECISIONS:
        true_positive = int(np.sum((expected == label) & (actual == label)))
        false_positive = int(np.sum((expected != label) & (actual == label)))
        false_negative = int(np.sum((expected == label) & (actual != label)))
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(f1_values))


def decision_accuracy(truth: Iterable[str], predictions: Iterable[str]) -> float:
    """Return exact three-way classification accuracy."""
    expected, actual = _validated_labels(truth, predictions)
    return float(np.mean(expected == actual))


def macro_f1(truth: Iterable[str], predictions: Iterable[str]) -> float:
    """Return macro-F1 over all three legal decision classes."""
    expected, actual = _validated_labels(truth, predictions)
    return _macro_f1_arrays(expected, actual)


def decision_metrics(
    truth: Iterable[str], predictions: Iterable[str]
) -> dict[str, float]:
    """Return three-way accuracy and macro-F1 over every legal class."""
    expected, actual = _validated_labels(truth, predictions)
    return {
        "accuracy": float(np.mean(expected == actual)),
        "macro_f1": _macro_f1_arrays(expected, actual),
    }


@dataclass(frozen=True)
class ThresholdFitRecord:
    """One validation row used to choose decision thresholds."""

    pair_id: str
    source_group: str
    split: str


@dataclass(frozen=True)
class DecisionThresholds:
    """Deterministic score cutoffs plus their complete validation provenance."""

    partial_min: float
    preserved_min: float
    fit_records: tuple[ThresholdFitRecord, ...]
    validation_metrics: Mapping[str, float]
    provenance: Mapping[str, Any]
    score_min: float = 1.0
    score_max: float = 10.0

    def __post_init__(self) -> None:
        numeric = [self.score_min, self.partial_min, self.preserved_min, self.score_max]
        if not np.isfinite(numeric).all():
            raise ValueError("Decision thresholds and score bounds must be finite")
        if not (
            self.score_min <= self.partial_min < self.preserved_min <= self.score_max
        ):
            raise ValueError(
                "Decision thresholds require score_min <= partial_min < "
                "preserved_min <= score_max"
            )
        if not isinstance(self.validation_metrics, Mapping) or not isinstance(
            self.provenance, Mapping
        ):
            raise ValueError("Decision threshold metrics and provenance must be mappings")
        seen: set[str] = set()
        for record in self.fit_records:
            if not isinstance(record, ThresholdFitRecord):
                raise ValueError("Decision threshold fit_records are malformed")
            if not record.pair_id.strip() or not record.source_group.strip():
                raise ValueError("Decision threshold fit provenance contains blank identity")
            if record.split != "validation":
                raise ValueError("Decision threshold fit provenance must be validation only")
            if record.pair_id in seen:
                raise ValueError(
                    f"Duplicate pair_id in decision threshold provenance: {record.pair_id}"
                )
            seen.add(record.pair_id)

    @property
    def fit_pair_ids(self) -> tuple[str, ...]:
        return tuple(record.pair_id for record in self.fit_records)

    @property
    def fit_source_groups(self) -> tuple[str, ...]:
        return tuple(record.source_group for record in self.fit_records)

    @property
    def fit_splits(self) -> tuple[str, ...]:
        return tuple(record.split for record in self.fit_records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "threshold_type": "validation_grid_search",
            "partial_min": self.partial_min,
            "preserved_min": self.preserved_min,
            "score_range": [self.score_min, self.score_max],
            "fit_split": "validation",
            "fit_records": [
                {
                    "pair_id": record.pair_id,
                    "source_group": record.source_group,
                    "split": record.split,
                }
                for record in self.fit_records
            ],
            "selection_objective": ["macro_f1", "accuracy"],
            "validation_metrics": dict(self.validation_metrics),
            "provenance": dict(self.provenance),
        }


def apply_decision_thresholds(
    scores: Iterable[float], thresholds: DecisionThresholds
) -> list[str]:
    """Apply ``>=`` preserved/partial boundaries in input order."""
    if not isinstance(thresholds, DecisionThresholds):
        raise TypeError("thresholds must be a validated DecisionThresholds object")
    try:
        values = np.asarray(list(scores), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Decision scores must be a finite numeric sequence") from exc
    if values.ndim != 1:
        raise ValueError("Decision scores must be one-dimensional")
    if values.size == 0:
        raise ValueError("Decision scores must be nonempty")
    if not np.isfinite(values).all():
        raise ValueError("Decision scores must all be finite")
    if ((values < thresholds.score_min) | (values > thresholds.score_max)).any():
        raise ValueError(
            f"Decision scores must be within "
            f"{thresholds.score_min:g}-{thresholds.score_max:g}"
        )
    labels = np.full(values.shape, "Not Preserved", dtype=object)
    labels[values >= thresholds.partial_min] = "Partially Preserved"
    labels[values >= thresholds.preserved_min] = "Preserved"
    return labels.tolist()


def fit_decision_thresholds(
    frame: pd.DataFrame,
    *,
    score_column: str = "prediction",
    decision_column: str = "decision",
    provenance: Mapping[str, Any] | None = None,
    score_min: float = 1.0,
    score_max: float = 10.0,
) -> DecisionThresholds:
    """Select deterministic cutoffs using validation rows and no other split."""
    if "draft" in score_column.casefold() or "draft" in decision_column.casefold():
        raise ValueError("draft fields must never be used to fit decision thresholds")
    if not np.isfinite([score_min, score_max]).all() or score_min >= score_max:
        raise ValueError("Decision threshold score bounds must be finite and increasing")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("Decision threshold provenance must be a mapping")

    required = {"pair_id", "source_group", "split", score_column, decision_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Decision threshold data is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Decision threshold fit requires nonempty validation data")

    normalized_identity: dict[str, pd.Series] = {}
    for column in ("pair_id", "source_group"):
        values = frame[column]
        normalized = values.astype("string").str.strip()
        blank = values.isna() | normalized.eq("").fillna(True)
        if blank.any():
            raise ValueError(f"Decision threshold {column} contains null or blank values")
        normalized_identity[column] = normalized.astype(str)
    duplicated = normalized_identity["pair_id"].duplicated(keep=False)
    if duplicated.any():
        examples = sorted(normalized_identity["pair_id"].loc[duplicated].unique())[:5]
        raise ValueError(f"Duplicate pair_id values in decision threshold data: {examples}")

    split_values = frame["split"]
    normalized_splits = split_values.astype("string").str.strip()
    invalid_split = (
        split_values.isna()
        | normalized_splits.eq("").fillna(True)
        | normalized_splits.ne("validation").fillna(True)
    )
    if invalid_split.any():
        found = sorted(
            {"<null>" if pd.isna(value) else str(value).strip() for value in split_values}
        )
        raise ValueError(
            "Decision thresholds must be fitted on validation rows only; "
            f"found split values {found}"
        )

    scores = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Decision threshold scores must all be finite")
    if ((scores < score_min) | (scores > score_max)).any():
        raise ValueError(
            f"Decision threshold scores must be within {score_min:g}-{score_max:g}"
        )
    truth, _ = _validated_labels(frame[decision_column], frame[decision_column])
    candidates = sorted({float(score_min), float(score_max), *scores.tolist()})
    best: tuple[tuple[float, ...], float, float, dict[str, float]] | None = None
    for partial_min in candidates:
        for preserved_min in candidates:
            if partial_min >= preserved_min:
                continue
            probe = DecisionThresholds(
                partial_min=partial_min,
                preserved_min=preserved_min,
                fit_records=(),
                validation_metrics={},
                provenance={},
                score_min=score_min,
                score_max=score_max,
            )
            predicted = apply_decision_thresholds(scores, probe)
            measured = decision_metrics(truth, predicted)
            tie_key = (
                -measured["macro_f1"],
                -measured["accuracy"],
                abs(partial_min - 5.0) + abs(preserved_min - 8.0),
                partial_min,
                preserved_min,
            )
            if best is None or tie_key < best[0]:
                best = (tie_key, partial_min, preserved_min, measured)
    if best is None:  # pragma: no cover - guarded by score range validation later
        raise ValueError("No valid ordered decision threshold pair is available")

    records = tuple(
        ThresholdFitRecord(
            pair_id=normalized_identity["pair_id"].iloc[index],
            source_group=normalized_identity["source_group"].iloc[index],
            split="validation",
        )
        for index in range(len(frame))
    )
    return DecisionThresholds(
        partial_min=best[1],
        preserved_min=best[2],
        fit_records=records,
        validation_metrics=best[3],
        provenance=dict(provenance or {"kind": "in_memory_dataframe"}),
        score_min=float(score_min),
        score_max=float(score_max),
    )


__all__ = [
    "DecisionThresholds",
    "ThresholdFitRecord",
    "apply_decision_thresholds",
    "decision_accuracy",
    "decision_metrics",
    "fit_decision_thresholds",
    "macro_f1",
]

