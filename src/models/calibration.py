"""Leakage-safe calibration of raw model scores to the 1-10 LMP scale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Mapping

import numpy as np
import pandas as pd

from src.data import schema
from src.utils import io as io_utils

_DRAFT_LABEL_COLUMNS = frozenset({"draft_lmp_score", "draft_error_type"})


@dataclass(frozen=True)
class CalibrationFitRecord:
    """Pair and split used to estimate calibration parameters."""

    pair_id: str
    split: str


@dataclass(frozen=True)
class LinearScoreCalibrator:
    """Affine score calibrator fitted only on explicitly allowed splits."""

    slope: float
    intercept: float
    fit_records: tuple[CalibrationFitRecord, ...]
    score_min: float = 1.0
    score_max: float = 10.0

    calibrator_type: ClassVar[str] = "linear_regression"

    @property
    def fit_pair_ids(self) -> tuple[str, ...]:
        """Pair IDs used during fitting, in fit order."""
        return tuple(record.pair_id for record in self.fit_records)

    @property
    def fit_splits(self) -> tuple[str, ...]:
        """Split name for each fitted pair, in fit order."""
        return tuple(record.split for record in self.fit_records)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation including fit-row provenance."""
        return {
            "schema_version": 1,
            "calibrator_type": self.calibrator_type,
            "slope": self.slope,
            "intercept": self.intercept,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "fit_records": [
                {"pair_id": record.pair_id, "split": record.split}
                for record in self.fit_records
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinearScoreCalibrator":
        """Validate and restore a serialized calibrator."""
        if payload.get("calibrator_type") != cls.calibrator_type:
            raise ValueError(
                f"Unknown calibrator_type: {payload.get('calibrator_type')!r}"
            )
        schema_version = payload.get("schema_version")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version != 1
        ):
            raise ValueError(
                f"Unsupported calibration schema_version: {schema_version!r}; "
                "expected the non-bool integer 1"
            )
        try:
            raw_parameters = {
                name: payload[name]
                for name in ("slope", "intercept", "score_min", "score_max")
            }
            raw_records = payload["fit_records"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Malformed serialized calibration payload") from exc
        for name, value in raw_parameters.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"Serialized calibration {name} must be a finite real number, "
                    "not a boolean"
                )
        slope = float(raw_parameters["slope"])
        intercept = float(raw_parameters["intercept"])
        score_min = float(raw_parameters["score_min"])
        score_max = float(raw_parameters["score_max"])
        if (
            score_min != float(schema.SCORE_MIN)
            or score_max != float(schema.SCORE_MAX)
        ):
            raise ValueError(
                f"Serialized calibration target range must be "
                f"{schema.SCORE_MIN}-{schema.SCORE_MAX}"
            )
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("Serialized calibration fit_records must be non-empty")

        records: list[CalibrationFitRecord] = []
        seen: set[str] = set()
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                raise ValueError("Malformed serialized calibration fit record")
            raw_pair_id = raw_record.get("pair_id")
            if raw_pair_id is None or pd.isna(raw_pair_id):
                raise ValueError("Serialized calibration has a null or blank pair_id")
            pair_id = str(raw_pair_id).strip()
            split = str(raw_record.get("split", "")).strip()
            if not pair_id:
                raise ValueError("Serialized calibration has a null or blank pair_id")
            if pair_id in seen:
                raise ValueError(f"Duplicate pair_id in serialized calibration: {pair_id}")
            if split == "test":
                raise ValueError("Serialized calibration provenance must never contain test rows")
            if split not in {"train", "validation"}:
                raise ValueError(f"Unknown serialized calibration split: {split!r}")
            seen.add(pair_id)
            records.append(CalibrationFitRecord(pair_id, split))

        return cls(
            slope=slope,
            intercept=intercept,
            fit_records=tuple(records),
            score_min=score_min,
            score_max=score_max,
        )

    def save(self, path: str | Path) -> Path:
        """Serialize this calibrator with the project's JSON writer."""
        return io_utils.write_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "LinearScoreCalibrator":
        """Load and validate a serialized calibrator."""
        payload = io_utils.read_json(path)
        if not isinstance(payload, Mapping):
            raise ValueError("Serialized calibration root must be a mapping")
        return cls.from_dict(payload)

    def transform(
        self, raw_scores: Iterable[float], *, clip: bool = False
    ) -> np.ndarray:
        """Apply calibration, rejecting out-of-range output unless clipping is explicit."""
        raw = np.asarray(list(raw_scores), dtype=float)
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError("Raw scores to transform must be a finite one-dimensional sequence")
        calibrated = self.slope * raw + self.intercept
        outside = (calibrated < self.score_min) | (calibrated > self.score_max)
        if outside.any() and not clip:
            raise ValueError(
                f"Calibrated scores outside {self.score_min:g}-{self.score_max:g}; "
                "pass clip=True to clip explicitly"
            )
        if clip:
            calibrated = np.clip(calibrated, self.score_min, self.score_max)
        return calibrated.astype(float, copy=False)

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        raw_score_column: str = "raw_score",
        label_column: str = "label",
        fit_splits: Iterable[str] = ("train",),
    ) -> "LinearScoreCalibrator":
        """Fit an affine mapping on selected train/validation rows."""
        for column_name in (raw_score_column, label_column):
            if column_name in _DRAFT_LABEL_COLUMNS:
                raise ValueError(
                    f"{column_name} is a draft field and must never be used for calibration"
                )
        if raw_score_column == label_column:
            raise ValueError("Raw-score and label columns must be different")

        required = {"pair_id", "split", raw_score_column, label_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Calibration data is missing columns: {missing}")

        requested = tuple(dict.fromkeys(str(name).strip() for name in fit_splits))
        disallowed = sorted(set(requested) - {"train", "validation"})
        if not requested or disallowed:
            raise ValueError(
                "Calibration fit split(s) must be train and/or validation; "
                f"got {list(requested)}"
            )

        split_values = frame["split"]
        if split_values.isna().any():
            raise ValueError("Unknown calibration split value: <null>")
        normalised_splits = split_values.astype(str).str.strip()
        unknown = sorted(
            set(normalised_splits) - {"train", "validation", "test"}
        )
        if unknown:
            raise ValueError(f"Unknown calibration split value(s): {unknown}")
        if (normalised_splits == "test").any():
            raise ValueError(
                "Calibration data contains test rows; test rows must never be passed to fit"
            )

        fit_frame = frame.loc[normalised_splits.isin(requested)].copy()
        fit_frame["split"] = normalised_splits.loc[fit_frame.index]
        if fit_frame.empty:
            raise ValueError(
                f"No rows available for calibration fit split(s) {list(requested)}"
            )

        ids = fit_frame["pair_id"]
        blank_ids = ids.isna() | ids.astype("string").str.strip().eq("").fillna(True)
        if blank_ids.any():
            raise ValueError("Calibration data contains null or blank pair_id value(s)")
        normalised_ids = ids.astype(str).str.strip()
        duplicates = normalised_ids[normalised_ids.duplicated(keep=False)]
        if not duplicates.empty:
            raise ValueError(
                f"Duplicate pair_id value(s) in calibration data: "
                f"{sorted(duplicates.unique())[:5]}"
            )
        fit_frame["pair_id"] = normalised_ids

        raw = pd.to_numeric(fit_frame[raw_score_column], errors="coerce").to_numpy(
            dtype=float
        )
        labels = pd.to_numeric(fit_frame[label_column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(raw).all():
            raise ValueError("Calibration raw scores must all be finite")
        if not np.isfinite(labels).all():
            raise ValueError("Calibration labels must all be finite")
        outside = (labels < schema.SCORE_MIN) | (labels > schema.SCORE_MAX)
        if outside.any():
            raise ValueError(
                f"Calibration labels must be within {schema.SCORE_MIN}-{schema.SCORE_MAX}"
            )

        design = np.column_stack([raw, np.ones(len(raw), dtype=float)])
        slope, intercept = np.linalg.lstsq(design, labels, rcond=None)[0]
        records = tuple(
            CalibrationFitRecord(str(row.pair_id), str(row.split))
            for row in fit_frame[["pair_id", "split"]].itertuples(index=False)
        )
        return cls(float(slope), float(intercept), records)
