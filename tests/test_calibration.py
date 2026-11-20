"""Tests for leakage-safe score calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import LinearScoreCalibrator
from src.utils.io import read_json


def test_score_calibration_defaults_to_train_rows_and_records_fit_provenance() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2", "V1"],
            "split": ["train", "train", "validation"],
            "raw_score": [0.0, 1.0, 2.0],
            "label": [1.0, 3.0, 10.0],
        }
    )

    calibrator = LinearScoreCalibrator.fit(frame)

    assert calibrator.slope == pytest.approx(2.0)
    assert calibrator.intercept == pytest.approx(1.0)
    assert calibrator.fit_pair_ids == ("T1", "T2")
    assert calibrator.fit_splits == ("train", "train")


def test_score_calibration_can_explicitly_fit_validation_rows() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "V1", "V2"],
            "split": ["train", "validation", "validation"],
            "raw_score": [100.0, 1.0, 3.0],
            "label": [1.0, 2.0, 6.0],
        }
    )

    calibrator = LinearScoreCalibrator.fit(
        frame, fit_splits=("validation",)
    )

    assert calibrator.slope == pytest.approx(2.0)
    assert calibrator.intercept == pytest.approx(0.0)
    assert calibrator.fit_pair_ids == ("V1", "V2")
    assert calibrator.fit_splits == ("validation", "validation")


def test_score_calibration_always_rejects_test_rows() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2", "X1"],
            "split": ["train", "train", "test"],
            "raw_score": [0.0, 1.0, 2.0],
            "label": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="test.*never"):
        LinearScoreCalibrator.fit(frame)


@pytest.mark.parametrize("fit_splits", [("test",), ("dev",)])
def test_score_calibration_rejects_disallowed_fit_splits(fit_splits) -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2"],
            "split": ["train", "train"],
            "raw_score": [0.0, 1.0],
            "label": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="fit split"):
        LinearScoreCalibrator.fit(frame, fit_splits=fit_splits)


def test_score_calibration_rejects_unknown_data_split() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2"],
            "split": ["train", "dev"],
            "raw_score": [0.0, 1.0],
            "label": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="Unknown calibration split"):
        LinearScoreCalibrator.fit(frame)


@pytest.mark.parametrize("bad_raw", [np.nan, np.inf, -np.inf, "not-a-number"])
def test_score_calibration_requires_finite_raw_scores(bad_raw) -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2"],
            "split": ["train", "train"],
            "raw_score": [0.0, bad_raw],
            "label": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="raw scores.*finite"):
        LinearScoreCalibrator.fit(frame)


@pytest.mark.parametrize("bad_label", [np.nan, np.inf, -np.inf, "not-a-number"])
def test_score_calibration_requires_finite_labels(bad_label) -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2"],
            "split": ["train", "train"],
            "raw_score": [0.0, 1.0],
            "label": [1.0, bad_label],
        }
    )

    with pytest.raises(ValueError, match="labels.*finite"):
        LinearScoreCalibrator.fit(frame)


@pytest.mark.parametrize("bad_label", [0.99, 10.01])
def test_score_calibration_requires_labels_in_one_to_ten(bad_label) -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2"],
            "split": ["train", "train"],
            "raw_score": [0.0, 1.0],
            "label": [1.0, bad_label],
        }
    )

    with pytest.raises(ValueError, match="labels.*1.*10"):
        LinearScoreCalibrator.fit(frame)


@pytest.mark.parametrize("argument", ["label_column", "raw_score_column"])
def test_score_calibration_never_uses_draft_scores(argument) -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2"],
            "split": ["train", "train"],
            "raw_score": [0.0, 1.0],
            "label": [1.0, 2.0],
            "draft_lmp_score": [9.0, 9.0],
        }
    )

    with pytest.raises(ValueError, match="draft_lmp_score.*never"):
        LinearScoreCalibrator.fit(frame, **{argument: "draft_lmp_score"})


@pytest.mark.parametrize("bad_id", [None, "", "  "])
def test_score_calibration_rejects_null_or_blank_pair_ids(bad_id) -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", bad_id],
            "split": ["train", "train"],
            "raw_score": [0.0, 1.0],
            "label": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="null or blank"):
        LinearScoreCalibrator.fit(frame)


def test_score_calibration_rejects_duplicate_pair_ids() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T1"],
            "split": ["train", "train"],
            "raw_score": [0.0, 1.0],
            "label": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="Duplicate pair_id"):
        LinearScoreCalibrator.fit(frame)


def test_score_calibrator_serialization_preserves_fit_provenance(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2"],
            "split": ["train", "train"],
            "raw_score": [0.0, 1.0],
            "label": [1.0, 3.0],
        }
    )
    calibrator = LinearScoreCalibrator.fit(frame)

    path = calibrator.save(tmp_path / "calibration.json")
    restored = LinearScoreCalibrator.load(path)

    assert restored == calibrator
    payload = read_json(path)
    assert payload["calibrator_type"] == "linear_regression"
    assert payload["fit_records"] == [
        {"pair_id": "T1", "split": "train"},
        {"pair_id": "T2", "split": "train"},
    ]


def test_score_calibrator_clips_only_when_explicitly_requested() -> None:
    frame = pd.DataFrame(
        {
            "pair_id": ["T1", "T2"],
            "split": ["train", "train"],
            "raw_score": [0.0, 1.0],
            "label": [1.0, 3.0],
        }
    )
    calibrator = LinearScoreCalibrator.fit(frame)

    with pytest.raises(ValueError, match="outside.*1.*10"):
        calibrator.transform([-1.0, 5.0])

    assert calibrator.transform([-1.0, 5.0], clip=True).tolist() == [1.0, 10.0]


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("schema_version", True),
        ("slope", True),
        ("intercept", False),
        ("score_min", True),
        ("score_max", "10.0"),
    ],
)
def test_deserialization_requires_strict_numeric_scalar_types(field, bad_value) -> None:
    payload = {
        "schema_version": 1,
        "calibrator_type": "linear_regression",
        "slope": 1.0,
        "intercept": 0.0,
        "score_min": 1.0,
        "score_max": 10.0,
        "fit_records": [{"pair_id": "T1", "split": "train"}],
    }
    payload[field] = bad_value

    with pytest.raises(ValueError, match="schema_version|finite real number"):
        LinearScoreCalibrator.from_dict(payload)


def test_deserialization_rejects_nonstandard_target_range() -> None:
    payload = {
        "schema_version": 1,
        "calibrator_type": "linear_regression",
        "slope": 1.0,
        "intercept": 0.0,
        "score_min": 0.0,
        "score_max": 10.0,
        "fit_records": [{"pair_id": "T1", "split": "train"}],
    }

    with pytest.raises(ValueError, match="1-10"):
        LinearScoreCalibrator.from_dict(payload)


def test_deserialization_rejects_test_fit_provenance() -> None:
    payload = {
        "schema_version": 1,
        "calibrator_type": "linear_regression",
        "slope": 1.0,
        "intercept": 0.0,
        "score_min": 1.0,
        "score_max": 10.0,
        "fit_records": [{"pair_id": "X1", "split": "test"}],
    }

    with pytest.raises(ValueError, match="never contain test"):
        LinearScoreCalibrator.from_dict(payload)


def test_deserialization_rejects_null_pair_id_provenance() -> None:
    payload = {
        "schema_version": 1,
        "calibrator_type": "linear_regression",
        "slope": 1.0,
        "intercept": 0.0,
        "score_min": 1.0,
        "score_max": 10.0,
        "fit_records": [{"pair_id": None, "split": "train"}],
    }

    with pytest.raises(ValueError, match="null or blank"):
        LinearScoreCalibrator.from_dict(payload)
