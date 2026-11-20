"""Pure, deterministic regression metrics for explicit in-memory arrays.

Correlation is mathematically undefined for fewer than two observations or
when either input is constant. This project reports ``0.0`` in those cases so
bootstrap and small subgroup results stay finite and reproducible. For a
constant truth vector, R² follows scikit-learn's finite convention: ``1.0`` for
an exact prediction and ``0.0`` otherwise.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def _as_numeric_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    try:
        vector = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric sequence") from exc
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return vector


def _validated_arrays(
    truth: Iterable[float], predictions: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    expected = _as_numeric_vector(truth, name="truth")
    actual = _as_numeric_vector(predictions, name="predictions")
    if expected.size == 0 or actual.size == 0:
        raise ValueError("truth and predictions must be nonempty")
    if expected.size != actual.size:
        raise ValueError("truth and predictions must have equal length")
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("truth and predictions must contain only finite values")
    return expected, actual


def _pearson_arrays(expected: np.ndarray, actual: np.ndarray) -> float:
    if expected.size < 2 or np.ptp(expected) == 0.0 or np.ptp(actual) == 0.0:
        return 0.0
    return float(np.corrcoef(expected, actual)[0, 1])


def pearson(truth: Iterable[float], predictions: Iterable[float]) -> float:
    """Pearson correlation, or ``0.0`` for a degenerate input."""
    expected, actual = _validated_arrays(truth, predictions)
    return _pearson_arrays(expected, actual)


def spearman(truth: Iterable[float], predictions: Iterable[float]) -> float:
    """Spearman rank correlation with average ranks and finite degeneracy."""
    expected, actual = _validated_arrays(truth, predictions)
    return _pearson_arrays(rankdata(expected), rankdata(actual))


def rmse(truth: Iterable[float], predictions: Iterable[float]) -> float:
    """Root mean squared error."""
    expected, actual = _validated_arrays(truth, predictions)
    return float(np.sqrt(np.mean((actual - expected) ** 2)))


def mae(truth: Iterable[float], predictions: Iterable[float]) -> float:
    """Mean absolute error."""
    expected, actual = _validated_arrays(truth, predictions)
    return float(np.mean(np.abs(actual - expected)))


def r2(truth: Iterable[float], predictions: Iterable[float]) -> float:
    """Coefficient of determination with finite constant-truth behavior."""
    expected, actual = _validated_arrays(truth, predictions)
    squared_error = float(np.sum((actual - expected) ** 2))
    denominator = float(np.sum((expected - expected.mean()) ** 2))
    if denominator == 0.0:
        return 1.0 if squared_error == 0.0 else 0.0
    return float(1.0 - squared_error / denominator)


def regression_metrics(
    truth: Iterable[float], predictions: Iterable[float]
) -> dict[str, float]:
    """Return Pearson, Spearman, RMSE, MAE, and R² after one validation pass."""
    expected, actual = _validated_arrays(truth, predictions)
    residual = actual - expected
    squared_error = residual**2
    denominator = float(np.sum((expected - expected.mean()) ** 2))
    error_sum = float(np.sum(squared_error))
    r2_value = (
        (1.0 if error_sum == 0.0 else 0.0)
        if denominator == 0.0
        else float(1.0 - error_sum / denominator)
    )
    return {
        "pearson": _pearson_arrays(expected, actual),
        "spearman": _pearson_arrays(rankdata(expected), rankdata(actual)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2_value,
    }


def stratified_regression_metrics(
    frame: pd.DataFrame,
    *,
    truth_column: str = "human_lmp_score",
    prediction_column: str = "prediction",
    group_columns: tuple[str, ...] = ("language", "legal_domain", "split"),
) -> dict[str, dict[str, object]]:
    """Report overall and deterministic per-metadata regression metrics."""
    required = {truth_column, prediction_column, *group_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Metric frame is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Metric frame must be nonempty")

    for column in group_columns:
        values = frame[column]
        blank = values.isna() | values.astype("string").str.strip().eq("").fillna(True)
        if blank.any():
            raise ValueError(f"{column} metadata contains null or blank values")

    def summarize(part: pd.DataFrame) -> dict[str, float | int]:
        return {
            "n_rows": int(len(part)),
            **regression_metrics(part[truth_column], part[prediction_column]),
        }

    report: dict[str, dict[str, object]] = {"overall": summarize(frame)}
    for column in group_columns:
        grouped: dict[str, object] = {}
        normalized = frame.assign(
            **{column: frame[column].astype(str).str.strip()}
        )
        for value, part in normalized.groupby(column, sort=True, dropna=False):
            grouped[str(value)] = summarize(part)
        report[f"by_{column}"] = grouped
    return report


__all__ = [
    "mae",
    "pearson",
    "r2",
    "regression_metrics",
    "rmse",
    "spearman",
    "stratified_regression_metrics",
]

