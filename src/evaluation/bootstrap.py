"""Deterministic cluster bootstrap confidence intervals.

Rows are sampled only through their ``source_group`` cluster. Sibling rows are
therefore repeated or omitted together rather than being treated as independent
observations.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.metrics import mae, pearson, r2, rmse, spearman

MetricFunction = Callable[[Iterable[float], Iterable[float]], float]
_METRICS: dict[str, MetricFunction] = {
    "pearson": pearson,
    "spearman": spearman,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
}


@dataclass(frozen=True)
class ClusterResample:
    """One cluster draw and the expanded canonical row indices."""

    sampled_groups: tuple[str, ...]
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class PairedBootstrapDelta:
    """Paired metric delta computed from the same cluster draws for both models."""

    metric: str
    delta_estimate: float
    lower: float
    upper: float
    confidence: float
    seed: int
    requested_replicates: int
    effective_replicates: int
    n_clusters: int
    model_a_replicates: tuple[float, ...]
    model_b_replicates: tuple[float, ...]
    replicate_deltas: tuple[float, ...]
    resampled_groups: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "delta_definition": "model_a_minus_model_b",
            "delta_estimate": self.delta_estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "seed": self.seed,
            "requested_replicates": self.requested_replicates,
            "effective_replicates": self.effective_replicates,
            "n_clusters": self.n_clusters,
        }


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile interval with explicit requested/effective replicate counts."""

    metric: str
    estimate: float
    lower: float
    upper: float
    confidence: float
    seed: int
    requested_replicates: int
    effective_replicates: int
    n_clusters: int
    replicates: tuple[float, ...]
    resampled_groups: tuple[tuple[str, ...], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "seed": self.seed,
            "requested_replicates": self.requested_replicates,
            "effective_replicates": self.effective_replicates,
            "n_clusters": self.n_clusters,
        }


def _validate_settings(n_resamples: int, confidence: float, seed: int) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not np.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be a finite number strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")


def _validated_groups(source_groups: Iterable[Any]) -> tuple[str, ...]:
    raw = list(source_groups)
    if not raw:
        raise ValueError("source_group IDs must be nonempty")
    groups: list[str] = []
    for value in raw:
        if isinstance(value, (list, tuple, set, dict, np.ndarray, pd.Series)):
            raise ValueError(
                "source_group IDs must be a one-dimensional sequence of scalar values"
            )
        if value is None or pd.isna(value):
            raise ValueError("source_group IDs contain null or blank values")
        group = str(value).strip()
        if not group:
            raise ValueError("source_group IDs contain null or blank values")
        groups.append(group)
    return tuple(groups)


def generate_cluster_resamples(
    source_groups: Iterable[Any], *, n_resamples: int = 1000, seed: int = 42
) -> tuple[ClusterResample, ...]:
    """Draw source-group clusters with replacement and expand all sibling rows."""
    _validate_settings(n_resamples, 0.95, seed)
    groups = _validated_groups(source_groups)
    unique_groups = tuple(dict.fromkeys(groups))
    members = {
        group: tuple(index for index, value in enumerate(groups) if value == group)
        for group in unique_groups
    }
    rng = np.random.default_rng(int(seed))
    plans: list[ClusterResample] = []
    for _ in range(n_resamples):
        selected = tuple(
            unique_groups[index]
            for index in rng.integers(0, len(unique_groups), size=len(unique_groups))
        )
        indices = tuple(index for group in selected for index in members[group])
        plans.append(ClusterResample(selected, indices))
    return tuple(plans)


def _metric_function(metric: str | MetricFunction) -> tuple[str, MetricFunction]:
    if isinstance(metric, str):
        name = metric.strip().casefold()
        if name not in _METRICS:
            raise ValueError(f"Unknown regression metric {metric!r}; expected {sorted(_METRICS)}")
        return name, _METRICS[name]
    if not callable(metric):
        raise ValueError("metric must be a supported name or callable")
    return getattr(metric, "__name__", "custom_metric"), metric


def _numeric_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    try:
        result = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric sequence") from exc
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional sequence")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _checked_metric(
    function: MetricFunction,
    truth: np.ndarray,
    predictions: np.ndarray,
    *,
    metric_name: str,
    replicate_index: int | None,
) -> float:
    location = "point estimate" if replicate_index is None else f"replicate {replicate_index}"
    try:
        value = float(function(truth, predictions))
    except Exception as exc:
        raise ValueError(
            f"Bootstrap {location} for metric {metric_name!r} is invalid: {exc}"
        ) from exc
    if not np.isfinite(value):
        raise ValueError(
            f"Bootstrap {location} for metric {metric_name!r} is non-finite"
        )
    return value


def cluster_bootstrap_ci(
    truth: Iterable[float],
    predictions: Iterable[float],
    source_groups: Iterable[Any],
    *,
    metric: str | MetricFunction = "rmse",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapInterval:
    """Return a seeded source-group percentile interval for one metric."""
    _validate_settings(n_resamples, confidence, seed)
    expected = _numeric_vector(truth, name="truth")
    actual = _numeric_vector(predictions, name="predictions")
    groups = _validated_groups(source_groups)
    if len(expected) != len(actual) or len(expected) != len(groups):
        raise ValueError("truth, predictions, and source_group IDs must have equal length")
    metric_name, function = _metric_function(metric)
    estimate = _checked_metric(
        function,
        expected,
        actual,
        metric_name=metric_name,
        replicate_index=None,
    )
    plans = generate_cluster_resamples(groups, n_resamples=n_resamples, seed=seed)
    values: list[float] = []
    for index, plan in enumerate(plans):
        rows = np.asarray(plan.row_indices, dtype=int)
        values.append(
            _checked_metric(
                function,
                expected[rows],
                actual[rows],
                metric_name=metric_name,
                replicate_index=index,
            )
        )
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(np.asarray(values), [alpha, 1.0 - alpha])
    return BootstrapInterval(
        metric=metric_name,
        estimate=estimate,
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        seed=int(seed),
        requested_replicates=n_resamples,
        effective_replicates=len(values),
        n_clusters=len(set(groups)),
        replicates=tuple(values),
        resampled_groups=tuple(plan.sampled_groups for plan in plans),
    )


def bootstrap_regression_metrics(
    truth: Iterable[float],
    predictions: Iterable[float],
    source_groups: Iterable[Any],
    *,
    metrics: Iterable[str] = ("pearson", "spearman", "rmse", "mae", "r2"),
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, BootstrapInterval]:
    """Return CIs for requested regression metrics using identical seeded draws."""
    metric_names = tuple(metrics)
    if not metric_names:
        raise ValueError("At least one regression metric is required")
    results: dict[str, BootstrapInterval] = {}
    for metric_name in metric_names:
        normalized, _ = _metric_function(metric_name)
        if normalized in results:
            raise ValueError(f"Duplicate regression metric requested: {normalized}")
        results[normalized] = cluster_bootstrap_ci(
            truth,
            predictions,
            source_groups,
            metric=normalized,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
    return results


def paired_metric_delta(
    truth: Iterable[float],
    model_a_predictions: Iterable[float],
    model_b_predictions: Iterable[float],
    source_groups: Iterable[Any],
    *,
    metric: str | MetricFunction = "rmse",
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PairedBootstrapDelta:
    """Compare models as A minus B using one shared cluster plan per replicate."""
    _validate_settings(n_resamples, confidence, seed)
    expected = _numeric_vector(truth, name="truth")
    model_a = _numeric_vector(model_a_predictions, name="model_a_predictions")
    model_b = _numeric_vector(model_b_predictions, name="model_b_predictions")
    groups = _validated_groups(source_groups)
    if not (len(expected) == len(model_a) == len(model_b) == len(groups)):
        raise ValueError(
            "truth, both model predictions, and source_group IDs must have equal length"
        )
    metric_name, function = _metric_function(metric)
    point_a = _checked_metric(
        function,
        expected,
        model_a,
        metric_name=metric_name,
        replicate_index=None,
    )
    point_b = _checked_metric(
        function,
        expected,
        model_b,
        metric_name=metric_name,
        replicate_index=None,
    )
    plans = generate_cluster_resamples(groups, n_resamples=n_resamples, seed=seed)
    values_a: list[float] = []
    values_b: list[float] = []
    for index, plan in enumerate(plans):
        rows = np.asarray(plan.row_indices, dtype=int)
        values_a.append(
            _checked_metric(
                function,
                expected[rows],
                model_a[rows],
                metric_name=metric_name,
                replicate_index=index,
            )
        )
        values_b.append(
            _checked_metric(
                function,
                expected[rows],
                model_b[rows],
                metric_name=metric_name,
                replicate_index=index,
            )
        )
    deltas = tuple(a - b for a, b in zip(values_a, values_b, strict=True))
    alpha = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(np.asarray(deltas), [alpha, 1.0 - alpha])
    return PairedBootstrapDelta(
        metric=metric_name,
        delta_estimate=point_a - point_b,
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        seed=int(seed),
        requested_replicates=n_resamples,
        effective_replicates=len(deltas),
        n_clusters=len(set(groups)),
        model_a_replicates=tuple(values_a),
        model_b_replicates=tuple(values_b),
        replicate_deltas=deltas,
        resampled_groups=tuple(plan.sampled_groups for plan in plans),
    )


__all__ = [
    "BootstrapInterval",
    "ClusterResample",
    "PairedBootstrapDelta",
    "bootstrap_regression_metrics",
    "cluster_bootstrap_ci",
    "generate_cluster_resamples",
    "paired_metric_delta",
]
