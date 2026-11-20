"""Tests for finite, deterministic regression evaluation metrics."""

from __future__ import annotations

from math import sqrt

import pytest


def test_regression_metrics_match_known_perfect_and_reversed_answers() -> None:
    from src.evaluation.metrics import regression_metrics

    perfect = regression_metrics([1, 2, 3], [1, 2, 3])
    reversed_scores = regression_metrics([1, 2, 3], [3, 2, 1])

    assert perfect == {
        "pearson": pytest.approx(1.0),
        "spearman": pytest.approx(1.0),
        "rmse": pytest.approx(0.0),
        "mae": pytest.approx(0.0),
        "r2": pytest.approx(1.0),
    }
    assert reversed_scores == {
        "pearson": pytest.approx(-1.0),
        "spearman": pytest.approx(-1.0),
        "rmse": pytest.approx(sqrt(8.0 / 3.0)),
        "mae": pytest.approx(4.0 / 3.0),
        "r2": pytest.approx(-3.0),
    }


def test_individual_metric_helpers_validate_and_match_the_bundle() -> None:
    from src.evaluation.metrics import (
        mae,
        pearson,
        r2,
        regression_metrics,
        rmse,
        spearman,
    )

    expected = regression_metrics([1, 2, 3], [3, 2, 1])

    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(expected["pearson"])
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(expected["spearman"])
    assert rmse([1, 2, 3], [3, 2, 1]) == pytest.approx(expected["rmse"])
    assert mae([1, 2, 3], [3, 2, 1]) == pytest.approx(expected["mae"])
    assert r2([1, 2, 3], [3, 2, 1]) == pytest.approx(expected["r2"])


@pytest.mark.parametrize(
    ("truth", "predictions", "message"),
    [
        ([], [], "nonempty"),
        ([1.0], [1.0, 2.0], "equal length"),
        ([1.0, float("nan")], [1.0, 2.0], "finite"),
        ([1.0, 2.0], [1.0, float("inf")], "finite"),
        ([[1.0, 2.0]], [[1.0, 2.0]], "one-dimensional"),
    ],
)
def test_regression_metrics_reject_invalid_sequences(
    truth, predictions, message
) -> None:
    from src.evaluation.metrics import regression_metrics

    with pytest.raises(ValueError, match=message):
        regression_metrics(truth, predictions)


def test_constant_correlation_and_r2_conventions_are_deterministic() -> None:
    from src.evaluation.metrics import regression_metrics

    perfect_constant = regression_metrics([4, 4, 4], [4, 4, 4])
    imperfect_constant = regression_metrics([4, 4, 4], [5, 5, 5])
    singleton = regression_metrics([4], [4])

    assert perfect_constant["pearson"] == 0.0
    assert perfect_constant["spearman"] == 0.0
    assert perfect_constant["r2"] == 1.0
    assert imperfect_constant["pearson"] == 0.0
    assert imperfect_constant["spearman"] == 0.0
    assert imperfect_constant["r2"] == 0.0
    assert singleton["pearson"] == 0.0
    assert singleton["spearman"] == 0.0
    assert singleton["r2"] == 1.0


def test_stratified_metrics_report_language_domain_and_split_groups() -> None:
    import pandas as pd

    from src.evaluation.metrics import stratified_regression_metrics

    frame = pd.DataFrame(
        {
            "human_lmp_score": [1.0, 2.0, 3.0, 4.0],
            "prediction": [1.0, 3.0, 3.0, 5.0],
            "language": ["en", "en", "hi", "hi"],
            "legal_domain": ["contract", "cyber", "contract", "cyber"],
            "split": ["validation", "test", "test", "test"],
        }
    )

    report = stratified_regression_metrics(frame)

    assert report["overall"]["n_rows"] == 4
    assert report["overall"]["mae"] == pytest.approx(0.5)
    assert set(report["by_language"]) == {"en", "hi"}
    assert report["by_language"]["en"]["n_rows"] == 2
    assert report["by_language"]["en"]["mae"] == pytest.approx(0.5)
    assert set(report["by_legal_domain"]) == {"contract", "cyber"}
    assert report["by_legal_domain"]["contract"]["rmse"] == 0.0
    assert set(report["by_split"]) == {"validation", "test"}
    assert report["by_split"]["test"]["n_rows"] == 3


def test_stratified_metrics_reject_blank_group_metadata() -> None:
    import pandas as pd

    from src.evaluation.metrics import stratified_regression_metrics

    frame = pd.DataFrame(
        {
            "human_lmp_score": [5.0],
            "prediction": [5.0],
            "language": [""],
            "legal_domain": ["contract"],
            "split": ["test"],
        }
    )

    with pytest.raises(ValueError, match="language.*null or blank"):
        stratified_regression_metrics(frame)
