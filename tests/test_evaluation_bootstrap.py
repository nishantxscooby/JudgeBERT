"""Tests for deterministic source-group cluster bootstrap intervals."""

from __future__ import annotations

from collections import Counter

import pytest


def test_cluster_bootstrap_is_seeded_and_resamples_whole_source_groups() -> None:
    from src.evaluation.bootstrap import (
        cluster_bootstrap_ci,
        generate_cluster_resamples,
    )

    groups = ["A", "A", "B", "C", "C"]
    first_plans = generate_cluster_resamples(groups, n_resamples=25, seed=17)
    second_plans = generate_cluster_resamples(groups, n_resamples=25, seed=17)

    assert first_plans == second_plans
    assert len(first_plans) == 25
    for plan in first_plans:
        draws = Counter(plan.sampled_groups)
        row_counts = Counter(plan.row_indices)
        assert len(plan.sampled_groups) == 3
        assert row_counts[0] == row_counts[1] == draws["A"]
        assert row_counts[2] == draws["B"]
        assert row_counts[3] == row_counts[4] == draws["C"]

    first = cluster_bootstrap_ci(
        [1, 2, 3, 4, 5],
        [1, 3, 2, 4, 6],
        groups,
        metric="rmse",
        n_resamples=25,
        confidence=0.90,
        seed=17,
    )
    second = cluster_bootstrap_ci(
        [1, 2, 3, 4, 5],
        [1, 3, 2, 4, 6],
        groups,
        metric="rmse",
        n_resamples=25,
        confidence=0.90,
        seed=17,
    )

    assert first == second
    assert first.metric == "rmse"
    assert first.requested_replicates == 25
    assert first.effective_replicates == 25
    assert first.n_clusters == 3
    assert len(first.replicates) == 25
    assert first.lower <= first.estimate <= first.upper
    assert first.to_dict()["effective_replicates"] == 25


def test_regression_cis_and_paired_deltas_share_cluster_resamples() -> None:
    from src.evaluation.bootstrap import (
        bootstrap_regression_metrics,
        paired_metric_delta,
    )

    truth = [1.0, 2.0, 3.0, 4.0, 5.0]
    model_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    model_b = [2.0, 2.5, 4.0, 4.5, 6.0]
    groups = ["A", "A", "B", "C", "C"]

    intervals = bootstrap_regression_metrics(
        truth,
        model_b,
        groups,
        n_resamples=40,
        confidence=0.90,
        seed=9,
    )
    paired = paired_metric_delta(
        truth,
        model_a,
        model_b,
        groups,
        metric="mae",
        n_resamples=40,
        confidence=0.90,
        seed=9,
    )

    assert set(intervals) == {"pearson", "spearman", "rmse", "mae", "r2"}
    group_draws = {result.resampled_groups for result in intervals.values()}
    assert len(group_draws) == 1
    assert paired.resampled_groups == intervals["mae"].resampled_groups
    assert paired.delta_estimate == pytest.approx(-0.8)
    assert paired.requested_replicates == 40
    assert paired.effective_replicates == 40
    assert len(paired.model_a_replicates) == 40
    assert len(paired.model_b_replicates) == 40
    assert paired.replicate_deltas == tuple(
        a - b
        for a, b in zip(
            paired.model_a_replicates,
            paired.model_b_replicates,
            strict=True,
        )
    )
    assert paired.to_dict()["delta_definition"] == "model_a_minus_model_b"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_resamples": 0}, "positive integer"),
        ({"n_resamples": True}, "positive integer"),
        ({"confidence": 0.0}, "between 0 and 1"),
        ({"confidence": 1.0}, "between 0 and 1"),
        ({"confidence": float("nan")}, "between 0 and 1"),
        ({"seed": 1.5}, "seed.*integer"),
        ({"seed": True}, "seed.*integer"),
    ],
)
def test_bootstrap_rejects_invalid_iteration_confidence_and_seed(kwargs, message) -> None:
    from src.evaluation.bootstrap import cluster_bootstrap_ci

    with pytest.raises(ValueError, match=message):
        cluster_bootstrap_ci(
            [1.0, 2.0],
            [1.0, 2.0],
            ["A", "B"],
            **kwargs,
        )


@pytest.mark.parametrize("groups", [["A", ""], ["A", "  "], ["A", None]])
def test_bootstrap_rejects_blank_or_null_group_ids(groups) -> None:
    from src.evaluation.bootstrap import cluster_bootstrap_ci

    with pytest.raises(ValueError, match="source_group.*null or blank"):
        cluster_bootstrap_ci([1.0, 2.0], [1.0, 2.0], groups, n_resamples=2)


def test_bootstrap_rejects_non_vector_group_ids() -> None:
    from src.evaluation.bootstrap import cluster_bootstrap_ci

    with pytest.raises(ValueError, match="source_group.*one-dimensional"):
        cluster_bootstrap_ci(
            [1.0, 2.0],
            [1.0, 2.0],
            [["A"], ["B"]],
            n_resamples=2,
        )


def test_bootstrap_rejects_length_mismatch() -> None:
    from src.evaluation.bootstrap import cluster_bootstrap_ci

    with pytest.raises(ValueError, match="equal length"):
        cluster_bootstrap_ci(
            [1.0, 2.0],
            [1.0],
            ["A", "B"],
            n_resamples=2,
        )


def test_bootstrap_raises_instead_of_dropping_an_invalid_replicate() -> None:
    from src.evaluation.bootstrap import cluster_bootstrap_ci

    calls = 0

    def invalid_after_point_estimate(truth, predictions):
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else float("nan")

    with pytest.raises(ValueError, match="replicate 0.*non-finite"):
        cluster_bootstrap_ci(
            [1.0, 2.0],
            [1.0, 2.0],
            ["A", "B"],
            metric=invalid_after_point_estimate,
            n_resamples=3,
        )
