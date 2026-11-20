"""Tests for leakage-safe three-way decision evaluation."""

from __future__ import annotations

import pandas as pd
import pytest


def test_three_way_accuracy_and_macro_f1_use_all_legal_classes() -> None:
    from src.evaluation.decision_metrics import decision_metrics

    truth = [
        "Preserved",
        "Partially Preserved",
        "Not Preserved",
        "Not Preserved",
    ]
    predictions = [
        "Preserved",
        "Not Preserved",
        "Not Preserved",
        "Partially Preserved",
    ]

    result = decision_metrics(truth, predictions)

    assert result == {
        "accuracy": pytest.approx(0.5),
        "macro_f1": pytest.approx(0.5),
    }


def test_individual_decision_helpers_match_the_metric_bundle() -> None:
    from src.evaluation.decision_metrics import decision_accuracy, macro_f1

    truth = ["Preserved", "Partially Preserved", "Not Preserved"]
    predicted = ["Preserved", "Not Preserved", "Not Preserved"]

    assert decision_accuracy(truth, predicted) == pytest.approx(2.0 / 3.0)
    assert macro_f1(truth, predicted) == pytest.approx(5.0 / 9.0)


@pytest.mark.parametrize(
    ("truth", "predictions", "message"),
    [
        ([], [], "nonempty"),
        (["Preserved"], ["Preserved", "Not Preserved"], "equal length"),
        (["Unknown"], ["Preserved"], "illegal decision"),
        (["Preserved"], [None], "illegal decision"),
    ],
)
def test_decision_metrics_reject_invalid_labels(truth, predictions, message) -> None:
    from src.evaluation.decision_metrics import decision_metrics

    with pytest.raises(ValueError, match=message):
        decision_metrics(truth, predictions)


def test_thresholds_fit_validation_only_and_record_exact_fit_provenance() -> None:
    from src.evaluation.decision_metrics import (
        apply_decision_thresholds,
        fit_decision_thresholds,
    )

    validation = pd.DataFrame(
        {
            "pair_id": ["V1", "V2", "V3", "V4", "V5", "V6"],
            "source_group": ["g1", "g1", "g2", "g3", "g4", "g5"],
            "split": ["validation"] * 6,
            "prediction": [2.0, 4.5, 5.0, 7.0, 8.0, 9.0],
            "decision": [
                "Not Preserved",
                "Not Preserved",
                "Partially Preserved",
                "Partially Preserved",
                "Preserved",
                "Preserved",
            ],
        }
    )

    thresholds = fit_decision_thresholds(
        validation,
        provenance={"prediction_artifact_sha256": "a" * 64},
    )
    applied = apply_decision_thresholds([4.999, 5.0, 7.999, 8.0], thresholds)
    payload = thresholds.to_dict()

    assert thresholds.partial_min == 5.0
    assert thresholds.preserved_min == 8.0
    assert applied == [
        "Not Preserved",
        "Partially Preserved",
        "Partially Preserved",
        "Preserved",
    ]
    assert thresholds.fit_pair_ids == tuple(validation["pair_id"])
    assert thresholds.fit_source_groups == tuple(validation["source_group"])
    assert thresholds.fit_splits == ("validation",) * 6
    assert payload["fit_split"] == "validation"
    assert payload["fit_records"] == [
        {"pair_id": row.pair_id, "source_group": row.source_group, "split": "validation"}
        for row in validation.itertuples(index=False)
    ]
    assert payload["provenance"]["prediction_artifact_sha256"] == "a" * 64
    assert payload["validation_metrics"] == {
        "accuracy": 1.0,
        "macro_f1": 1.0,
    }


@pytest.mark.parametrize(
    "splits",
    [
        ["train", "train"],
        ["validation", "test"],
        ["validation", "dev"],
        ["validation", None],
    ],
)
def test_threshold_fit_rejects_nonvalidation_mixed_and_unknown_splits(splits) -> None:
    from src.evaluation.decision_metrics import fit_decision_thresholds

    frame = pd.DataFrame(
        {
            "pair_id": ["A", "B"],
            "source_group": ["g1", "g2"],
            "split": splits,
            "prediction": [3.0, 8.0],
            "decision": ["Not Preserved", "Preserved"],
        }
    )

    with pytest.raises(ValueError, match="validation rows only"):
        fit_decision_thresholds(frame)


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("pair_id", None, "pair_id.*null or blank"),
        ("pair_id", "  ", "pair_id.*null or blank"),
        ("source_group", None, "source_group.*null or blank"),
        ("source_group", "", "source_group.*null or blank"),
        ("prediction", float("nan"), "scores.*finite"),
        ("prediction", float("inf"), "scores.*finite"),
        ("prediction", 0.99, "scores.*1.*10"),
        ("prediction", 10.01, "scores.*1.*10"),
    ],
)
def test_threshold_fit_rejects_invalid_identity_group_and_scores(
    column, bad_value, message
) -> None:
    from src.evaluation.decision_metrics import fit_decision_thresholds

    frame = pd.DataFrame(
        {
            "pair_id": ["V1", "V2", "V3"],
            "source_group": ["g1", "g2", "g3"],
            "split": ["validation"] * 3,
            "prediction": [2.0, 6.0, 9.0],
            "decision": [
                "Not Preserved",
                "Partially Preserved",
                "Preserved",
            ],
        }
    )
    frame.loc[1, column] = bad_value

    with pytest.raises(ValueError, match=message):
        fit_decision_thresholds(frame)


def test_threshold_fit_rejects_duplicate_pair_ids() -> None:
    from src.evaluation.decision_metrics import fit_decision_thresholds

    frame = pd.DataFrame(
        {
            "pair_id": ["V1", "V1", "V3"],
            "source_group": ["g1", "g2", "g3"],
            "split": ["validation"] * 3,
            "prediction": [2.0, 6.0, 9.0],
            "decision": [
                "Not Preserved",
                "Partially Preserved",
                "Preserved",
            ],
        }
    )

    with pytest.raises(ValueError, match="Duplicate pair_id"):
        fit_decision_thresholds(frame)


@pytest.mark.parametrize("column", ["draft_lmp_score", "draft_error_type"])
def test_threshold_fit_never_uses_draft_fields(column) -> None:
    from src.evaluation.decision_metrics import fit_decision_thresholds

    frame = pd.DataFrame(
        {
            "pair_id": ["V1", "V2", "V3"],
            "source_group": ["g1", "g2", "g3"],
            "split": ["validation"] * 3,
            "prediction": [2.0, 6.0, 9.0],
            "decision": [
                "Not Preserved",
                "Partially Preserved",
                "Preserved",
            ],
            "draft_lmp_score": [9.0, 9.0, 9.0],
            "draft_error_type": ["None", "None", "None"],
        }
    )
    kwargs = (
        {"score_column": column}
        if column == "draft_lmp_score"
        else {"decision_column": column}
    )

    with pytest.raises(ValueError, match="draft.*never"):
        fit_decision_thresholds(frame, **kwargs)


def test_decision_threshold_objects_and_application_validate_score_range() -> None:
    from src.evaluation.decision_metrics import (
        DecisionThresholds,
        apply_decision_thresholds,
    )

    with pytest.raises(ValueError, match="partial_min.*preserved_min"):
        DecisionThresholds(
            partial_min=8.0,
            preserved_min=5.0,
            fit_records=(),
            validation_metrics={},
            provenance={},
        )

    thresholds = DecisionThresholds(
        partial_min=5.0,
        preserved_min=8.0,
        fit_records=(),
        validation_metrics={},
        provenance={},
    )
    for bad_scores, message in [
        ([], "nonempty"),
        ([float("nan")], "finite"),
        ([float("inf")], "finite"),
        ([0.99], "within.*1.*10"),
        ([10.01], "within.*1.*10"),
    ]:
        with pytest.raises(ValueError, match=message):
            apply_decision_thresholds(bad_scores, thresholds)
