"""Tests for strict pair-ID prediction alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _canonical_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["P1", "P2", "P3"],
            "language": ["en", "hi", "en"],
            "legal_domain": ["contract", "cyber", "sale_of_goods"],
            "source_group": ["g1", "g2", "g3"],
            "split": ["train", "validation", "test"],
            "human_lmp_score": [2.0, 6.0, 9.0],
            "decision": [
                "Not Preserved",
                "Partially Preserved",
                "Preserved",
            ],
            "annotation_status": ["human_consensus"] * 3,
            "draft_lmp_score": [9.0, 9.0, 1.0],
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": ["P3", "P1", "P2"],
            "language": ["en", "en", "hi"],
            "legal_domain": ["sale_of_goods", "contract", "cyber"],
            "source_group": ["g3", "g1", "g2"],
            "split": ["test", "train", "validation"],
            "prediction": [8.5, 2.5, 6.5],
        }
    )


def test_predictions_align_by_pair_id_and_preserve_canonical_order_and_metadata() -> None:
    from src.evaluation.alignment import align_predictions

    aligned = align_predictions(_canonical_pairs(), _predictions())

    assert aligned.columns.tolist() == [
        "pair_id",
        "language",
        "legal_domain",
        "source_group",
        "split",
        "human_lmp_score",
        "decision",
        "annotation_status",
        "prediction",
    ]
    assert aligned["pair_id"].tolist() == ["P1", "P2", "P3"]
    assert aligned["prediction"].tolist() == [2.5, 6.5, 8.5]
    assert aligned["language"].tolist() == ["en", "hi", "en"]
    assert "draft_lmp_score" not in aligned.columns


def test_alignment_accepts_baseline_contract_without_prediction_source_group() -> None:
    from src.evaluation.alignment import align_predictions

    aligned = align_predictions(
        _canonical_pairs(), _predictions().drop(columns=["source_group"])
    )

    assert aligned["source_group"].tolist() == ["g1", "g2", "g3"]
    assert aligned["prediction"].tolist() == [2.5, 6.5, 8.5]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.assign(pair_id=[None, "P1", "P2"]), "null or blank pair_id"),
        (lambda frame: frame.assign(pair_id=["  ", "P1", "P2"]), "null or blank pair_id"),
        (lambda frame: frame.assign(pair_id=["P1", "P1", "P2"]), "Duplicate pair_id"),
        (lambda frame: frame.iloc[:-1].copy(), "Missing prediction IDs"),
        (lambda frame: frame.assign(pair_id=["UNKNOWN", "P1", "P2"]), "Unknown/extra"),
        (lambda frame: frame.assign(split=["train", "train", "validation"]), "split.*mismatch"),
        (lambda frame: frame.assign(language=["hi", "en", "hi"]), "language.*mismatch"),
        (
            lambda frame: frame.assign(
                legal_domain=["contract", "contract", "cyber"]
            ),
            "legal_domain.*mismatch",
        ),
        (lambda frame: frame.assign(source_group=["g1", "g1", "g2"]), "source_group.*mismatch"),
        (lambda frame: frame.assign(prediction=[np.nan, 2.5, 6.5]), "finite"),
        (lambda frame: frame.assign(prediction=[np.inf, 2.5, 6.5]), "finite"),
        (lambda frame: frame.assign(prediction=[10.01, 2.5, 6.5]), "within.*1.*10"),
    ],
)
def test_alignment_rejects_invalid_ids_metadata_and_scores(mutator, message) -> None:
    from src.evaluation.alignment import align_predictions

    with pytest.raises(ValueError, match=message):
        align_predictions(_canonical_pairs(), mutator(_predictions()))


def test_alignment_rejects_missing_metadata_and_label_leakage() -> None:
    from src.evaluation.alignment import align_predictions

    with pytest.raises(ValueError, match="missing required metadata.*split"):
        align_predictions(_canonical_pairs(), _predictions().drop(columns=["split"]))

    with pytest.raises(ValueError, match="forbidden label.*draft_lmp_score"):
        align_predictions(
            _canonical_pairs(),
            _predictions().assign(draft_lmp_score=[1.0, 1.0, 1.0]),
        )


@pytest.mark.parametrize("column", ["ai_label", "synthetic_label", "manual_label"])
def test_alignment_rejects_nonhuman_label_columns(column) -> None:
    from src.evaluation.alignment import align_predictions

    predictions = _predictions()
    predictions[column] = ["invented"] * len(predictions)

    with pytest.raises(ValueError, match=f"forbidden label.*{column}"):
        align_predictions(_canonical_pairs(), predictions)


def test_alignment_rejects_source_groups_that_cross_splits() -> None:
    from src.evaluation.alignment import align_predictions

    canonical = _canonical_pairs()
    canonical.loc[2, "source_group"] = "g2"
    predictions = _predictions()
    predictions.loc[predictions["pair_id"] == "P3", "source_group"] = "g2"

    with pytest.raises(ValueError, match="source_group.*cross splits"):
        align_predictions(canonical, predictions)


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, 0.99, 10.01])
def test_alignment_rejects_invalid_canonical_human_scores(bad_score) -> None:
    from src.evaluation.alignment import align_predictions

    canonical = _canonical_pairs()
    canonical.loc[1, "human_lmp_score"] = bad_score

    with pytest.raises(ValueError, match="human_lmp_score.*finite.*1.*10"):
        align_predictions(canonical, _predictions())
