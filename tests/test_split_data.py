"""Tests for grouped, stratified train/validation/test splitting.

Two properties carry the weight here: a source group must never span two
splits (or every experiment is evaluating on data the model has seen), and the
assignment must be reproducible from the seed alone (or the paper's numbers
cannot be regenerated).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import split_data


@pytest.fixture
def corpus() -> pd.DataFrame:
    """A synthetic corpus with sibling groups and both languages.

    Each provision contributes an English and a Hindi row sharing one
    ``source_group``, which is the structure the real corpus has and the
    structure the leakage guarantee is about.
    """
    rows = []
    for i in range(60):
        group = f"ACT::{(i // 2) + 1}"
        language = "en" if i % 2 == 0 else "hi"
        rows.append(
            {
                "pair_id": f"{'ENG' if language == 'en' else 'HIN'}_CON_{i:03d}",
                "language": language,
                "legal_domain": "contract" if i % 3 else "cyber",
                # Deliberately land on all three score bands.
                "draft_lmp_score": [2, 5, 9][i % 3],
                "score_bucket": ["low", "mid", "high"][i % 3],
                "source_group": group,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The leakage guarantee
# ---------------------------------------------------------------------------
def test_no_source_group_spans_two_splits(corpus):
    out = split_data.assign_splits(corpus, seed=7)
    spans = out.groupby("source_group")["split"].nunique()
    assert (spans == 1).all(), out.groupby("source_group")["split"].unique()[spans > 1]


def test_every_row_receives_a_split(corpus):
    out = split_data.assign_splits(corpus, seed=7)
    assert out["split"].notna().all()
    assert set(out["split"]) <= {"train", "validation", "test"}


def test_siblings_land_in_the_same_split(corpus):
    """The English and Hindi halves of one provision are siblings."""
    out = split_data.assign_splits(corpus, seed=7)
    group_splits = out.groupby("source_group")["split"].nunique()
    assert set(group_splits) == {1}


def test_diagnostics_report_zero_spanning_groups(corpus):
    out = split_data.assign_splits(corpus, seed=7)
    diag = split_data.split_diagnostics(out)
    assert diag["groups_spanning_multiple_splits"] == 0
    assert diag["n_groups"] == out["source_group"].nunique()


def test_diagnostics_do_report_leakage_when_it_is_deliberately_introduced(corpus):
    """A diagnostic that can only ever print zero proves nothing.

    This test corrupts one group by hand and checks the detector notices, so
    the zero in the test above carries information.
    """
    out = split_data.assign_splits(corpus, seed=7)
    group = out["source_group"].iloc[0]
    members = out.index[out["source_group"] == group]
    out.loc[members[0], "split"] = "test"
    out.loc[members[1], "split"] = "train"

    diag = split_data.split_diagnostics(out)
    assert diag["groups_spanning_multiple_splits"] == 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def test_the_same_seed_gives_the_same_assignment(corpus):
    first = split_data.assign_splits(corpus, seed=11)
    second = split_data.assign_splits(corpus, seed=11)
    assert first["split"].tolist() == second["split"].tolist()


def test_a_different_seed_changes_some_assignments(corpus):
    """A split that ignores the seed is not reproducible, it is fixed."""
    first = split_data.assign_splits(corpus, seed=1)
    second = split_data.assign_splits(corpus, seed=2)
    assert first["split"].tolist() != second["split"].tolist()


def test_split_data_does_not_mutate_the_input_frame(corpus):
    before = corpus.copy()
    split_data.assign_splits(corpus, seed=7)
    pd.testing.assert_frame_equal(corpus, before)


def test_assign_splits_requires_the_group_column():
    df = pd.DataFrame({"pair_id": ["X"], "language": ["en"]})
    with pytest.raises(KeyError, match="source_group"):
        split_data.assign_splits(df, seed=1)


# ---------------------------------------------------------------------------
# Allocation shape
# ---------------------------------------------------------------------------
def test_all_three_splits_are_populated(corpus):
    out = split_data.assign_splits(corpus, seed=7)
    assert set(out["split"]) == {"train", "validation", "test"}


def test_roughly_seventy_fifteen_fifteen(corpus):
    """Grouped allocation cannot hit the fractions exactly; it must be close."""
    out = split_data.assign_splits(corpus, seed=7)
    shares = out["split"].value_counts(normalize=True)
    assert 0.60 <= shares["train"] <= 0.80
    assert 0.08 <= shares.get("validation", 0) <= 0.25
    assert 0.08 <= shares.get("test", 0) <= 0.25


def test_both_languages_appear_in_every_split(corpus):
    """Experiment 2 compares languages, so each split needs both."""
    out = split_data.assign_splits(corpus, seed=7)
    grid = pd.crosstab(out["split"], out["language"])
    assert (grid > 0).all().all(), grid


def test_a_single_group_goes_to_train():
    """A lone group cannot be split, and validation/test must stay clean."""
    df = pd.DataFrame(
        {
            "pair_id": ["A", "B"],
            "language": ["en", "hi"],
            "legal_domain": ["contract", "contract"],
            "score_bucket": ["mid", "mid"],
            "source_group": ["ONLY::1", "ONLY::1"],
        }
    )
    out = split_data.assign_splits(df, seed=1)
    assert set(out["split"]) == {"train"}


def test_two_groups_in_a_stratum_do_not_both_go_to_train():
    df = pd.DataFrame(
        {
            "pair_id": ["A", "B"],
            "language": ["en", "en"],
            "legal_domain": ["contract", "contract"],
            "score_bucket": ["mid", "mid"],
            "source_group": ["ACT::1", "ACT::2"],
        }
    )
    out = split_data.assign_splits(df, seed=1)
    assert out["split"].nunique() == 2


def test_stratification_keeps_the_score_bands_present_in_every_split(corpus):
    out = split_data.assign_splits(corpus, seed=7)
    grid = pd.crosstab(out["split"], out["score_bucket"])
    assert (grid > 0).all().all(), grid


# ---------------------------------------------------------------------------
# Diagnostics rendering
# ---------------------------------------------------------------------------
def test_render_diagnostics_produces_markdown_tables(corpus):
    out = split_data.assign_splits(corpus, seed=7)
    text = split_data.render_diagnostics(split_data.split_diagnostics(out))
    assert "## Split diagnostics" in text
    assert "Language x split" in text
    assert "| Split |" in text


def test_group_table_gives_each_group_exactly_one_row(corpus):
    table = split_data._group_table(corpus, "source_group")
    assert len(table) == corpus["source_group"].nunique()
    assert table["source_group"].is_unique
    assert (table["n_members"] >= 1).all()


def test_group_table_uses_a_modal_stratum_for_mixed_groups(corpus):
    """Two English siblings and one Hindi: the group's stratum must be stable."""
    table = split_data._group_table(corpus, "source_group")
    assert table["stratum"].notna().all()
    assert not table["stratum"].str.contains("nan").any()


def test_effective_split_seed_uses_the_configured_split_seed_by_default():
    cfg = {"project": {"seed": 42}, "splits": {"seed": 17}}

    assert split_data.effective_split_seed(cfg, None) == 17


def test_effective_split_seed_prefers_an_explicit_cli_override():
    cfg = {"project": {"seed": 42}, "splits": {"seed": 17}}

    assert split_data.effective_split_seed(cfg, 99) == 99
