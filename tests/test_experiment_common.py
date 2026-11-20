"""Tests for common experiment infrastructure and data integrity safeguards."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import schema
from src.data.load_data import assert_gold, load_master
from src.experiments.common import (
    ExperimentBlockedError,
    assert_no_draft_or_synthetic_leakage,
    build_experiment_provenance,
    create_mock_predictions,
    create_synthetic_experiment_fixture,
    get_dataset_fingerprint,
    load_experiment_config,
    load_gold_or_block,
    validate_split_and_ids,
    write_blocked_status,
)
from src.utils.io import TODO_HUMAN_ANNOTATION, read_json


def test_gold_loading_blocks_on_unannotated_dataset(unannotated_corpus_env) -> None:
    """Loading gold on an unannotated dataset must immediately raise ExperimentBlockedError."""
    with pytest.raises(ExperimentBlockedError, match=f"Blocked: {TODO_HUMAN_ANNOTATION}"):
        load_gold_or_block(unannotated_corpus_env["cfg"], context="Unit Test")


def test_gold_loading_succeeds_on_production_gold_dataset() -> None:
    """Loading gold on the production dataset must succeed with 340 gold rows."""
    cfg = load_experiment_config()
    gold = load_gold_or_block(cfg, context="Production Unit Test")
    assert len(gold) == 340
    assert schema.is_gold(gold).all()
    assert gold["human_lmp_score"].notna().all()


def test_draft_label_rejection() -> None:
    """A dataset containing only draft labels must be strictly rejected."""
    draft_df = pd.DataFrame({
        "pair_id": ["P1", "P2"],
        "draft_lmp_score": [8.0, 4.0],
        "annotation_status": ["draft", "draft"],
    })
    with pytest.raises(ValueError, match="Disallowed annotation status"):
        assert_no_draft_or_synthetic_leakage(draft_df, context="draft test")


def test_synthetic_fixture_generation() -> None:
    """The synthetic test fixture must be well-formed, balanced, and non-empty."""
    fixture = create_synthetic_experiment_fixture(n_per_split=6, seed=42)

    assert len(fixture) == 18
    assert set(fixture["split"].unique()) == {"train", "validation", "test"}
    assert set(fixture["language"].unique()) == {"en", "hi"}
    assert set(fixture["legal_domain"].unique()) == {"contract", "cyber", "sale_of_goods"}

    # All rows must have required fields populated
    assert fixture["pair_id"].notna().all()
    assert fixture["original_text"].notna().all()
    assert fixture["simplified_text"].notna().all()
    assert fixture["human_lmp_score"].between(1.0, 10.0).all()
    assert fixture["decision"].isin(["Preserved", "Partially Preserved", "Not Preserved"]).all()


def test_mock_predictions_generation() -> None:
    """Mock predictions must align with the fixture and contain valid scores."""
    fixture = create_synthetic_experiment_fixture(n_per_split=4, seed=42)
    preds = create_mock_predictions(fixture, models=["tfidf", "sbert", "judgebert"], seed=42)

    assert set(preds.keys()) == {"tfidf", "sbert", "judgebert"}
    for model_name, pred_df in preds.items():
        assert len(pred_df) == len(fixture)
        assert (pred_df["pair_id"] == fixture["pair_id"]).all()
        assert pred_df["prediction"].between(1.0, 10.0).all()
        assert (pred_df["model_name"] == model_name).all()


def test_validate_split_and_ids() -> None:
    """Validates unique IDs and recognized split vocabulary."""
    valid_df = pd.DataFrame({
        "pair_id": ["ID_01", "ID_02"],
        "split": ["train", "test"],
    })
    ids = validate_split_and_ids(valid_df)
    assert ids == ["ID_01", "ID_02"]

    # Reject duplicate IDs
    dup_df = pd.DataFrame({"pair_id": ["ID_01", "ID_01"], "split": ["train", "test"]})
    with pytest.raises(ValueError, match="Duplicate pair_id"):
        validate_split_and_ids(dup_df)

    # Reject invalid split
    bad_split_df = pd.DataFrame({"pair_id": ["ID_01"], "split": ["unknown_split"]})
    with pytest.raises(ValueError, match="Unknown split value"):
        validate_split_and_ids(bad_split_df)


def test_provenance_structure() -> None:
    """Complete provenance block must be populated with all required fields."""
    cfg = load_experiment_config()
    prov = build_experiment_provenance(
        cfg,
        experiment_name="Unit Test Experiment",
        model_identity="test_model",
        is_smoke=True,
    )

    required_keys = {
        "generated_at_utc",
        "git_revision",
        "python_version",
        "platform",
        "config_fingerprint",
        "random_seed",
        "dataset_fingerprint",
        "model_identity",
        "experiment_name",
        "human_gold_required",
        "is_smoke",
        "research_valid",
        "disclaimer",
    }
    assert required_keys <= set(prov.keys())
    assert prov["is_smoke"] is True
    assert prov["research_valid"] is False
    assert "SYNTHETIC NON-RESEARCH DATA" in prov["disclaimer"]


def test_write_blocked_status(tmp_path: Path) -> None:
    """Blocked status writing produces valid blocked.json and blocked.md."""
    cfg = load_experiment_config()
    prov = build_experiment_provenance(cfg, experiment_name="Blocked Test", is_smoke=False)

    json_path = write_blocked_status(
        tmp_path,
        experiment_name="Blocked Test",
        reason="Human annotation pending",
        provenance=prov,
    )

    assert json_path.is_file()
    payload = read_json(json_path)
    assert payload["status"] == "BLOCKED"
    assert payload["code"] == TODO_HUMAN_ANNOTATION

    md_path = tmp_path / "blocked.md"
    assert md_path.is_file()
    md_text = md_path.read_text(encoding="utf-8")
    assert TODO_HUMAN_ANNOTATION in md_text
    assert "Blocked Test — BLOCKED" in md_text


def test_no_accidental_mutation_of_master_corpus() -> None:
    """Master corpus file must not be mutated by loading or validation operations."""
    master_path = Path("data/processed/lmp_pairs.csv")
    if master_path.is_file():
        hash_before = hashlib.sha256(master_path.read_bytes()).hexdigest()
        df = load_master(subset="all")
        assert_gold(schema.gold_subset(df))
        hash_after = hashlib.sha256(master_path.read_bytes()).hexdigest()
        assert hash_before == hash_after
        assert len(df) == 340
