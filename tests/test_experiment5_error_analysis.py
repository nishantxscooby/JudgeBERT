"""Tests for Experiment 5 (Qualitative Error Analysis)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.experiments.common import ExperimentBlockedError
from src.experiments.experiment5_error_analysis import (
    MIN_CASES_REQUIRED,
    main,
    run_experiment5,
    sample_qualitative_cases,
)
from src.utils.io import TODO_HUMAN_ANNOTATION, read_json


def test_exp5_blocked_on_unannotated_dataset(
    tmp_path: Path, unannotated_corpus_env
) -> None:
    """Final mode execution must fail loudly with TODO_HUMAN_ANNOTATION when gold is absent."""
    with pytest.raises(ExperimentBlockedError, match=f"Blocked: {TODO_HUMAN_ANNOTATION}"):
        run_experiment5(
            cfg=unannotated_corpus_env["cfg"],
            output_dir=tmp_path,
            smoke_test=False,
        )

    blocked_json = tmp_path / "blocked.json"
    assert blocked_json.is_file()
    payload = read_json(blocked_json)
    assert payload["status"] == "BLOCKED"
    assert payload["code"] == TODO_HUMAN_ANNOTATION


def test_exp5_never_uses_draft_error_type() -> None:
    """Sampling must reject inputs where human error_type is missing even if draft_error_type exists."""
    df_with_only_draft = pd.DataFrame({
        "pair_id": ["P1", "P2"],
        "original_text": ["A", "B"],
        "simplified_text": ["A1", "B1"],
        "human_lmp_score": [8.0, 4.0],
        "draft_error_type": ["Negation", "Scope"],
        # Notice: human error_type is absent
    })

    with pytest.raises(ValueError, match="'error_type' column is absent or empty"):
        sample_qualitative_cases(df_with_only_draft, n_examples=2)


def test_sample_qualitative_cases_logic() -> None:
    """Sampling must prioritize high-error cases while covering distinct error phenomena."""
    mock_df = pd.DataFrame({
        "pair_id": [f"ID_{i:02d}" for i in range(20)],
        "error_type": ["Obligation", "Negation", "Scope", "Condition"] * 5,
        "human_lmp_score": [5.0] * 20,
        "prediction": [5.0 + (i * 0.2) for i in range(20)],
    })

    sampled, counts = sample_qualitative_cases(mock_df, n_examples=16, min_required=15)
    assert len(sampled) >= 15
    assert len(counts) >= 4  # All 4 distinct error types covered
    # Highest absolute error case should be at top
    assert sampled["absolute_error"].iloc[0] >= sampled["absolute_error"].iloc[-1]


def test_exp5_smoke_mode_sampling(tmp_path: Path) -> None:
    """Smoke mode must sample >= 15 cases and output all machine-readable and report files."""
    result = run_experiment5(output_dir=tmp_path, smoke_test=True, n_examples=20)

    assert result["n_sampled_cases"] >= MIN_CASES_REQUIRED
    assert len(result["cases"]) >= MIN_CASES_REQUIRED

    cases_json = tmp_path / "error_analysis_cases.json"
    cases_csv = tmp_path / "error_analysis_cases.csv"
    report_md = tmp_path / "error_analysis_report.md"
    prov_file = tmp_path / "provenance.json"

    assert cases_json.is_file()
    assert cases_csv.is_file()
    assert report_md.is_file()
    assert prov_file.is_file()

    # Verify preserved fields in cases table
    cases_df = pd.read_csv(cases_csv)
    expected_cols = {
        "pair_id",
        "language",
        "legal_domain",
        "original_text",
        "simplified_text",
        "human_lmp_score",
        "prediction",
        "absolute_error",
        "error_type",
    }
    assert expected_cols <= set(cases_df.columns)


def test_exp5_cli_blocked_exit_code(
    tmp_path: Path, unannotated_corpus_env, capsys: pytest.CaptureFixture
) -> None:
    """CLI without smoke-test flag must return 1 and state blocked status when unannotated."""
    exit_code = main([
        "--config",
        str(unannotated_corpus_env["cfg_file"]),
        "--output-dir",
        str(tmp_path),
    ])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert f"Blocked: {TODO_HUMAN_ANNOTATION}" in captured.err


def test_exp5_cli_smoke_exit_code(tmp_path: Path) -> None:
    """CLI with --smoke-test flag must return 0."""
    exit_code = main(["--output-dir", str(tmp_path), "--smoke-test"])
    assert exit_code == 0


def test_exp5_production_gold_error_sampling() -> None:
    """Production test split must sample >= 15 cases covering genuine human error types."""
    from src.data.load_data import assert_gold, load_master

    gold = load_master(subset="gold")
    assert_gold(gold)
    test_gold = gold[gold["split"] == "test"].copy()

    # Create mock prediction for test rows to verify sampling
    test_gold["prediction"] = test_gold["human_lmp_score"].to_numpy()
    sampled, counts = sample_qualitative_cases(test_gold, n_examples=20, min_required=15)

    assert len(sampled) >= 15
    assert sampled["human_lmp_score"].notna().all()
    assert sampled["error_type"].notna().all()
    assert len(counts) >= 5  # Diverse error phenomena represented
