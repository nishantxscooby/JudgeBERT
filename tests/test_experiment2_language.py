"""Tests for Experiment 2 (Cross-lingual Comparison: English vs Hindi)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.common import ExperimentBlockedError
from src.experiments.experiment2_language import (
    PROVENANCE_LIMITATION_DISCLAIMER,
    main,
    run_experiment2,
)
from src.utils.io import TODO_HUMAN_ANNOTATION, read_json


def test_exp2_blocked_on_unannotated_dataset(
    tmp_path: Path, unannotated_corpus_env
) -> None:
    """Final mode execution must fail loudly with TODO_HUMAN_ANNOTATION when gold is absent."""
    with pytest.raises(ExperimentBlockedError, match=f"Blocked: {TODO_HUMAN_ANNOTATION}"):
        run_experiment2(
            cfg=unannotated_corpus_env["cfg"],
            output_dir=tmp_path,
            smoke_test=False,
        )

    blocked_json = tmp_path / "blocked.json"
    assert blocked_json.is_file()
    payload = read_json(blocked_json)
    assert payload["status"] == "BLOCKED"
    assert payload["code"] == TODO_HUMAN_ANNOTATION


def test_exp2_smoke_mode_breakdown(tmp_path: Path) -> None:
    """Smoke mode must compute English, Hindi, and gap metrics."""
    result = run_experiment2(output_dir=tmp_path, smoke_test=True)

    assert result["split_evaluated"] == "test"
    assert set(result["languages"]) == {"en", "hi"}
    assert result["sample_counts"]["en"] > 0
    assert result["sample_counts"]["hi"] > 0

    # Output files
    metrics_file = tmp_path / "language_metrics.json"
    preds_file = tmp_path / "language_predictions.csv"
    report_file = tmp_path / "language_report.md"
    prov_file = tmp_path / "provenance.json"

    assert metrics_file.is_file()
    assert preds_file.is_file()
    assert report_file.is_file()
    assert prov_file.is_file()

    # Content checks
    data = read_json(metrics_file)
    assert "metrics_by_language" in data
    assert "english" in data["metrics_by_language"]
    assert "hindi" in data["metrics_by_language"]
    assert "language_gap" in data

    # Check that provenance limitation is present in report and JSON
    assert data["provenance_limitation"] == PROVENANCE_LIMITATION_DISCLAIMER
    report_text = report_file.read_text(encoding="utf-8")
    assert "reports/source_provenance.md" in report_text
    assert "Provenance Limitation" in report_text


def test_exp2_cli_blocked_exit_code(
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


def test_exp2_cli_smoke_exit_code(tmp_path: Path) -> None:
    """CLI with --smoke-test flag must return 0."""
    exit_code = main(["--output-dir", str(tmp_path), "--smoke-test"])
    assert exit_code == 0


def test_exp2_production_gold_integrity() -> None:
    """Production corpus must have balanced English and Hindi splits with 100% human gold."""
    from src.data import schema
    from src.data.load_data import assert_gold, load_master

    gold = load_master(subset="gold")
    assert len(gold) == 340
    assert_gold(gold)
    assert schema.is_gold(gold).all()

    counts = gold["language"].value_counts().to_dict()
    assert counts.get("en") == 170
    assert counts.get("hi") == 170

    test_gold = gold[gold["split"] == "test"]
    assert len(test_gold) == 56
    test_counts = test_gold["language"].value_counts().to_dict()
    assert test_counts.get("en") == 28
    assert test_counts.get("hi") == 28
