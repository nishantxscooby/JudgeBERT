"""Tests for Experiment 1 (Baselines vs JUDGEBERT)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.common import ExperimentBlockedError
from src.experiments.experiment1_baselines import main, run_experiment1
from src.utils.io import TODO_HUMAN_ANNOTATION, read_json


def test_exp1_blocked_on_unannotated_dataset(
    tmp_path: Path, unannotated_corpus_env
) -> None:
    """Final mode execution must fail loudly with TODO_HUMAN_ANNOTATION when gold is absent."""
    with pytest.raises(ExperimentBlockedError, match=f"Blocked: {TODO_HUMAN_ANNOTATION}"):
        run_experiment1(
            cfg=unannotated_corpus_env["cfg"],
            output_dir=tmp_path,
            smoke_test=False,
        )

    # Must have recorded the blocked state
    blocked_json = tmp_path / "blocked.json"
    assert blocked_json.is_file()
    payload = read_json(blocked_json)
    assert payload["status"] == "BLOCKED"
    assert payload["code"] == TODO_HUMAN_ANNOTATION


def test_exp1_smoke_mode_orchestration(tmp_path: Path) -> None:
    """Smoke mode must successfully exercise pipeline orchestration using synthetic fixtures."""
    result = run_experiment1(output_dir=tmp_path, smoke_test=True)

    assert result["split_evaluated"] == "test"
    assert set(result["models_evaluated"]) == {"tfidf", "sbert", "judgebert"}

    # Verify output artifacts exist
    metrics_file = tmp_path / "metrics.json"
    preds_file = tmp_path / "predictions.csv"
    report_file = tmp_path / "report.md"
    prov_file = tmp_path / "provenance.json"

    assert metrics_file.is_file()
    assert preds_file.is_file()
    assert report_file.is_file()
    assert prov_file.is_file()

    # Check metrics structure
    metrics_data = read_json(metrics_file)
    assert "metrics" in metrics_data
    for model in ("tfidf", "sbert", "judgebert"):
        assert model in metrics_data["metrics"]
        m = metrics_data["metrics"][model]["overall"]
        assert "rmse" in m
        assert "pearson" in m
        assert "spearman" in m

    # Check non-research marker in smoke report
    report_text = report_file.read_text(encoding="utf-8")
    assert "NON-RESEARCH SMOKE TEST OUTPUT" in report_text


def test_exp1_cli_blocked_exit_code(
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


def test_exp1_cli_smoke_exit_code(tmp_path: Path) -> None:
    """CLI with --smoke-test flag must return 0."""
    exit_code = main(["--output-dir", str(tmp_path), "--smoke-test"])
    assert exit_code == 0


def test_exp1_production_gold_integrity() -> None:
    """Production corpus must pass assert_gold and have 340 human annotations without draft leakage."""
    from src.data import schema
    from src.data.load_data import assert_gold, load_master

    gold = load_master(subset="gold")
    assert len(gold) == 340
    assert_gold(gold)
    assert schema.is_gold(gold).all()
    assert set(gold["annotation_status"].unique()) <= {"human_single", "human_consensus"}
    assert (gold["human_lmp_score"] != gold["draft_lmp_score"]).any()
