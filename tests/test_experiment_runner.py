"""Tests for the master experiment runner (src.experiments.run_all)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.run_all import main, run_all_experiments
from src.utils.io import TODO_HUMAN_ANNOTATION, read_json


def test_run_all_blocks_unannotated_experiments(
    tmp_path: Path, unannotated_corpus_env
) -> None:
    """When human annotations are absent, Exp 1, 2, 3, 5 are BLOCKED while Exp 4 succeeds."""
    result = run_all_experiments(
        cfg=unannotated_corpus_env["cfg"],
        output_dir=tmp_path,
        smoke_test=False,
    )

    summary = result["summary"]
    assert summary["experiment1"]["status"] == "BLOCKED"
    assert summary["experiment2"]["status"] == "BLOCKED"
    assert summary["experiment3"]["status"] == "BLOCKED"
    assert summary["experiment4"]["status"] == "SUCCESS"
    assert summary["experiment5"]["status"] == "BLOCKED"

    for blocked_exp in ("experiment1", "experiment2", "experiment3", "experiment5"):
        assert TODO_HUMAN_ANNOTATION in summary[blocked_exp]["detail"]

    # Manifest and report written
    manifest_file = tmp_path / "summary_manifest.json"
    report_file = tmp_path / "experiments_summary.md"

    assert manifest_file.is_file()
    assert report_file.is_file()

    manifest_data = read_json(manifest_file)
    assert manifest_data["suite_name"] == "Phase 6 Experiments"
    assert "provenance" in manifest_data


def test_run_all_production_gold_summary() -> None:
    """Production manifest must record SUCCESS for all five experiments on gold data."""
    manifest_file = Path("results/experiments/summary_manifest.json")
    if manifest_file.is_file():
        manifest_data = read_json(manifest_file)
        summary = manifest_data.get("summary", {})
        for exp in ("experiment1", "experiment2", "experiment3", "experiment4", "experiment5"):
            assert exp in summary
            assert summary[exp]["status"] == "SUCCESS"
        assert manifest_data.get("provenance", {}).get("research_valid") is True


def test_run_all_smoke_mode(tmp_path: Path) -> None:
    """Smoke mode executes all 5 experiments without blocking."""
    result = run_all_experiments(output_dir=tmp_path, smoke_test=True)

    summary = result["summary"]
    for i in range(1, 6):
        exp_key = f"experiment{i}"
        assert exp_key in summary
        assert summary[exp_key]["status"] == "SUCCESS_SMOKE"


def test_run_all_selective_execution(tmp_path: Path) -> None:
    """Selective execution must run only specified experiments."""
    result = run_all_experiments(
        output_dir=tmp_path,
        selected_experiments=["experiment4"],
        smoke_test=True,
    )

    summary = result["summary"]
    assert list(summary.keys()) == ["experiment4"]
    assert summary["experiment4"]["status"] == "SUCCESS_SMOKE"


def test_runner_cli(tmp_path: Path) -> None:
    """CLI execution in smoke mode must return 0 and write manifest."""
    exit_code = main(["--output-dir", str(tmp_path), "--smoke-test"])
    assert exit_code == 0
    assert (tmp_path / "summary_manifest.json").is_file()
