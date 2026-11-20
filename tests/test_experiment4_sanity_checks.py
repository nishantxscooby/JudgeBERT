"""Tests for Experiment 4 (Behavioral Sanity Checks)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.legal_cues import compare_texts
from src.experiments.experiment4_sanity_checks import (
    PROBE_PAIRS,
    evaluate_cue_consistency,
    main,
    run_experiment4,
)
from src.utils.io import read_json


def test_sanity_checks_can_run_without_human_labels(tmp_path: Path) -> None:
    """Sanity checks must run independently of human gold annotations."""
    result = run_experiment4(output_dir=tmp_path, smoke_test=False)

    assert result["n_probes"] > 0
    assert result["invariant_passed"] is True

    metrics_file = tmp_path / "sanity_metrics.json"
    pairs_file = tmp_path / "sanity_pairs.csv"
    report_file = tmp_path / "sanity_report.md"
    prov_file = tmp_path / "provenance.json"

    assert metrics_file.is_file()
    assert pairs_file.is_file()
    assert report_file.is_file()
    assert prov_file.is_file()


def test_probe_categories_represented() -> None:
    """Probes must cover all required minimal perturbation categories."""
    categories = {p.category for p in PROBE_PAIRS}
    required = {"identical", "unrelated", "modal", "negation", "number", "date"}
    assert required <= categories


def test_legal_cues_used_in_perturbations() -> None:
    """Verify that cue detector identifies intended shifts in controlled probes."""
    # Test identical: no cue changes
    identical_diag = evaluate_cue_consistency(
        "An agreement enforceable by law is a contract.",
        "An agreement enforceable by law is a contract.",
        lang="en",
    )
    assert identical_diag["has_changes"] is False

    # Test modal weakening
    modal_diag = evaluate_cue_consistency(
        "The seller shall deliver the goods within thirty days.",
        "The seller may deliver the goods within thirty days.",
        lang="en",
    )
    assert modal_diag["modal_shift"] != 0

    # Test negation flip
    negation_diag = evaluate_cue_consistency(
        "An agreement enforceable by law is a contract.",
        "An agreement enforceable by law is not a contract.",
        lang="en",
    )
    assert negation_diag["negation_flipped"] is True


def test_identical_scores_higher_than_unrelated(tmp_path: Path) -> None:
    """Identical pairs must receive higher simulated preservation scores than unrelated pairs."""
    result = run_experiment4(output_dir=tmp_path, smoke_test=True)

    categories = result["categories"]
    mean_identical = categories["identical"]["mean"]
    mean_unrelated = categories["unrelated"]["mean"]

    assert mean_identical > mean_unrelated
    assert mean_identical >= 9.0
    assert mean_unrelated <= 4.0


def test_exp4_cli(tmp_path: Path) -> None:
    """CLI execution must succeed and return 0."""
    exit_code = main(["--output-dir", str(tmp_path), "--smoke-test"])
    assert exit_code == 0
