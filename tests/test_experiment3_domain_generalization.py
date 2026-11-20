"""Tests for Experiment 3 (Held-Out Domain Generalisation)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.common import ExperimentBlockedError
from src.experiments.experiment3_domain_generalization import (
    main,
    run_experiment3,
    validate_domain_configuration,
)
from src.utils.io import TODO_HUMAN_ANNOTATION, read_json


def test_domain_exclusion_assertion() -> None:
    """Held-out domain appearing in training domains must be rejected immediately."""
    with pytest.raises(ValueError, match="Domain contamination"):
        validate_domain_configuration(
            train_domains=["contract", "cyber"],
            held_out_domain="cyber",
            configured_domains=["contract", "cyber", "sale_of_goods"],
        )


def test_invalid_held_out_domain_rejected() -> None:
    """Empty or unconfigured held-out domain must raise ValueError."""
    with pytest.raises(ValueError, match="non-empty string"):
        validate_domain_configuration(
            train_domains=["contract"],
            held_out_domain="",
            configured_domains=["contract", "cyber"],
        )

    with pytest.raises(ValueError, match="absent from configured"):
        validate_domain_configuration(
            train_domains=["contract"],
            held_out_domain="unknown_domain",
            configured_domains=["contract", "cyber"],
        )


def test_unknown_train_domain_rejected() -> None:
    """Training domains not present in dataset configuration must be rejected."""
    with pytest.raises(ValueError, match="Unknown training domain"):
        validate_domain_configuration(
            train_domains=["contract", "space_law"],
            held_out_domain="cyber",
            configured_domains=["contract", "cyber", "sale_of_goods"],
        )


def test_exp3_blocked_on_unannotated_dataset(
    tmp_path: Path, unannotated_corpus_env
) -> None:
    """Final mode execution must fail loudly with TODO_HUMAN_ANNOTATION when gold is absent."""
    with pytest.raises(ExperimentBlockedError, match=f"Blocked: {TODO_HUMAN_ANNOTATION}"):
        run_experiment3(
            cfg=unannotated_corpus_env["cfg"],
            output_dir=tmp_path,
            smoke_test=False,
        )

    blocked_json = tmp_path / "blocked.json"
    assert blocked_json.is_file()
    payload = read_json(blocked_json)
    assert payload["status"] == "BLOCKED"
    assert payload["code"] == TODO_HUMAN_ANNOTATION


def test_exp3_smoke_mode(tmp_path: Path) -> None:
    """Smoke mode must evaluate held-out domain and write all output artifacts."""
    result = run_experiment3(output_dir=tmp_path, smoke_test=True)

    assert result["held_out_domain"] == "cyber"
    assert set(result["train_domains"]) == {"contract", "sale_of_goods"}
    assert result["n_samples"] > 0

    metrics_file = tmp_path / "domain_generalization_metrics.json"
    preds_file = tmp_path / "domain_generalization_predictions.csv"
    report_file = tmp_path / "domain_generalization_report.md"
    prov_file = tmp_path / "provenance.json"

    assert metrics_file.is_file()
    assert preds_file.is_file()
    assert report_file.is_file()
    assert prov_file.is_file()

    data = read_json(metrics_file)
    assert "metrics" in data
    assert "tfidf" in data["metrics"]
    assert "sbert" in data["metrics"]
    assert "judgebert" in data["metrics"]


def test_exp3_cli_blocked_exit_code(
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


def test_exp3_cli_smoke_exit_code(tmp_path: Path) -> None:
    """CLI with --smoke-test flag must return 0."""
    exit_code = main(["--output-dir", str(tmp_path), "--smoke-test"])
    assert exit_code == 0


def test_exp3_production_domain_anti_leakage() -> None:
    """Production corpus must satisfy Experiment 3 domain split and 0-overlap constraints."""
    from src.data.load_data import assert_gold, load_master

    gold = load_master(subset="gold")
    assert len(gold) == 340
    assert_gold(gold)

    domain_counts = gold["legal_domain"].value_counts().to_dict()
    assert domain_counts.get("contract") == 202
    assert domain_counts.get("sale_of_goods") == 66
    assert domain_counts.get("cyber") == 72

    train_domains = gold[gold["legal_domain"].isin(["contract", "sale_of_goods"])]
    cyber_domain = gold[gold["legal_domain"] == "cyber"]

    assert len(train_domains) == 268
    assert len(cyber_domain) == 72

    train_groups = set(train_domains["source_group"].dropna().unique())
    cyber_groups = set(cyber_domain["source_group"].dropna().unique())

    overlap = train_groups & cyber_groups
    assert len(overlap) == 0
