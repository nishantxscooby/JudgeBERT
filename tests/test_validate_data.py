"""Tests for Phase 1 validation command behavior."""

from __future__ import annotations

from src.data import validate_data


def _warning(code: str) -> validate_data.Issue:
    return validate_data.Issue(
        severity=validate_data.SEVERITY_WARNING,
        code=code,
        message=code,
    )


def test_strict_mode_allows_only_expected_unannotated_warnings():
    report = validate_data.ValidationReport(
        n_rows=340,
        issues=[
            _warning("gold_coverage"),
            _warning("no_human_annotations"),
            _warning("annotation_incomplete"),
        ],
    )

    assert validate_data.strict_blocking_warnings(report) == []


def test_strict_mode_still_blocks_structural_or_quality_warnings():
    report = validate_data.ValidationReport(
        n_rows=340,
        issues=[_warning("gold_coverage"), _warning("unknown_columns")],
    )

    blocking = validate_data.strict_blocking_warnings(report)

    assert [issue.code for issue in blocking] == ["unknown_columns"]


def test_strict_cli_succeeds_when_human_annotation_is_the_only_blocker(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "master.csv"
    data_path.write_text("placeholder\nvalue\n", encoding="utf-8")
    report = validate_data.ValidationReport(
        n_rows=340,
        issues=[_warning("gold_coverage"), _warning("no_human_annotations")],
    )
    monkeypatch.setattr(validate_data, "validate_dataframe", lambda *args, **kwargs: report)

    exit_code = validate_data.main(["--data", str(data_path), "--strict"])

    assert exit_code == 0
