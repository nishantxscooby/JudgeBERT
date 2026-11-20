"""Dataset validation and data-quality reporting.

Runs are expected to fail loudly rather than proceed on a malformed corpus:
a silently mis-parsed score column poisons every downstream metric.

Usage
-----
Command line (exits non-zero if any ERROR is found)::

    python -m src.data.validate_data
    python -m src.data.validate_data --data data/processed/lmp_pairs.csv
    python -m src.data.validate_data --strict --write-report

As a library::

    from src.data.validate_data import validate_dataframe
    report = validate_dataframe(df)
    report.raise_if_errors()
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import schema
from src.data.preprocess import token_jaccard
from src.utils.config import load_config, resolve_path
from src.utils.io import (
    dataframe_to_markdown,
    provenance,
    read_csv,
    write_json,
)
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)

# Issues at or above this severity make the corpus unusable.
SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"
SEVERITY_INFO = "INFO"

# These warnings describe the expected pre-annotation state rather than a
# structural or data-quality defect.  Strict Phase 1 validation must still pass
# while they are the only findings; every other warning remains blocking.
STRICT_NON_BLOCKING_WARNING_CODES: frozenset[str] = frozenset(
    {"gold_coverage", "no_human_annotations", "annotation_incomplete"}
)


@dataclass
class Issue:
    """A single validation finding."""

    severity: str
    code: str
    message: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        suffix = f" (n={self.count})" if self.count else ""
        example = f" e.g. {self.examples[0]}" if self.examples else ""
        return f"[{self.severity}] {self.code}: {self.message}{suffix}{example}"


@dataclass
class ValidationReport:
    """Aggregated validation result plus descriptive statistics."""

    n_rows: int = 0
    issues: list[Issue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == SEVERITY_WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, message: str, count: int = 0,
            examples: list[str] | None = None) -> None:
        self.issues.append(
            Issue(severity=severity, code=code, message=message,
                  count=count, examples=examples or [])
        )

    def raise_if_errors(self) -> None:
        """Raise :class:`DataValidationError` if any ERROR-level issue exists."""
        if self.errors:
            detail = "\n  ".join(str(e) for e in self.errors)
            raise DataValidationError(
                f"Dataset validation failed with {len(self.errors)} error(s):\n  {detail}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "ok": self.ok,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "count": i.count,
                    "examples": i.examples[:5],
                }
                for i in self.issues
            ],
            "stats": self.stats,
        }

    def summary_lines(self) -> list[str]:
        lines = [f"Rows: {self.n_rows}"]
        lines.append(f"Errors: {len(self.errors)}  Warnings: {len(self.warnings)}")
        for issue in self.issues:
            lines.append("  " + str(issue))
        return lines


class DataValidationError(ValueError):
    """Raised when the corpus violates a hard schema requirement."""


def strict_blocking_warnings(report: ValidationReport) -> list[Issue]:
    """Return warnings that should fail ``--strict`` validation.

    Missing human annotations are an explicit downstream experiment blocker,
    but they do not make the unannotated corpus structurally invalid.  Strict
    mode therefore permits only the two warnings that document that expected
    state and continues to reject every other warning.
    """
    return [
        issue
        for issue in report.warnings
        if issue.code not in STRICT_NON_BLOCKING_WARNING_CODES
    ]


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------
def validate_dataframe(
    df: pd.DataFrame,
    cfg: dict | None = None,
    check_leakage: bool = True,
) -> ValidationReport:
    """Validate a master LMP DataFrame against the schema and the config.

    Parameters
    ----------
    df
        The corpus.  Column names are matched exactly.
    cfg
        Project config.  Loaded from disk when omitted.
    check_leakage
        Run the cross-split near-duplicate check.  Quadratic in the number of
        rows, so it can be disabled for very large corpora.

    Returns
    -------
    ValidationReport
        Never raises for data problems; inspect :attr:`~ValidationReport.ok`
        or call :meth:`~ValidationReport.raise_if_errors`.  Raises
        :class:`DataValidationError` only for a structurally impossible input
        (e.g. an empty frame with no columns at all).
    """
    cfg = cfg or load_config()
    report = ValidationReport(n_rows=len(df))

    if df is None or df.shape[1] == 0:
        raise DataValidationError("DataFrame has no columns; nothing to validate.")

    dataset_cfg = cfg.get("dataset", {})
    score_min = float(dataset_cfg.get("score_min", schema.SCORE_MIN))
    score_max = float(dataset_cfg.get("score_max", schema.SCORE_MAX))
    gold_statuses = set(dataset_cfg.get("gold_status_values", schema.GOLD_STATUSES))

    # -- 1. required columns ------------------------------------------------
    missing = [c for c in schema.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report.add(
            SEVERITY_ERROR, "missing_required_columns",
            "Required columns absent from the dataset", len(missing), missing,
        )
        # Everything below assumes the required columns exist.
        return report

    unknown = [c for c in df.columns if c not in schema.ALL_COLUMNS]
    if unknown:
        report.add(
            SEVERITY_WARNING, "unknown_columns",
            "Columns not declared in the schema are ignored by the pipeline",
            len(unknown), unknown,
        )

    # -- 2. row count -------------------------------------------------------
    if len(df) == 0:
        report.add(SEVERITY_ERROR, "empty_dataset", "Dataset contains zero rows")
        return report

    # -- 3. empty text and missing required values --------------------------
    for column in ("original_text", "simplified_text", "source"):
        blank = df[column].isna() | (df[column].astype(str).str.strip() == "")
        if blank.any():
            report.add(
                SEVERITY_ERROR, f"blank_{column}",
                f"'{column}' is empty or null",
                int(blank.sum()), df.loc[blank, "pair_id"].astype(str).head(5).tolist(),
            )

    # -- 4. pair_id uniqueness ---------------------------------------------
    duplicated_ids = df["pair_id"].duplicated(keep=False)
    if duplicated_ids.any():
        dupes = sorted(df.loc[duplicated_ids, "pair_id"].astype(str).unique())
        report.add(
            SEVERITY_ERROR, "duplicate_pair_id",
            "'pair_id' must be unique", len(dupes), dupes[:5],
        )

    # -- 5. categorical vocabularies ---------------------------------------
    vocabulary_checks = {
        "language": schema.LANGUAGES.keys(),
        "legal_domain": schema.DOMAINS.keys(),
        "decision": schema.DECISIONS,
        "error_type": schema.ERROR_LABELS,
        "split": schema.SPLITS,
        "annotation_status": schema.ANNOTATION_STATUSES,
    }
    for column, allowed in vocabulary_checks.items():
        if column not in df.columns:
            continue
        values = df[column].dropna()
        if column == "decision":
            # An unannotated row legitimately has no decision yet.
            values = values[values.astype(str).str.strip() != ""]
        bad = schema.validate_vocabulary(values, allowed)
        if bad:
            report.add(
                SEVERITY_ERROR, f"invalid_{column}",
                f"'{column}' contains values outside the controlled vocabulary "
                f"{sorted(allowed)}",
                len(bad), bad[:5],
            )

    # -- 6. score validity --------------------------------------------------
    score = pd.to_numeric(df["human_lmp_score"], errors="coerce")
    non_numeric = df["human_lmp_score"].notna() & score.isna()
    if non_numeric.any():
        report.add(
            SEVERITY_ERROR, "non_numeric_score",
            "human_lmp_score contains non-numeric values",
            int(non_numeric.sum()),
            df.loc[non_numeric, "pair_id"].astype(str).head(5).tolist(),
        )

    out_of_range = score.notna() & ((score < score_min) | (score > score_max))
    if out_of_range.any():
        report.add(
            SEVERITY_ERROR, "score_out_of_range",
            f"human_lmp_score outside [{score_min:g}, {score_max:g}]",
            int(out_of_range.sum()),
            df.loc[out_of_range, "pair_id"].astype(str).head(5).tolist(),
        )

    fractional = score.notna() & (score % 1 != 0)
    if fractional.any():
        report.add(
            SEVERITY_INFO, "fractional_scores",
            "human_lmp_score contains fractional values; the guidelines use "
            "integers, so confirm these are intentional",
            int(fractional.sum()),
        )

    # -- 7. annotation status vs. score consistency -------------------------
    status = df.get("annotation_status", pd.Series("unannotated", index=df.index))
    status = status.fillna("unannotated").astype(str)

    gold_mask = status.isin(gold_statuses)
    gold_missing_score = gold_mask & score.isna()
    if gold_missing_score.any():
        report.add(
            SEVERITY_ERROR, "gold_without_score",
            "Rows marked as human-annotated have no human_lmp_score",
            int(gold_missing_score.sum()),
            df.loc[gold_missing_score, "pair_id"].astype(str).head(5).tolist(),
        )

    non_gold_with_score = (~gold_mask) & score.notna()
    if non_gold_with_score.any():
        report.add(
            SEVERITY_WARNING, "non_gold_with_score",
            "Rows whose annotation_status is not a human status carry a "
            "human_lmp_score. This is the failure mode that would let a "
            "machine-drafted score masquerade as ground truth — confirm these "
            "rows were genuinely annotated by a human.",
            int(non_gold_with_score.sum()),
            df.loc[non_gold_with_score, "pair_id"].astype(str).head(5).tolist(),
        )

    n_gold = int(gold_mask.sum())
    report.add(
        SEVERITY_INFO if n_gold else SEVERITY_WARNING,
        "gold_coverage",
        "Rows usable as ground truth (human-annotated with a score)",
        n_gold,
    )

    # -- 8. exact duplicate pairs ------------------------------------------
    pair_keys = (
        df["original_text"].astype(str).str.strip().str.lower()
        + " ||| "
        + df["simplified_text"].astype(str).str.strip().str.lower()
    )
    exact_dupes = pair_keys.duplicated(keep=False)
    if exact_dupes.any():
        report.add(
            SEVERITY_ERROR, "exact_duplicate_pairs",
            "Identical (original, simplified) pairs appear more than once; "
            "this leaks between splits and inflates test scores",
            int(exact_dupes.sum()),
            df.loc[exact_dupes, "pair_id"].astype(str).head(5).tolist(),
        )

    # Same simplified text reused against different originals — a subtler
    # form of duplication that still inflates apparent performance.
    reused = df["simplified_text"].astype(str).str.strip().str.lower().duplicated(keep=False)
    if reused.any():
        report.add(
            SEVERITY_WARNING, "reused_simplified_text",
            "The same simplified sentence is paired with more than one original",
            int(reused.sum()),
            df.loc[reused, "pair_id"].astype(str).head(5).tolist(),
        )

    # -- 9. statistics ------------------------------------------------------
    report.stats = _compute_stats(df, score, status, gold_statuses)

    # -- 10. distributions and imbalance -----------------------------------
    _check_distribution(report, df, "language", min_per_group=2)
    _check_distribution(report, df, "legal_domain", min_per_group=2)
    _check_split_balance(report, df)

    # -- 11. cross-split leakage -------------------------------------------
    if check_leakage:
        _check_leakage(report, df, cfg)

    # -- 12. brief compliance ------------------------------------------------
    _check_brief_requirements(report, df, report.stats)

    return report


def _compute_stats(
    df: pd.DataFrame, score: pd.Series, status: pd.Series, gold_statuses: set[str]
) -> dict[str, Any]:
    """Descriptive statistics embedded in the report and the dashboard."""
    gold_mask = status.isin(gold_statuses) & score.notna()
    stats: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_gold": int(gold_mask.sum()),
        "n_unannotated": int((~gold_mask).sum()),
        "gold_fraction": round(float(gold_mask.mean()), 4) if len(df) else 0.0,
        "language_counts": _value_counts(df, "language"),
        "domain_counts": _value_counts(df, "legal_domain"),
        "split_counts": _value_counts(df, "split"),
        "annotation_status_counts": _value_counts(status.to_frame("s"), "s"),
    }
    if gold_mask.any():
        gold_scores = score[gold_mask]
        stats["score_distribution"] = {
            "count": int(gold_scores.count()),
            "mean": round(float(gold_scores.mean()), 3),
            "std": round(float(gold_scores.std(ddof=0)), 3),
            "min": float(gold_scores.min()),
            "max": float(gold_scores.max()),
            "histogram": {
                str(int(k)): int(v)
                for k, v in gold_scores.round().astype(int).value_counts().sort_index().items()
            },
            "bucket_counts": {
                label: int(sum(schema.score_bucket(v) == label for v in gold_scores))
                for label in schema.SCORE_BUCKET_LABELS
            },
        }
        stats["error_type_counts"] = _value_counts(df.loc[gold_mask], "error_type")
    else:
        stats["score_distribution"] = None
        stats["error_type_counts"] = {}

    # Cross-tab of language x domain — the sampling grid of the corpus.
    if {"language", "legal_domain"}.issubset(df.columns):
        grid = pd.crosstab(df["language"], df["legal_domain"])
        stats["language_domain_grid"] = {
            str(idx): {str(c): int(v) for c, v in row.items()}
            for idx, row in grid.iterrows()
        }
    return stats


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {
        str(k): int(v)
        for k, v in df[column].fillna("<missing>").astype(str).value_counts().items()
    }


def _check_distribution(
    report: ValidationReport, df: pd.DataFrame, column: str, min_per_group: int = 2
) -> None:
    """Flag empty or severely under-populated categorical groups."""
    if column not in df.columns:
        return
    counts = df[column].fillna("<missing>").astype(str).value_counts()
    sparse = counts[counts < min_per_group]
    if len(sparse):
        report.add(
            SEVERITY_WARNING, f"sparse_{column}",
            f"'{column}' has groups with fewer than {min_per_group} rows; "
            f"per-group metrics for these will be unreliable",
            int(len(sparse)),
            [f"{k} (n={v})" for k, v in sparse.items()][:5],
        )


def _check_split_balance(report: ValidationReport, df: pd.DataFrame) -> None:
    """Warn when a split is empty or when languages are badly unbalanced."""
    if "split" not in df.columns:
        return
    counts = df["split"].astype(str).value_counts()
    absent = [s for s in schema.SPLITS if s not in counts.index]
    if absent:
        report.add(
            SEVERITY_WARNING, "empty_split",
            "No rows assigned to split(s)",
            len(absent), absent,
        )

    # Every split should contain every language, otherwise Experiment 2
    # cannot be run on that split.
    if "language" in df.columns and not absent:
        grid = pd.crosstab(df["split"], df["language"])
        for split in schema.SPLITS:
            if split not in grid.index:
                continue
            empty_langs = [lang for lang in grid.columns if grid.loc[split, lang] == 0]
            if empty_langs:
                report.add(
                    SEVERITY_WARNING, f"missing_language_in_{split}",
                    f"Split '{split}' has no rows for language(s) "
                    f"{empty_langs}; Experiment 2 cannot report per-language "
                    f"metrics for it",
                    len(empty_langs), empty_langs,
                )


def _check_leakage(report: ValidationReport, df: pd.DataFrame, cfg: dict) -> None:
    """Detect near-duplicate originals straddling two different splits.

    Two mechanisms are checked:
      1. ``source_group`` — rows sharing a group must share a split.
      2. Token-Jaccard on ``original_text`` — catches paraphrases of the same
         provision that were not given a shared group key.
    """
    threshold = float(cfg.get("splits", {}).get("leakage_similarity_threshold", 0.85))

    if "source_group" in df.columns and "split" in df.columns:
        groups = df.dropna(subset=["source_group"])
        groups = groups[groups["source_group"].astype(str).str.strip() != ""]
        if len(groups):
            span = groups.groupby("source_group")["split"].nunique()
            leaking = span[span > 1]
            if len(leaking):
                report.add(
                    SEVERITY_ERROR, "source_group_split_leak",
                    "Rows sharing a source_group are assigned to different "
                    "splits; sibling reformulations of one provision must stay "
                    "together",
                    int(len(leaking)), [str(k) for k in leaking.index[:5]],
                )

    if "split" not in df.columns or "language" not in df.columns:
        return

    # O(n^2) within language — fine for a few thousand rows, and the corpus is
    # specified at ~300-1000.
    by_language: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for row in df.itertuples(index=False):
        language = str(getattr(row, "language"))
        split = str(getattr(row, "split"))
        pair_id = str(getattr(row, "pair_id"))
        text = str(getattr(row, "original_text"))
        by_language.setdefault((language, split), []).append((pair_id, text, split))

    cross_split_dupes: list[str] = []
    splits = sorted({s for (_, s) in by_language})
    for language in {lang for (lang, _) in by_language}:
        for i, split_a in enumerate(splits):
            for split_b in splits[i + 1 :]:
                group_a = by_language.get((language, split_a), [])
                group_b = by_language.get((language, split_b), [])
                for id_a, text_a, _ in group_a:
                    for id_b, text_b, _ in group_b:
                        if token_jaccard(text_a, text_b, language) >= threshold:
                            cross_split_dupes.append(f"{id_a}~{id_b}")

    if cross_split_dupes:
        report.add(
            SEVERITY_ERROR, "cross_split_near_duplicate",
            f"Originals in different splits have token Jaccard >= {threshold}; "
            f"this is data leakage",
            len(cross_split_dupes), cross_split_dupes[:5],
        )


def _check_brief_requirements(
    report: ValidationReport, df: pd.DataFrame, stats: dict[str, Any]
) -> None:
    """Check the explicit numeric requirements from the project brief."""
    n_rows = len(df)
    if n_rows < 300:
        report.add(
            SEVERITY_WARNING, "below_brief_pair_count",
            "The brief requires at least 300 original-simplified pairs",
            n_rows,
        )

    if "language" in df.columns:
        counts = df["language"].astype(str).value_counts()
        for code in ("en", "hi"):
            if code not in counts.index:
                report.add(
                    SEVERITY_ERROR, f"missing_language_{code}",
                    f"No rows for language '{code}'; the brief requires "
                    f"English plus one Indian language",
                )
            elif counts[code] < 150:
                report.add(
                    SEVERITY_WARNING, f"language_{code}_low",
                    f"Language '{code}' has {counts[code]} rows; the brief "
                    f"targets 150+ per language",
                    int(counts[code]),
                )

    n_gold = stats.get("n_gold", 0)
    if n_gold == 0:
        report.add(
            SEVERITY_WARNING, "no_human_annotations",
            "No human LMP annotations exist yet. The pipeline can be exercised "
            "end-to-end on draft labels, but NO experiment result may be "
            "reported as final until human annotation is complete. "
            "See src/annotation/annotation_app.py.",
        )
    elif n_gold < 300:
        report.add(
            SEVERITY_WARNING, "annotation_incomplete",
            f"Only {n_gold} human-annotated rows. Results computed on this "
            f"subset are preliminary.",
            int(n_gold),
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_markdown(report: ValidationReport, cfg: dict) -> str:
    """Render the validation report as a Markdown document."""
    lines: list[str] = []
    lines.append("# Data Quality Report\n")
    lines.append(
        f"_Generated {provenance(cfg)['generated_at_utc']} "
        f"· config `{provenance(cfg)['config_fingerprint']}`_\n"
    )

    status = "PASS" if report.ok else "FAIL"
    lines.append(f"## Status: **{status}**\n")
    lines.append(f"- Rows: **{report.n_rows}**")
    lines.append(f"- Errors: **{len(report.errors)}**")
    lines.append(f"- Warnings: **{len(report.warnings)}**\n")

    if report.issues:
        lines.append("## Findings\n")
        issue_df = pd.DataFrame(
            [
                {
                    "Severity": i.severity,
                    "Code": i.code,
                    "Count": i.count,
                    "Message": i.message,
                    "Examples": ", ".join(str(e) for e in i.examples[:3]),
                }
                for i in report.issues
            ]
        )
        lines.append(dataframe_to_markdown(issue_df))
        lines.append("")

    stats = report.stats
    lines.append("## Distributions\n")

    for key, title in (
        ("language_counts", "Language"),
        ("domain_counts", "Legal domain"),
        ("split_counts", "Split"),
        ("annotation_status_counts", "Annotation status"),
        ("error_type_counts", "Error type (annotated rows only)"),
    ):
        counts = stats.get(key) or {}
        if not counts:
            continue
        lines.append(f"### {title}\n")
        table = pd.DataFrame(
            sorted(counts.items(), key=lambda kv: -kv[1]), columns=[title, "Rows"]
        )
        table["Share"] = (table["Rows"] / max(report.n_rows, 1)).map("{:.1%}".format)
        lines.append(dataframe_to_markdown(table))
        lines.append("")

    grid = stats.get("language_domain_grid")
    if grid:
        lines.append("### Language x domain grid\n")
        grid_df = pd.DataFrame(grid).T
        grid_df.index.name = "language"
        lines.append(dataframe_to_markdown(grid_df.reset_index()))
        lines.append("")

    dist = stats.get("score_distribution")
    lines.append("### Human LMP score distribution\n")
    if dist:
        lines.append(
            f"- n = {dist['count']}, mean = {dist['mean']}, "
            f"sd = {dist['std']}, range = [{dist['min']:g}, {dist['max']:g}]"
        )
        hist = pd.DataFrame(
            sorted(dist["histogram"].items(), key=lambda kv: int(kv[0])),
            columns=["Score", "Count"],
        )
        lines.append("")
        lines.append(dataframe_to_markdown(hist))
        buckets = pd.DataFrame(
            dist["bucket_counts"].items(), columns=["Band", "Count"]
        )
        lines.append("")
        lines.append(dataframe_to_markdown(buckets))
    else:
        lines.append(
            "**No human annotations yet.** Every score is either absent or a "
            "non-gold draft. No experiment may be reported as final. "
            "`TODO_HUMAN_ANNOTATION`\n"
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the LMP corpus and emit a data-quality report."
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Path to the master CSV (default: config dataset.master_file)")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat data-quality warnings as errors; expected missing-human-"
            "annotation warnings remain non-blocking"
        ),
    )
    parser.add_argument("--write-report", action="store_true",
                        help="Write results/data_validation.json and reports/data_quality.md")
    parser.add_argument("--no-leakage-check", action="store_true",
                        help="Skip the O(n^2) cross-split duplicate scan")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging("INFO")

    if args.data:
        data_path = Path(args.data)
    else:
        data_path = resolve_path(cfg, "processed_dir") / Path(
            cfg["dataset"]["master_file"]
        ).name

    if not data_path.exists():
        logger.error(
            "Master dataset not found: %s\n"
            "Build it first with:  python -m src.data.build_dataset",
            data_path,
        )
        return 2

    df = read_csv(data_path)
    report = validate_dataframe(df, cfg, check_leakage=not args.no_leakage_check)

    print()
    for line in report.summary_lines():
        print(line)
    print()

    if args.write_report:
        results_dir = resolve_path(cfg, "results_dir", create=True)
        reports_dir = resolve_path(cfg, "reports_dir", create=True)
        write_json(
            {"provenance": provenance(cfg), **report.to_dict()},
            results_dir / "data_validation.json",
        )
        markdown = render_markdown(report, cfg)
        (reports_dir / "data_quality.md").write_text(markdown, encoding="utf-8")
        logger.info("Wrote %s", reports_dir / "data_quality.md")

    if not report.ok:
        logger.error("Validation FAILED with %d error(s).", len(report.errors))
        return 1
    if args.strict:
        blocking_warnings = strict_blocking_warnings(report)
        if blocking_warnings:
            logger.error(
                "Strict mode: %d data-quality warning(s) treated as failure: %s",
                len(blocking_warnings),
                [issue.code for issue in blocking_warnings],
            )
            return 1
        if report.warnings:
            logger.info(
                "Strict mode passed with %d expected pre-annotation warning(s): %s",
                len(report.warnings),
                [issue.code for issue in report.warnings],
            )

    logger.info("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
