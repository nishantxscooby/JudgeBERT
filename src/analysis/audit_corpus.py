"""Independent, provenance-stamped audit of the rebuilt master corpus.

The audit is intentionally descriptive: it records duplicate and leakage
signals without turning absent human annotations into a data failure.  The
master is loaded through :func:`src.data.load_data.load_master` so schema
coercion and derived columns match the rest of the project.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import schema
from src.data.load_data import load_master
from src.utils.config import load_config, resolve_path
from src.utils.io import provenance, write_json

_NONEMPTY_AUDIT_COLUMNS = [
    "pair_id",
    "language",
    "legal_domain",
    "original_text",
    "simplified_text",
    "source",
    "split",
    "source_group",
    "verbatim_quote",
]


def _populated(series: pd.Series) -> pd.Series:
    """Return a mask for values that are neither null nor blank text."""
    return series.notna() & (series.astype(str).str.strip() != "")


def audit_dataframe(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the complete JSON-serializable corpus audit."""
    total = int(len(df))
    missing_columns = [column for column in _NONEMPTY_AUDIT_COLUMNS if column not in df]
    missing_fields: list[tuple[int, str, str]] = []
    for column in _NONEMPTY_AUDIT_COLUMNS:
        if column not in df:
            continue
        blank = ~_populated(df[column])
        for index in df.index[blank]:
            pair_id = str(df.at[index, "pair_id"]) if "pair_id" in df else ""
            missing_fields.append((int(index) + 1, column, pair_id))

    language_counts = Counter(df["language"].astype(str)) if "language" in df else Counter()
    domain_counts = Counter(df["legal_domain"].astype(str)) if "legal_domain" in df else Counter()

    split_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    if {"language", "split", "legal_domain"}.issubset(df.columns):
        for row in df.itertuples(index=False):
            key = f"{row.language}_{row.split}"
            split_counts[key][str(row.legal_domain)] += 1

    duplicate_pair_ids: list[str] = []
    if "pair_id" in df:
        duplicate_pair_ids = sorted(
            str(pair_id)
            for pair_id, count in Counter(df["pair_id"].astype(str)).items()
            if count > 1
        )

    def duplicate_values(column: str) -> list[str]:
        if column not in df:
            return []
        return [
            str(value)
            for value, count in Counter(df[column].astype(str)).items()
            if count > 1
        ][:5]

    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    if {"source_group", "split"}.issubset(df.columns):
        for row in df.itertuples(index=False):
            group_splits[str(row.source_group)].add(str(row.split))
    groups_spanning = {
        group: sorted(splits)
        for group, splits in sorted(group_splits.items())
        if len(splits) > 1
    }

    verbatim_mismatch: list[str] = []
    if {"pair_id", "language", "verbatim_quote"}.issubset(df.columns):
        for row in df.itertuples(index=False):
            quote = str(row.verbatim_quote).lower()
            if (row.language == "en" and quote != "true") or (
                row.language == "hi" and quote != "false"
            ):
                verbatim_mismatch.append(str(row.pair_id))

    identical_simpl: list[str] = []
    if {"pair_id", "original_text", "simplified_text"}.issubset(df.columns):
        identical = (
            df["original_text"].astype(str).str.strip()
            == df["simplified_text"].astype(str).str.strip()
        )
        identical_simpl = df.loc[identical, "pair_id"].astype(str).tolist()

    status_counts = (
        {
            str(status): int(count)
            for status, count in df["annotation_status"].astype(str).value_counts().items()
        }
        if "annotation_status" in df
        else {}
    )
    gold_rows = int(schema.is_gold(df).sum())

    return {
        "provenance": provenance(cfg, audit="final_corpus"),
        "total_rows": total,
        "unique_pair_ids": int(df["pair_id"].nunique()) if "pair_id" in df else 0,
        "missing_required_columns": missing_columns,
        "missing_fields": missing_fields,
        "language_counts": {str(k): int(v) for k, v in sorted(language_counts.items())},
        "domain_counts": {str(k): int(v) for k, v in sorted(domain_counts.items())},
        "split_totals": (
            {str(k): int(v) for k, v in df["split"].value_counts().items()}
            if "split" in df
            else {}
        ),
        "split_counts": {
            key: {str(k): int(v) for k, v in sorted(counts.items())}
            for key, counts in sorted(split_counts.items())
        },
        "duplicate_pair_ids": duplicate_pair_ids,
        "duplicate_original_texts": duplicate_values("original_text"),
        "duplicate_simplified_texts": duplicate_values("simplified_text"),
        "source_groups_spanning_splits": groups_spanning,
        "leakage_free": not groups_spanning,
        "verbatim_quote_mismatches": verbatim_mismatch,
        "identical_original_simplified": identical_simpl,
        "human_annotation": {
            "gold_rows": gold_rows,
            "human_lmp_score_populated": (
                int(df["human_lmp_score"].notna().sum())
                if "human_lmp_score" in df
                else 0
            ),
            "error_type_populated": (
                int(_populated(df["error_type"]).sum()) if "error_type" in df else 0
            ),
            "decision_populated": (
                int(_populated(df["decision"]).sum()) if "decision" in df else 0
            ),
            "annotation_status_counts": status_counts,
            "draft_rows_are_gold": int(
                (
                    (df["annotation_status"] == "draft")
                    & schema.is_gold(df)
                ).sum()
            )
            if "annotation_status" in df
            else 0,
        },
    }


def report(
    cfg: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load, audit, and write the master corpus audit result."""
    cfg = cfg or load_config()
    df = load_master(cfg=cfg, subset="all")
    report_data = audit_dataframe(df, cfg)
    path = (
        Path(output_path)
        if output_path is not None
        else resolve_path(cfg, "reports_dir", create=True) / "final_corpus_audit.json"
    )
    write_json(report_data, path)
    print("Audit completed. JSON saved to", path)
    return report_data


def main() -> int:
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
