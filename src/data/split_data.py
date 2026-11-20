"""Reproducible, leakage-aware train / validation / test splitting.

Two failure modes drive this design.

1. **Sibling leakage.**  The corpus contains several reformulations of the
   same source provision.  If one lands in train and another in test, the
   model is evaluated on text it has effectively already seen.  Rows are
   therefore grouped by ``source_group`` and whole groups are assigned to a
   single split.

2. **Distribution drift between splits.**  With ~300 rows, an unlucky random
   draw can put every low-scoring Hindi contract pair into train.  Groups are
   assigned by a greedy stratified algorithm that keeps the language x domain
   x score-band profile as close to the target fractions as possible.

Usage
-----
::

    python -m src.data.split_data --in data/processed/lmp_pairs.csv \\
                                  --out data/processed/lmp_pairs.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data import schema
from src.utils.config import load_config, resolve_path, set_global_seed
from src.utils.io import dataframe_to_markdown, provenance, read_csv, write_csv, write_json
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def effective_split_seed(cfg: dict, explicit_seed: int | None) -> int:
    """Resolve the seed actually used by splitting and provenance metadata."""
    if explicit_seed is not None:
        return int(explicit_seed)
    split_cfg = cfg.get("splits", {})
    return int(split_cfg.get("seed", cfg["project"]["seed"]))


# ---------------------------------------------------------------------------
# Group-level assignment
# ---------------------------------------------------------------------------
def _group_table(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Collapse the corpus to one row per ``source_group``.

    Each group's stratum is the *modal* combination of its members, so a group
    containing an English and a Hindi row still gets a single, deterministic
    assignment key.
    """
    records = []
    for group_id, members in df.groupby(group_column, sort=True):
        stratum_parts = []
        for column in ("language", "legal_domain", "score_bucket"):
            if column in members.columns and members[column].notna().any():
                stratum_parts.append(str(members[column].mode().iloc[0]))
            else:
                stratum_parts.append("unknown")
        records.append(
            {
                group_column: group_id,
                "n_members": len(members),
                "stratum": "|".join(stratum_parts),
            }
        )
    return pd.DataFrame(records)


def assign_splits(
    df: pd.DataFrame,
    cfg: dict | None = None,
    group_column: str = "source_group",
    seed: int | None = None,
) -> pd.DataFrame:
    """Assign every row to train / validation / test without splitting groups.

    The algorithm is a greedy largest-remainder allocator over strata:

    1. Collapse rows to groups; each group gets a stratum key.
    2. For each stratum, compute the target number of *groups* per split using
       the configured fractions, then hand out floor() quotas.
    3. Distribute the remaining slots by largest fractional remainder.
    4. Within a stratum, shuffle groups with the seeded RNG and deal them out.

    Returns a copy of ``df`` with the ``split`` column replaced.
    """
    cfg = cfg or load_config()
    split_cfg = cfg.get("splits", {})
    fractions = {
        "train": float(split_cfg.get("train", 0.70)),
        "validation": float(split_cfg.get("validation", 0.15)),
        "test": float(split_cfg.get("test", 0.15)),
    }
    seed = effective_split_seed(cfg, seed)

    if group_column not in df.columns:
        raise KeyError(
            f"Group column '{group_column}' missing. Run "
            f"src.data.load_data.add_derived_columns first."
        )

    import random

    rng = random.Random(seed)

    groups = _group_table(df, group_column)
    assignments: dict[str, str] = {}

    for stratum, block in groups.groupby("stratum", sort=True):
        group_ids = sorted(block[group_column].astype(str).tolist())
        rng.shuffle(group_ids)
        n = len(group_ids)

        if n == 1:
            # A lone group cannot be split; send it to train so that
            # validation/test stay as clean as possible.
            assignments[group_ids[0]] = "train"
            continue
        if n == 2:
            assignments[group_ids[0]] = "train"
            assignments[group_ids[1]] = "validation" if fractions["validation"] > 0 else "train"
            continue

        # Largest-remainder allocation of groups to splits.
        exact = {name: n * frac for name, frac in fractions.items()}
        quota = {name: int(value) for name, value in exact.items()}
        remainder = n - sum(quota.values())
        order = sorted(
            fractions, key=lambda name: (-(exact[name] - quota[name]), name)
        )
        for i in range(remainder):
            quota[order[i % len(order)]] += 1

        # Guarantee at least one group in validation and test when the
        # stratum is big enough, otherwise those splits end up empty for the
        # stratum and per-group metrics become impossible.
        for name in ("validation", "test"):
            if quota[name] == 0 and n >= 5 and fractions[name] > 0:
                donor = max(quota, key=lambda k: quota[k])
                if quota[donor] > 1:
                    quota[donor] -= 1
                    quota[name] = 1

        cursor = 0
        for name in ("train", "validation", "test"):
            for group_id in group_ids[cursor : cursor + quota[name]]:
                assignments[group_id] = name
            cursor += quota[name]

    out = df.copy()
    out["split"] = out[group_column].astype(str).map(assignments)
    if out["split"].isna().any():
        missing = out.loc[out["split"].isna(), group_column].astype(str).unique()[:5]
        raise RuntimeError(f"Unassigned groups after splitting: {list(missing)}")
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def split_diagnostics(df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Quantify how balanced and how leak-free the split is."""
    cfg = cfg or load_config()
    diag: dict = {
        "n_rows": int(len(df)),
        "split_sizes": {
            str(k): int(v) for k, v in df["split"].value_counts().items()
        },
    }

    for column in ("language", "legal_domain"):
        if column in df.columns:
            grid = pd.crosstab(df["split"], df[column])
            diag[f"{column}_by_split"] = {
                str(idx): {str(c): int(v) for c, v in row.items()}
                for idx, row in grid.iterrows()
            }

    if "score_bucket" in df.columns:
        grid = pd.crosstab(df["split"], df["score_bucket"])
        diag["score_bucket_by_split"] = {
            str(idx): {str(c): int(v) for c, v in row.items()}
            for idx, row in grid.iterrows()
        }

    # Group integrity: no group may span two splits.
    if "source_group" in df.columns:
        span = df.groupby("source_group")["split"].nunique()
        diag["groups_spanning_multiple_splits"] = int((span > 1).sum())
        diag["n_groups"] = int(len(span))

    return diag


def render_diagnostics(diag: dict) -> str:
    """Render split diagnostics as Markdown."""
    lines = ["## Split diagnostics\n"]
    lines.append(f"- Rows: **{diag['n_rows']}**")
    if "n_groups" in diag:
        lines.append(
            f"- Source groups: **{diag['n_groups']}** "
            f"(spanning multiple splits: **{diag['groups_spanning_multiple_splits']}**)"
        )
    lines.append("")

    lines.append("### Rows per split\n")
    table = pd.DataFrame(
        sorted(diag["split_sizes"].items()), columns=["Split", "Rows"]
    )
    table["Share"] = (table["Rows"] / max(diag["n_rows"], 1)).map("{:.1%}".format)
    lines.append(dataframe_to_markdown(table))
    lines.append("")

    for key, title in (
        ("language_by_split", "Language x split"),
        ("legal_domain_by_split", "Domain x split"),
        ("score_bucket_by_split", "Score band x split"),
    ):
        if key not in diag:
            continue
        lines.append(f"### {title}\n")
        frame = pd.DataFrame(diag[key]).T.fillna(0).astype(int)
        frame.index.name = "split"
        lines.append(dataframe_to_markdown(frame.reset_index()))
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign train/validation/test splits.")
    parser.add_argument("--in", dest="input", type=str, default=None)
    parser.add_argument("--out", dest="output", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--report", action="store_true", help="Write split diagnostics")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    setup_logging("INFO")
    split_seed = effective_split_seed(cfg, args.seed)
    set_global_seed(split_seed)

    from src.data.load_data import add_derived_columns, master_path

    input_path = Path(args.input) if args.input else master_path(cfg)
    output_path = Path(args.output) if args.output else input_path

    # Go through the schema coercion before deriving: a CSV column that is
    # entirely empty (all of `decision`, before any human annotation exists)
    # is read back as float64 and will reject the string labels we assign.
    df = read_csv(input_path)
    df = schema.ensure_columns(df)
    df = schema.coerce_types(df)
    df = add_derived_columns(df, cfg)
    df = assign_splits(df, cfg, seed=split_seed)

    write_csv(df, output_path)

    diag = split_diagnostics(df, cfg)
    logger.info("Split sizes: %s", diag["split_sizes"])
    if diag.get("groups_spanning_multiple_splits"):
        logger.error(
            "%d source group(s) span multiple splits — this is leakage.",
            diag["groups_spanning_multiple_splits"],
        )
        return 1

    if args.report:
        reports_dir = resolve_path(cfg, "reports_dir", create=True)
        results_dir = resolve_path(cfg, "results_dir", create=True)
        write_json(
            {"provenance": provenance(cfg, seed=split_seed), **diag},
            results_dir / "split_diagnostics.json",
        )
        path = reports_dir / "split_diagnostics.md"
        path.write_text(
            "# Split Diagnostics\n\n" + render_diagnostics(diag), encoding="utf-8"
        )
        logger.info("Wrote %s", path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
