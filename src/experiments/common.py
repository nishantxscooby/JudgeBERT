"""Common infrastructure, data integrity safeguards, and utilities for Phase 6 experiments.

Centralises:
- Gold data loading and strict human-annotation gating
- Configuration loading and provenance extraction
- Split and pair-ID validation
- Atomic result writing (JSON, CSV, Markdown)
- Synthetic non-research test fixtures for smoke orchestration
- Standardized blocked-status reporting when human gold labels are pending
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import load_data, schema
from src.data.load_data import assert_gold, load_master
from src.models.provenance import (
    atomic_write_csv,
    atomic_write_json,
    dataframe_sha256,
    ensure_output_directory,
    file_sha256,
)
from src.utils.config import config_fingerprint, load_config
from src.utils.io import (
    RESULT_PENDING,
    TODO_HUMAN_ANNOTATION,
    TODO_RUN_EXPERIMENT,
    git_revision,
    utc_now_iso,
)


class ExperimentBlockedError(ValueError):
    """Raised when an experiment cannot proceed because human gold labels are missing."""

    def __init__(self, message: str = f"Blocked: {TODO_HUMAN_ANNOTATION}") -> None:
        super().__init__(message)
        self.message = message


def load_experiment_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the project configuration."""
    return load_config(config_path)


def load_gold_or_block(
    cfg: Mapping[str, Any] | None = None,
    context: str = "experiment",
) -> pd.DataFrame:
    """Load authoritative human gold annotations or raise ExperimentBlockedError.

    Enforces the non-negotiable data integrity rule:
    Only rows with human_* annotation status and populated human_lmp_score
    are loaded. Never falls back to draft or AI annotations.
    """
    effective_cfg = dict(cfg) if cfg is not None else load_experiment_config()
    gold = load_master(cfg=effective_cfg, subset="gold")

    try:
        assert_gold(gold, context)
    except ValueError as exc:
        raise ExperimentBlockedError(
            f"Blocked: {TODO_HUMAN_ANNOTATION} — {exc}"
        ) from exc

    if gold.empty:
        raise ExperimentBlockedError(
            f"Blocked: {TODO_HUMAN_ANNOTATION} — No human gold annotations exist for {context}."
        )

    # Double-check schema integrity
    if not schema.is_gold(gold).all():
        raise ExperimentBlockedError(
            f"Blocked: {TODO_HUMAN_ANNOTATION} — Data contains rows that do not satisfy schema.is_gold"
        )

    return gold


def assert_no_draft_or_synthetic_leakage(df: pd.DataFrame, context: str = "run") -> None:
    """Assert that no draft, AI-generated, or synthetic data is treated as ground truth."""
    if "draft_lmp_score" in df.columns and "human_lmp_score" in df.columns:
        # Check that human_lmp_score was not simply copied from draft_lmp_score
        non_null_human = df["human_lmp_score"].dropna()
        if not non_null_human.empty and "draft_lmp_score" in df.columns:
            matches_draft = (df["human_lmp_score"] == df["draft_lmp_score"]).all()
            if matches_draft and len(non_null_human) > 10:
                raise ValueError(
                    f"Integrity violation in {context}: human_lmp_score appears identical "
                    "to draft_lmp_score across all rows"
                )

    if "annotation_status" in df.columns:
        statuses = set(df["annotation_status"].dropna().astype(str).unique())
        disallowed = statuses - set(schema.GOLD_STATUSES)
        if disallowed:
            raise ValueError(
                f"Integrity violation in {context}: Disallowed annotation status values "
                f"found in gold set: {sorted(disallowed)}"
            )


def validate_split_and_ids(
    df: pd.DataFrame,
    expected_split: str | None = None,
    id_column: str = "pair_id",
) -> list[str]:
    """Validate pair IDs are non-blank, unique, and splits conform to schema."""
    if id_column not in df.columns:
        raise ValueError(f"DataFrame is missing required identifier column {id_column!r}")

    raw_ids = df[id_column]
    blank = raw_ids.isna() | raw_ids.astype("string").str.strip().eq("").fillna(True)
    if blank.any():
        raise ValueError(
            f"DataFrame contains {int(blank.sum())} null or blank {id_column} value(s)"
        )

    normalized_ids = raw_ids.astype(str).str.strip()
    duplicates = normalized_ids[normalized_ids.duplicated(keep=False)]
    if not duplicates.empty:
        examples = sorted(duplicates.unique())[:5]
        raise ValueError(f"Duplicate {id_column} values: {examples}")

    if "split" in df.columns:
        splits = df["split"].dropna().astype(str).str.strip().unique()
        unknown_splits = set(splits) - set(schema.SPLITS)
        if unknown_splits:
            raise ValueError(f"Unknown split value(s): {sorted(unknown_splits)}; expected {schema.SPLITS}")
        if expected_split is not None:
            non_matching = df[df["split"].astype(str).str.strip() != expected_split]
            if not non_matching.empty:
                raise ValueError(
                    f"DataFrame contains {len(non_matching)} rows outside expected split {expected_split!r}"
                )

    return normalized_ids.tolist()


def get_dataset_fingerprint(cfg: Mapping[str, Any] | None = None) -> str:
    """Compute the SHA-256 fingerprint of the master dataset file."""
    effective_cfg = dict(cfg) if cfg is not None else load_experiment_config()
    dataset_cfg = effective_cfg.get("dataset") or {}
    paths_cfg = effective_cfg.get("paths") or {}
    master_file = dataset_cfg.get("master_file", "data/processed/lmp_pairs.csv")
    master_path = Path(paths_cfg.get("root", ".")) / master_file
    if master_path.is_file():
        return file_sha256(master_path)
    return "dataset_file_not_found"


def build_experiment_provenance(
    cfg: Mapping[str, Any],
    experiment_name: str,
    model_identity: str = "all",
    *,
    is_smoke: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct complete traceable provenance metadata for an experiment run."""
    project = cfg.get("project") or {}
    seed = project.get("seed", 42)
    dataset_sha256 = get_dataset_fingerprint(cfg)

    provenance: dict[str, Any] = {
        "generated_at_utc": utc_now_iso(),
        "git_revision": git_revision(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "config_fingerprint": config_fingerprint(cfg),
        "random_seed": seed,
        "dataset_fingerprint": dataset_sha256,
        "model_identity": model_identity,
        "experiment_name": experiment_name,
        "human_gold_required": not is_smoke,
        "is_smoke": is_smoke,
        "research_valid": not is_smoke,
    }

    if is_smoke:
        provenance["disclaimer"] = (
            "SYNTHETIC NON-RESEARCH DATA — ORCHESTRATION SMOKE TEST ONLY. "
            "DO NOT CITE AS EMPIRICAL FINDINGS."
        )

    if extra:
        provenance.update(extra)

    return provenance


def write_blocked_status(
    output_dir: str | Path,
    experiment_name: str,
    reason: str,
    provenance: Mapping[str, Any],
) -> Path:
    """Write explicit blocked status files when human annotations are pending."""
    out_dir = ensure_output_directory(output_dir)
    status_payload = {
        "experiment_name": experiment_name,
        "status": "BLOCKED",
        "reason": reason,
        "human_annotation_required": True,
        "code": TODO_HUMAN_ANNOTATION,
        "provenance": dict(provenance),
    }

    json_path = out_dir / "blocked.json"
    atomic_write_json(status_payload, json_path)

    md_content = f"""# {experiment_name} — BLOCKED

> **Status:** `BLOCKED`  
> **Code:** `{TODO_HUMAN_ANNOTATION}`  
> **Reason:** {reason}

## Non-Negotiable Data Integrity Policy
This experiment requires authoritative human gold ground truth.
In accordance with the project data integrity policy:
- Draft labels (`draft_lmp_score`, `draft_error_type`) are suggestions, not ground truth.
- Synthetic and AI annotations are never substituted for human labels.
- Evaluation on test splits is blocked until human annotators complete scoring in `src/annotation/annotation_app.py`.

*Generated at: {provenance.get('generated_at_utc', 'unknown')}*
"""
    md_path = out_dir / "blocked.md"
    temp_md = md_path.with_name(f".{md_path.name}.tmp")
    temp_md.write_text(md_content, encoding="utf-8")
    os.replace(temp_md, md_path)

    return json_path


def create_synthetic_experiment_fixture(
    n_per_split: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate an in-memory synthetic fixture for smoke tests.

    Never touches disk and never alters data/processed/lmp_pairs.csv.
    Provides balanced splits across languages (en, hi) and domains
    (contract, cyber, sale_of_goods).
    """
    rng = np.random.RandomState(seed)
    rows: list[dict[str, Any]] = []

    splits = ["train", "validation", "test"]
    languages = ["en", "hi"]
    domains = ["contract", "cyber", "sale_of_goods"]
    error_types = ["None", "Obligation/Permission", "Negation", "Condition", "Scope", "Quantity"]

    counter = 1
    for split in splits:
        for _ in range(n_per_split):
            lang = languages[counter % len(languages)]
            domain = domains[counter % len(domains)]
            err = error_types[counter % len(error_types)]

            # Generate synthetic score and matching decision
            score = float(np.round(rng.uniform(1.0, 10.0), 1))
            if score >= 8.0:
                decision = "Preserved"
            elif score >= 5.0:
                decision = "Partially Preserved"
            else:
                decision = "Not Preserved"

            orig = f"Synthetic statutory sentence {counter} in {lang} for {domain}."
            simp = f"Synthetic simplified sentence {counter} in {lang} for {domain}."

            rows.append({
                "pair_id": f"SMOKE_{lang.upper()}_{domain[:3].upper()}_{counter:04d}",
                "original_text": orig,
                "simplified_text": simp,
                "language": lang,
                "legal_domain": domain,
                "source_group": f"smoke_group_{counter % 8}",
                "split": split,
                "human_lmp_score": score,
                "decision": decision,
                "error_type": err,
                "annotation_status": "human_single",
            })
            counter += 1

    return pd.DataFrame(rows)


def create_mock_predictions(
    df: pd.DataFrame,
    models: Sequence[str] = ("tfidf", "sbert", "judgebert"),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Generate synthetic predictions aligned with a DataFrame for smoke testing."""
    rng = np.random.RandomState(seed)
    result = {}

    for model_name in models:
        scores = []
        for _, row in df.iterrows():
            true_score = float(row.get("human_lmp_score", 5.0))
            # Model-specific slight noise around true score
            noise_scale = 0.8 if model_name == "judgebert" else (1.2 if model_name == "sbert" else 1.8)
            pred = float(np.clip(true_score + rng.normal(0, noise_scale), 1.0, 10.0))
            scores.append(round(pred, 2))

        pred_frame = pd.DataFrame({
            "pair_id": df["pair_id"].astype(str),
            "prediction": scores,
            "model_name": model_name,
            "language": df["language"].astype(str),
            "legal_domain": df["legal_domain"].astype(str),
            "split": df["split"].astype(str),
        })
        result[model_name] = pred_frame

    return result
