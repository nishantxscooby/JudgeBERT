"""Tests for the provenance-stamped Phase 1 corpus audit."""

from __future__ import annotations

import json

from src.analysis import audit_corpus
from src.utils.config import load_config


def test_audit_dataframe_records_provenance_and_annotation_integrity(master_frame):
    result = audit_corpus.audit_dataframe(master_frame, load_config())

    assert {
        "generated_at_utc",
        "git_revision",
        "python_version",
        "platform",
        "config_fingerprint",
        "project_version",
        "seed",
    } <= result["provenance"].keys()
    assert result["total_rows"] == len(master_frame)
    assert result["unique_pair_ids"] == master_frame["pair_id"].nunique()
    assert result["human_annotation"]["gold_rows"] == 0
    assert result["human_annotation"]["human_lmp_score_populated"] == 0
    assert result["human_annotation"]["annotation_status_counts"] == {"draft": 6}


def test_audit_dataframe_reports_group_leakage_as_json_serializable_lists(master_frame):
    leaking = master_frame.copy()
    leaking.loc[1, "split"] = "test"

    result = audit_corpus.audit_dataframe(leaking, load_config())

    assert result["source_groups_spanning_splits"] == {"ICA1872::1": ["test", "train"]}
    json.dumps(result)
