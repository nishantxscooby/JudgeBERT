"""Tests for exact prediction-artifact provenance validation."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_digest(pair_ids) -> str:
    payload = json.dumps(list(pair_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_valid_artifact(tmp_path):
    path = tmp_path / "tfidf_predictions.csv"
    frame = pd.DataFrame(
        {
            "pair_id": ["P3", "P1", "P2"],
            "language": ["en", "en", "hi"],
            "legal_domain": ["sale_of_goods", "contract", "cyber"],
            "source_group": ["g3", "g1", "g2"],
            "split": ["test", "train", "validation"],
            "prediction": [8.5, 2.5, 6.5],
            "run_mode": ["final"] * 3,
            "data_source": ["human_gold_corpus"] * 3,
            "research_use": ["research"] * 3,
        }
    )
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    metadata = {
        "provenance": {
            "generated_at_utc": "2026-01-01T00:00:00Z",
            "git_revision": "unknown",
            "config_fingerprint": "abc123def456",
            "project_version": "0.1.0",
            "seed": 42,
        },
        "model_name": "tfidf",
        "run_mode": "final",
        "seed": 42,
        "device": "cpu",
        "n_predictions": 3,
        "splits": ["test", "train", "validation"],
        "pair_id_sha256": _pair_digest(frame["pair_id"]),
        "code_sha256": "b" * 64,
        "synthetic": False,
        "smoke_test": False,
        "data_source": "human_gold_corpus",
        "research_use": "research",
        "research_metrics_reported": False,
        "score_range": [1.0, 10.0],
        "calibration": {
            "schema_version": 1,
            "calibrator_type": "linear_regression",
            "slope": 2.0,
            "intercept": 1.0,
            "score_min": 1.0,
            "score_max": 10.0,
            "fit_records": [
                {"pair_id": "T1", "split": "train"},
                {"pair_id": "T2", "split": "train"},
            ],
        },
        "predictions_sha256": _sha256(path),
        "input_data_sha256": "c" * 64,
    }
    sidecar = path.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return path, sidecar, frame, metadata


def _write_valid_judgebert_artifact(tmp_path):
    path, sidecar, frame, metadata = _write_valid_artifact(tmp_path)
    frame = frame.drop(columns=["data_source"]).copy()
    frame["run_mode"] = "inference"
    frame["research_use"] = "prediction-only"
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    metadata.update(
        {
            "model_name": "judgebert",
            "model_identity": "JUDGEBERT-inspired MuRIL cross-encoder",
            "run_mode": "inference",
            "smoke_test": False,
            "research_use": "prediction-only",
            "checkpoint_run_mode": "final",
            "checkpoint_synthetic": False,
            "checkpoint_sha256": "d" * 64,
            "checkpoint_manifest_sha256": "e" * 64,
            "loading_policy": {
                "allow_download": False,
                "local_files_only": True,
                "trust_remote_code": False,
                "tokenizer_source": "configured_muril",
                "model_source": "configured_muril",
            },
            "predictions_sha256": _sha256(path),
            "output_sha256": _sha256(path),
        }
    )
    metadata.pop("data_source")
    metadata.pop("calibration")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path, sidecar, frame, metadata


def test_valid_prediction_sidecar_is_bound_to_exact_artifact_and_model_run(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, frame, metadata = _write_valid_artifact(tmp_path)

    artifact = load_prediction_artifact(path)

    assert artifact.path == path.resolve()
    assert artifact.sidecar_path == sidecar.resolve()
    assert artifact.frame["pair_id"].tolist() == frame["pair_id"].tolist()
    assert artifact.metadata == metadata
    assert artifact.predictions_sha256 == _sha256(path)
    assert artifact.sidecar_sha256 == _sha256(sidecar)
    assert artifact.model_name == "tfidf"
    assert len(artifact.run_identity) == 64
    assert len(artifact.calibration_identity) == 64
    assert artifact.checkpoint_identity is None
    descriptor = artifact.to_dict()
    assert descriptor["prediction_file"]["sha256"] == _sha256(path)
    assert descriptor["provenance_sidecar"]["sha256"] == _sha256(sidecar)
    assert descriptor["run_provenance"] == {
        "run_mode": "final",
        "seed": 42,
        "config_fingerprint": "abc123def456",
        "git_revision": "unknown",
        "code_sha256": "b" * 64,
    }


def test_prediction_provenance_rejects_tampered_or_stale_artifacts(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, _sidecar, _frame, _metadata = _write_valid_artifact(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("8.5", "8.6", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale or tampered.*predictions_sha256"):
        load_prediction_artifact(path)


def test_prediction_loader_hashes_the_same_csv_bytes_that_it_parses(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import provenance
    from src.utils import io as io_utils

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    original_bytes = path.read_bytes()
    mutated_bytes = original_bytes.replace(b"8.5", b"8.6", 1)
    metadata["predictions_sha256"] = hashlib.sha256(mutated_bytes).hexdigest()
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    real_read_json = io_utils.read_json

    def mutate_csv_after_parse(target):
        parsed = real_read_json(target)
        if target == sidecar:
            path.write_bytes(mutated_bytes)
        return parsed

    monkeypatch.setattr(io_utils, "read_json", mutate_csv_after_parse)

    with pytest.raises(ValueError, match="stale or tampered.*predictions_sha256"):
        provenance.load_prediction_artifact(path)


def test_prediction_provenance_requires_the_exact_sidecar(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, _metadata = _write_valid_artifact(tmp_path)
    sidecar.unlink()

    with pytest.raises(FileNotFoundError, match="sidecar not found.*exact artifact"):
        load_prediction_artifact(path)


def test_prediction_provenance_supports_model_bundle_sidecar_name(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, _metadata = _write_valid_artifact(tmp_path)
    bundle_sidecar = path.with_name("tfidf_metadata.json")
    sidecar.rename(bundle_sidecar)

    artifact = load_prediction_artifact(path)

    assert artifact.sidecar_path == bundle_sidecar.resolve()


def test_final_baseline_smoke_test_must_be_exact_false(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["smoke_test"] = "false"
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="smoke_test must be exactly false"):
        load_prediction_artifact(path)


@pytest.mark.parametrize("bad_value", [3.9, "3", True])
def test_prediction_provenance_rejects_non_integer_prediction_count(
    tmp_path, bad_value
) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["n_predictions"] = bad_value
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="n_predictions must be a non-bool integer"):
        load_prediction_artifact(path)


def test_prediction_provenance_rejects_boolean_score_bounds(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["score_range"] = [True, 10.0]
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="score_range.*finite non-bool numbers"):
        load_prediction_artifact(path)


def test_prediction_provenance_rejects_malformed_calibration_schema(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["calibration"] = {
        "schema_version": 1,
        "calibrator_type": "bogus",
        "score_min": 1.0,
        "score_max": 10.0,
        "fit_records": [{"pair_id": "T1", "split": "train"}],
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown calibrator_type"):
        load_prediction_artifact(path)


def test_prediction_provenance_rejects_test_fitted_calibration(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["calibration"]["fit_records"][1]["split"] = "test"
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="calibration.*never contain test rows"):
        load_prediction_artifact(path)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("predictions_sha256", "predictions_sha256.*SHA-256"),
        ("code_sha256", "code_sha256.*SHA-256"),
        ("pair_id_sha256", "pair_id_sha256.*SHA-256"),
        ("n_predictions", "n_predictions.*non-bool integer"),
        ("score_range", "score_range"),
        ("synthetic", "Synthetic or smoke"),
    ],
)
def test_prediction_provenance_rejects_missing_run_identity_fields(
    tmp_path, field, message
) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata.pop(field)
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises((ValueError, KeyError), match=message):
        load_prediction_artifact(path)


def test_prediction_provenance_rejects_pair_split_and_record_mismatches(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, frame, metadata = _write_valid_artifact(tmp_path)
    metadata["records"] = [
        {"pair_id": pair_id, "split": split}
        for pair_id, split in zip(frame["pair_id"], frame["split"], strict=True)
    ]
    metadata["records"][0]["split"] = "train"
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="records do not match"):
        load_prediction_artifact(path)


def test_judgebert_hash_only_bundle_without_artifacts_is_rejected(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, _sidecar, _frame, _metadata = _write_valid_judgebert_artifact(tmp_path)

    with pytest.raises(ValueError, match="checkpoint artifact reference"):
        load_prediction_artifact(path)


def test_judgebert_provenance_requires_resolved_checkpoint_reference(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_judgebert_artifact(tmp_path)
    metadata["checkpoint_artifact"] = {
        "path": "checkpoint.pt",
        "sha256": metadata["checkpoint_sha256"],
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint artifact reference path must be resolved"):
        load_prediction_artifact(path)


def test_judgebert_provenance_rejects_missing_checkpoint_manifest_identity(
    tmp_path,
) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_judgebert_artifact(tmp_path)
    metadata.pop("checkpoint_manifest_sha256")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_manifest_sha256"):
        load_prediction_artifact(path)


def test_judgebert_provenance_requires_paired_text_input_lineage(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_judgebert_artifact(tmp_path)
    metadata.pop("input_data_sha256")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input_data_sha256"):
        load_prediction_artifact(path)


def test_judgebert_provenance_requires_manifest_artifact_reference(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_judgebert_artifact(tmp_path)
    metadata["checkpoint_artifact"] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint_manifest artifact reference"):
        load_prediction_artifact(path)


def test_final_judgebert_provenance_requires_explicit_false_smoke_marker(
    tmp_path,
) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_judgebert_artifact(tmp_path)
    metadata.pop("smoke_test")
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="smoke_test.*exactly false"):
        load_prediction_artifact(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("synthetic", "false"),
        ("synthetic", 0),
        ("smoke_test", "false"),
        ("smoke_test", 0),
        ("smoke_test", True),
        ("smoke_test", None),
        ("checkpoint_synthetic", "false"),
        ("checkpoint_synthetic", 0),
        ("checkpoint_synthetic", True),
        ("checkpoint_synthetic", None),
        ("run_mode", "final"),
        ("checkpoint_run_mode", "inference"),
        ("checkpoint_run_mode", None),
    ],
)
def test_final_judgebert_rejects_malformed_or_inconsistent_markers(
    tmp_path, field, value
) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_judgebert_artifact(tmp_path)
    metadata[field] = value
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_prediction_artifact(path)


def test_prediction_provenance_rejects_conflicting_seed_fields(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["provenance"]["seed"] = 999
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seed fields do not match"):
        load_prediction_artifact(path)


def test_prediction_provenance_seed_fields_require_exact_integer_types(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["provenance"]["seed"] = 42.0
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="provenance seed must be an integer"):
        load_prediction_artifact(path)


@pytest.mark.parametrize("bad_value", [True, 0, "false", None])
def test_prediction_provenance_requires_explicit_false_research_metrics_marker(
    tmp_path, bad_value
) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["research_metrics_reported"] = bad_value
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="research_metrics_reported.*exactly false"):
        load_prediction_artifact(path)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("allow_download", "false"),
        ("allow_download", 0),
        ("local_files_only", "true"),
        ("local_files_only", 1),
        ("trust_remote_code", 0),
    ],
)
def test_sbert_provenance_requires_strict_loading_policy_booleans(
    tmp_path, field, bad_value
) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["model_name"] = "sbert"
    metadata["model_identity"] = {
        "library": "sentence-transformers",
        "model_name": "example/model",
        "revision": None,
    }
    metadata["encoder_runtime"] = {
        "source": "configured_sentence_transformer",
        "configured_model_weights_used": True,
    }
    metadata["loading_policy"] = {
        "allow_download": False,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    metadata["loading_policy"][field] = bad_value
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="loading_policy.*boolean opposites"):
        load_prediction_artifact(path)


def test_sbert_provenance_rejects_injected_test_encoder_in_final_run(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_artifact(tmp_path)
    metadata["model_name"] = "sbert"
    metadata["model_identity"] = {
        "library": "sentence-transformers",
        "model_name": "example/model",
        "revision": "abc123",
    }
    metadata["encoder_runtime"] = {
        "source": "injected_test_encoder",
        "configured_model_weights_used": False,
    }
    metadata["loading_policy"] = {
        "allow_download": False,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="configured model weights"):
        load_prediction_artifact(path)


def test_judgebert_provenance_rejects_injected_prediction_components(tmp_path) -> None:
    from src.evaluation.provenance import load_prediction_artifact

    path, sidecar, _frame, metadata = _write_valid_judgebert_artifact(tmp_path)
    metadata["loading_policy"]["model_source"] = "injected_test_model"
    metadata["loading_policy"]["tokenizer_source"] = "injected_test_tokenizer"
    sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="configured checkpoint model and tokenizer"):
        load_prediction_artifact(path)
