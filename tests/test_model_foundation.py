"""Tests for the model-agnostic training and inference foundation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data import load_data
from src import models
from src.models import common as model_common
from src.models.common import (
    load_model_config,
    load_model_corpus,
    prepare_model_run,
    require_gold,
    select_device,
    setup_model_seed,
    validate_model_corpus,
    validate_predictions,
)
from src.models.provenance import (
    build_result_metadata,
    dataframe_sha256,
    ensure_output_directory,
    file_sha256,
    write_result_bundle,
)
from src.utils.io import read_csv, read_json, write_csv


def test_models_package_exports_the_foundation_api() -> None:
    assert {
        "LinearScoreCalibrator",
        "ModelRunContext",
        "build_result_metadata",
        "load_model_corpus",
        "prepare_model_run",
        "select_device",
        "validate_predictions",
        "write_result_bundle",
    } <= set(models.__all__)


def test_load_model_config_uses_the_project_configuration() -> None:
    cfg = load_model_config()

    assert cfg["project"]["seed"] == 42
    assert cfg["dataset"]["score_min"] == 1
    assert cfg["dataset"]["score_max"] == 10


@pytest.mark.parametrize("bound", ["score_min", "score_max"])
def test_model_config_rejects_boolean_dataset_score_bounds(bound) -> None:
    from src.utils.config import ConfigError

    with pytest.raises(ConfigError, match=f"dataset\\.{bound}"):
        load_model_config(overrides={"dataset": {bound: True}})


def test_load_model_corpus_filters_with_existing_split_contract(
    tmp_path, master_frame
) -> None:
    corpus_path = write_csv(master_frame, tmp_path / "master.csv")

    corpus = load_model_corpus(
        corpus_path=corpus_path,
        mode="smoke",
        splits=("validation",),
    )

    assert corpus["pair_id"].tolist() == ["ENG_CYB_001", "HIN_CYB_001"]
    assert set(corpus["split"]) == {"validation"}


def test_model_corpus_rejects_duplicate_pair_ids(master_frame) -> None:
    frame = master_frame.copy()
    frame.loc[1, "pair_id"] = frame.loc[0, "pair_id"]

    with pytest.raises(ValueError, match="Duplicate pair_id"):
        validate_model_corpus(frame)


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_model_corpus_rejects_null_or_blank_pair_ids(master_frame, bad_id) -> None:
    frame = master_frame.copy()
    frame.loc[0, "pair_id"] = bad_id

    with pytest.raises(ValueError, match="null or blank"):
        validate_model_corpus(frame)


def test_model_corpus_rejects_unknown_split_values(master_frame) -> None:
    frame = master_frame.copy()
    frame.loc[0, "split"] = "dev"

    with pytest.raises(ValueError, match="Unknown split"):
        validate_model_corpus(frame)


def test_model_corpus_rejects_source_groups_crossing_splits(master_frame) -> None:
    frame = master_frame.copy()
    frame.loc[1, "split"] = "validation"

    with pytest.raises(ValueError, match="source_group.*multiple splits"):
        validate_model_corpus(frame)


def test_require_gold_always_calls_the_existing_gold_gate(
    monkeypatch, master_frame
) -> None:
    calls = []

    def record_call(frame, context="") -> None:
        calls.append((frame, context))

    monkeypatch.setattr(load_data, "assert_gold", record_call)
    gold = master_frame.copy()
    gold["annotation_status"] = "human_consensus"
    gold["human_lmp_score"] = 6.0

    require_gold(gold, context="foundation test")

    assert calls == [(gold, "foundation test")]


def test_require_gold_rejects_human_status_without_a_score(master_frame) -> None:
    frame = master_frame.copy()
    frame["annotation_status"] = "human_consensus"
    frame["human_lmp_score"] = np.nan

    with pytest.raises(ValueError, match="human_lmp_score"):
        require_gold(frame, context="foundation test")


def test_require_gold_rejects_missing_annotation_status(master_frame) -> None:
    frame = master_frame.drop(columns=["annotation_status"]).copy()
    frame["human_lmp_score"] = 6.0

    with pytest.raises(ValueError, match="annotation_status"):
        require_gold(frame, context="foundation test")


@pytest.mark.parametrize("bad_score", [np.inf, -np.inf])
def test_require_gold_rejects_nonfinite_human_scores(master_frame, bad_score) -> None:
    frame = master_frame.copy()
    frame["annotation_status"] = "human_consensus"
    frame["human_lmp_score"] = 6.0
    frame.loc[0, "human_lmp_score"] = bad_score

    with pytest.raises(ValueError, match="finite"):
        require_gold(frame, context="foundation test")


@pytest.mark.parametrize("bad_score", [0.99, 10.01])
def test_require_gold_rejects_out_of_range_human_scores(
    master_frame, bad_score
) -> None:
    frame = master_frame.copy()
    frame["annotation_status"] = "human_consensus"
    frame["human_lmp_score"] = 6.0
    frame.loc[0, "human_lmp_score"] = bad_score

    with pytest.raises(ValueError, match="1.*10"):
        require_gold(frame, context="foundation test")


def test_final_corpus_mode_requires_human_gold(tmp_path, master_frame) -> None:
    corpus_path = write_csv(master_frame, tmp_path / "master.csv")

    with pytest.raises(ValueError, match="TODO_HUMAN_ANNOTATION"):
        load_model_corpus(corpus_path=corpus_path, mode="final")


def test_final_corpus_mode_loads_gold_subset_before_model_use(
    monkeypatch, master_frame
) -> None:
    gold = master_frame.copy()
    gold["annotation_status"] = "human_consensus"
    gold["human_lmp_score"] = 6.0
    subsets = []
    gate_calls = []

    def fake_load_master(*args, subset, **kwargs):
        subsets.append(subset)
        return gold.copy()

    def fake_assert_gold(frame, context="") -> None:
        gate_calls.append((frame["pair_id"].tolist(), context))

    monkeypatch.setattr(load_data, "load_master", fake_load_master)
    monkeypatch.setattr(load_data, "assert_gold", fake_assert_gold)

    loaded = load_model_corpus(mode="final")

    assert subsets == ["gold"]
    assert gate_calls == [(gold["pair_id"].tolist(), "final model run")]
    assert loaded["pair_id"].tolist() == gold["pair_id"].tolist()


def test_select_device_supports_explicit_cpu() -> None:
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_device("auto") == torch.device("cuda")


def test_select_device_auto_uses_mps_before_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_device("auto") == torch.device("mps")


def test_select_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert select_device("auto") == torch.device("cpu")


@pytest.mark.parametrize("requested", ["cuda", "mps"])
def test_select_device_rejects_unavailable_accelerator(
    monkeypatch, requested
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="not available"):
        select_device(requested)


def test_predictions_are_aligned_to_expected_pair_id_order() -> None:
    predictions = pd.DataFrame(
        {"pair_id": ["P2", "P1"], "prediction": [7.5, 2.5]}
    )

    aligned = validate_predictions(predictions, ["P1", "P2"])

    assert aligned.to_dict("records") == [
        {"pair_id": "P1", "prediction": 2.5},
        {"pair_id": "P2", "prediction": 7.5},
    ]


def test_predictions_reject_duplicate_pair_ids() -> None:
    predictions = pd.DataFrame(
        {"pair_id": ["P1", "P1"], "prediction": [2.0, 3.0]}
    )

    with pytest.raises(ValueError, match="Duplicate pair_id"):
        validate_predictions(predictions, ["P1"])


@pytest.mark.parametrize("bad_id", [None, "", "  "])
def test_predictions_reject_null_or_blank_pair_ids(bad_id) -> None:
    predictions = pd.DataFrame({"pair_id": [bad_id], "prediction": [4.0]})

    with pytest.raises(ValueError, match="null or blank"):
        validate_predictions(predictions, ["P1"])


def test_predictions_reject_missing_expected_ids() -> None:
    predictions = pd.DataFrame({"pair_id": ["P1"], "prediction": [4.0]})

    with pytest.raises(ValueError, match="Missing expected prediction IDs.*P2"):
        validate_predictions(predictions, ["P1", "P2"])


def test_predictions_reject_unknown_ids() -> None:
    predictions = pd.DataFrame(
        {"pair_id": ["P1", "UNKNOWN"], "prediction": [4.0, 5.0]}
    )

    with pytest.raises(ValueError, match="Unknown prediction IDs.*UNKNOWN"):
        validate_predictions(predictions, ["P1"])


@pytest.mark.parametrize("bad_score", [np.nan, np.inf, -np.inf])
def test_predictions_reject_nonfinite_scores(bad_score) -> None:
    predictions = pd.DataFrame({"pair_id": ["P1"], "prediction": [bad_score]})

    with pytest.raises(ValueError, match="finite"):
        validate_predictions(predictions, ["P1"])


def test_predictions_reject_scores_outside_one_to_ten_by_default() -> None:
    predictions = pd.DataFrame(
        {"pair_id": ["LOW", "HIGH"], "prediction": [0.5, 10.5]}
    )

    with pytest.raises(ValueError, match="outside.*1.*10"):
        validate_predictions(predictions, ["LOW", "HIGH"])


def test_predictions_clip_only_when_explicitly_requested() -> None:
    predictions = pd.DataFrame(
        {"pair_id": ["LOW", "HIGH"], "prediction": [0.5, 10.5]}
    )

    clipped = validate_predictions(
        predictions, ["LOW", "HIGH"], clip=True
    )

    assert clipped["prediction"].tolist() == [1.0, 10.0]


def test_setup_model_seed_uses_training_configuration(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        model_common.config_utils,
        "set_global_seed",
        lambda seed, deterministic: calls.append((seed, deterministic)),
    )
    cfg = {"project": {"seed": 2}, "training": {"seed": 17, "deterministic": True}}

    used = setup_model_seed(cfg)

    assert used == 17
    assert calls == [(17, True)]


def test_result_metadata_includes_reproducibility_and_pair_provenance() -> None:
    cfg = load_model_config()

    metadata = build_result_metadata(
        cfg,
        model_name="foundation-test",
        run_mode="smoke",
        pair_ids=["P1", "P2"],
        splits=["train"],
        seed=17,
        device="cpu",
    )

    assert {
        "generated_at_utc",
        "git_revision",
        "config_fingerprint",
        "project_version",
    } <= metadata["provenance"].keys()
    assert metadata["model_name"] == "foundation-test"
    assert metadata["run_mode"] == "smoke"
    assert metadata["seed"] == 17
    assert metadata["device"] == "cpu"
    assert metadata["n_predictions"] == 2
    assert metadata["splits"] == ["train"]
    assert len(metadata["pair_id_sha256"]) == 64
    assert len(metadata["code_sha256"]) == 64


@pytest.mark.parametrize("bad_seed", [True, "17"])
def test_result_metadata_rejects_non_integer_seed(bad_seed) -> None:
    with pytest.raises(ValueError, match="seed must be a non-bool integer"):
        build_result_metadata(
            load_model_config(),
            model_name="foundation-test",
            run_mode="smoke",
            pair_ids=["P1"],
            splits=["train"],
            seed=bad_seed,
            device="cpu",
        )


def test_provenance_hash_helpers_track_file_and_dataframe_content(tmp_path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same", encoding="utf-8")
    second.write_text("same", encoding="utf-8")
    frame = pd.DataFrame({"pair_id": ["P1"], "prediction": [5.0]})

    assert file_sha256(first) == file_sha256(second)
    assert len(file_sha256(first)) == 64
    assert len(dataframe_sha256(frame)) == 64
    assert dataframe_sha256(frame) != dataframe_sha256(
        frame.assign(prediction=[6.0])
    )


@pytest.mark.parametrize(
    ("pair_ids", "message"),
    [
        (["P1", "P1"], "Duplicate pair_id"),
        (["P1", ""], "null or blank"),
        (["P1", None], "null or blank"),
    ],
)
def test_result_metadata_rejects_invalid_pair_id_provenance(
    pair_ids, message
) -> None:
    with pytest.raises(ValueError, match=message):
        build_result_metadata(
            load_model_config(),
            model_name="foundation-test",
            run_mode="smoke",
            pair_ids=pair_ids,
            splits=["train"],
            seed=17,
            device="cpu",
        )


def test_output_directory_is_created_safely_under_an_allowed_root(tmp_path) -> None:
    target = tmp_path / "nested" / "model"

    created = ensure_output_directory(target, allowed_root=tmp_path)

    assert created == target.resolve()
    assert created.is_dir()


def test_output_directory_cannot_escape_allowed_root(tmp_path) -> None:
    allowed_root = tmp_path / "allowed"
    outside = tmp_path / "outside"

    with pytest.raises(ValueError, match="outside allowed root"):
        ensure_output_directory(outside, allowed_root=allowed_root)

    assert not outside.exists()


def test_output_directory_rejects_an_existing_file(tmp_path) -> None:
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("occupied", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        ensure_output_directory(output_path)


def test_result_bundle_uses_project_writers_and_contains_no_label_columns(
    tmp_path,
) -> None:
    cfg = load_model_config()
    predictions = pd.DataFrame(
        {
            "pair_id": ["P1", "P2"],
            "prediction": [3.0, 8.0],
            "human_lmp_score": [2.0, 9.0],
            "draft_lmp_score": [7.0, 7.0],
            "error_type": ["Omission", "None"],
            "draft_error_type": ["None", "Scope"],
        }
    )
    metadata = build_result_metadata(
        cfg,
        model_name="foundation-test",
        run_mode="smoke",
        pair_ids=predictions["pair_id"],
        splits=["train"],
        seed=42,
        device="cpu",
    )

    paths = write_result_bundle(
        predictions,
        metadata,
        tmp_path / "outputs",
        stem="foundation",
        allowed_root=tmp_path,
    )

    written = read_csv(paths.predictions_csv)
    assert written.columns.tolist() == ["pair_id", "prediction"]
    assert read_json(paths.metadata_json) == metadata


@pytest.mark.parametrize(
    ("predictions", "message"),
    [
        (
            pd.DataFrame({"pair_id": ["P1", "P1"], "prediction": [2.0, 3.0]}),
            "Duplicate pair_id",
        ),
        (
            pd.DataFrame({"pair_id": ["P1"], "prediction": [np.inf]}),
            "finite",
        ),
        (
            pd.DataFrame({"pair_id": ["P1"], "prediction": [10.5]}),
            "1.*10",
        ),
    ],
)
def test_result_bundle_rejects_invalid_predictions_before_creating_output(
    tmp_path, predictions, message
) -> None:
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match=message):
        write_result_bundle(predictions, {}, output_dir)

    assert not output_dir.exists()


def test_result_bundle_rejects_non_integer_prediction_count_before_output(tmp_path) -> None:
    predictions = pd.DataFrame({"pair_id": ["P1"], "prediction": [5.0]})
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="invalid prediction count"):
        write_result_bundle(
            predictions,
            {"n_predictions": "1"},
            output_dir,
        )

    assert not output_dir.exists()


def test_result_bundle_rejects_mismatched_pair_provenance(tmp_path) -> None:
    predictions = pd.DataFrame({"pair_id": ["P1"], "prediction": [5.0]})
    metadata = build_result_metadata(
        load_model_config(),
        model_name="foundation-test",
        run_mode="smoke",
        pair_ids=["DIFFERENT"],
        splits=["train"],
        seed=42,
        device="cpu",
    )
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="metadata.*pair-ID provenance"):
        write_result_bundle(predictions, metadata, output_dir)

    assert not output_dir.exists()


def test_unannotated_corpus_blocks_final_run_before_output_creation(
    tmp_path, unannotated_corpus_env
) -> None:
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="TODO_HUMAN_ANNOTATION"):
        prepare_model_run(
            model_name="foundation-test",
            mode="final",
            corpus_path=unannotated_corpus_env["csv_path"],
            device="cpu",
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_production_corpus_allows_final_run_preparation(tmp_path) -> None:
    """Production corpus with genuine gold allows final run preparation."""
    output_dir = tmp_path / "production_run_out"
    ctx = prepare_model_run(
        model_name="foundation-test",
        mode="final",
        device="cpu",
        output_dir=output_dir,
    )
    assert len(ctx.corpus) == 340
    assert ctx.frames["train"] is not None
    assert len(ctx.frames["train"]) == 234


def test_smoke_run_prepares_seeded_split_frames_and_output_directory(
    tmp_path, master_frame
) -> None:
    corpus_path = write_csv(master_frame, tmp_path / "master.csv")
    output_dir = tmp_path / "outputs" / "smoke"

    context = prepare_model_run(
        model_name="foundation-test",
        mode="smoke",
        corpus_path=corpus_path,
        splits=("train",),
        seed=19,
        device="cpu",
        output_dir=output_dir,
        output_root=tmp_path,
    )

    assert context.seed == 19
    assert context.device == torch.device("cpu")
    assert context.output_dir == output_dir.resolve()
    assert set(context.corpus["split"]) == {"train"}
    assert set(context.frames) == {"train", "validation", "test"}
    assert len(context.frames["train"]) == len(context.corpus)
    assert context.frames["validation"].empty
    assert context.frames["test"].empty
