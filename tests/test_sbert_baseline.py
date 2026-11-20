"""Tests for the offline-safe multilingual Sentence-BERT baseline."""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
import pytest

from src.models.common import load_model_config
from src.utils.io import read_csv, read_json


class FakeEncoder:
    """Small deterministic encoder that records the public encode contract."""

    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings
        self.calls: list[dict[str, object]] = []
        self.max_seq_length: int | None = None

    def encode(self, sentences, **kwargs):
        values = list(sentences)
        self.calls.append({"sentences": values, **kwargs})
        return np.asarray([self.embeddings[value] for value in values], dtype=float)


class QueueEncoder:
    """Encoder returning predetermined arrays for validation failure tests."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = list(outputs)
        self.max_seq_length: int | None = None

    def encode(self, sentences, **kwargs):
        return self.outputs.pop(0)


class DeterministicTextEncoder:
    """Download-free encoder for end-to-end smoke tests with arbitrary text."""

    def __init__(self) -> None:
        self.max_seq_length: int | None = None

    def encode(self, sentences, **kwargs):
        return np.asarray(
            [
                [
                    float(len(text) + 1),
                    float(sum(ord(char) for char in text) % 997 + 1),
                    float(text.count(" ") + 1),
                ]
                for text in sentences
            ],
            dtype=float,
        )


def _synthetic_frame() -> pd.DataFrame:
    rows = [
        ("T_SAME", "train", "o_same", "s_same", 10.0),
        ("T_ORTHOGONAL", "train", "o_orthogonal", "s_orthogonal", 5.5),
        ("T_OPPOSITE", "train", "o_opposite", "s_opposite", 1.0),
        ("V_ONE", "validation", "o_validation", "s_validation", 7.0),
        ("X_ONE", "test", "o_test", "s_test", 4.0),
    ]
    return pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "language": "hi" if pair_id in {"T_OPPOSITE", "X_ONE"} else "en",
                "legal_domain": "cyber" if split == "validation" else "contract",
                "source_group": f"synthetic::{pair_id}",
                "split": split,
                "original_text": original,
                "simplified_text": simplified,
                "label": label,
            }
            for pair_id, split, original, simplified, label in rows
        ]
    )


def _known_embeddings() -> dict[str, list[float]]:
    return {
        "o_same": [3.0, 0.0],
        "s_same": [2.0, 0.0],
        "o_orthogonal": [1.0, 0.0],
        "s_orthogonal": [0.0, 4.0],
        "o_opposite": [5.0, 0.0],
        "s_opposite": [-2.0, 0.0],
        "o_validation": [1.0, 1.0],
        "s_validation": [1.0, 0.0],
        "o_test": [1.0, 2.0],
        "s_test": [2.0, 1.0],
    }


def test_sbert_configuration_is_loaded_from_project_config() -> None:
    from src.models.sbert_baseline import SbertConfig

    settings = SbertConfig.from_project_config(load_model_config())

    assert settings.model_name == (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    assert settings.batch_size == 32
    assert settings.normalize_embeddings is True
    assert settings.max_seq_length == 128
    assert settings.calibration == "linear_regression"
    assert settings.clip_to_score_range is True


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("model_name", "", "model_name"),
        ("batch_size", 0, "batch_size"),
        ("normalize_embeddings", "yes", "normalize_embeddings"),
        ("max_seq_length", 0, "max_seq_length"),
        ("calibration", "isotonic", "linear_regression"),
        ("clip_to_score_range", 1, "clip_to_score_range"),
    ],
)
def test_invalid_sbert_settings_have_explicit_errors(setting, value, message) -> None:
    from src.models.sbert_baseline import SbertConfig

    cfg = load_model_config(overrides={"models": {"sbert": {setting: value}}})

    with pytest.raises(ValueError, match=message):
        SbertConfig.from_project_config(cfg)


def test_raw_similarities_encode_each_side_separately_and_compute_known_cosines() -> None:
    from src.models.sbert_baseline import SbertBaseline

    frame = _synthetic_frame().iloc[:3].copy()
    encoder = FakeEncoder(_known_embeddings())
    cfg = load_model_config(overrides={"models": {"sbert": {"batch_size": 2}}})
    model = SbertBaseline(cfg, encoder=encoder, device="cpu")

    similarities = model.raw_similarities(frame)

    assert similarities.columns.tolist() == ["cosine_similarity"]
    assert similarities["cosine_similarity"].tolist() == pytest.approx([1.0, 0.0, -1.0])
    assert encoder.max_seq_length == 128
    assert [call["sentences"] for call in encoder.calls] == [
        frame["original_text"].tolist(),
        frame["simplified_text"].tolist(),
    ]
    for call in encoder.calls:
        assert call["batch_size"] == 2
        assert call["normalize_embeddings"] is True
        assert call["convert_to_numpy"] is True
        assert call["show_progress_bar"] is False


def test_fit_calibrates_on_train_only_and_predictions_preserve_order_without_labels() -> None:
    from src.models.sbert_baseline import SbertBaseline

    frame = _synthetic_frame().iloc[[3, 0, 4, 2, 1]].reset_index(drop=True)
    encoder = FakeEncoder(_known_embeddings())
    model = SbertBaseline(load_model_config(), encoder=encoder, device="cpu")

    model.fit(frame, label_column="label")

    train = frame.loc[frame["split"] == "train"]
    assert model.calibrator is not None
    assert model.calibrator.fit_pair_ids == tuple(train["pair_id"])
    assert set(model.calibrator.fit_splits) == {"train"}
    assert [call["sentences"] for call in encoder.calls] == [
        train["original_text"].tolist(),
        train["simplified_text"].tolist(),
    ]

    encoder.calls.clear()
    predictions = model.predict(frame)

    assert predictions.columns.tolist() == [
        "pair_id",
        "language",
        "legal_domain",
        "source_group",
        "split",
        "cosine_similarity",
        "prediction",
    ]
    assert predictions["pair_id"].tolist() == frame["pair_id"].tolist()
    assert predictions["language"].tolist() == frame["language"].tolist()
    assert predictions["legal_domain"].tolist() == frame["legal_domain"].tolist()
    assert predictions["split"].tolist() == frame["split"].tolist()
    assert predictions["prediction"].between(1.0, 10.0).all()
    assert [call["sentences"] for call in encoder.calls] == [
        frame["original_text"].tolist(),
        frame["simplified_text"].tolist(),
    ]
    assert not {
        "label",
        "human_lmp_score",
        "draft_lmp_score",
        "decision",
        "error_type",
        "annotation_status",
    } & set(predictions.columns)


def test_predictions_retain_source_group_when_the_input_provides_it() -> None:
    from src.models.sbert_baseline import SbertBaseline

    frame = _synthetic_frame()
    predictions = SbertBaseline(
        load_model_config(), encoder=FakeEncoder(_known_embeddings()), device="cpu"
    ).fit(frame, label_column="label").predict(frame)

    assert predictions["source_group"].tolist() == frame["source_group"].tolist()


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_fit_rejects_malformed_pair_ids_through_foundation(bad_id) -> None:
    from src.models.sbert_baseline import SbertBaseline

    frame = _synthetic_frame()
    frame.loc[0, "pair_id"] = bad_id

    with pytest.raises(ValueError, match="null or blank pair_id"):
        SbertBaseline(
            load_model_config(), encoder=FakeEncoder(_known_embeddings()), device="cpu"
        ).fit(frame, label_column="label")


def test_fit_rejects_duplicate_pair_ids_through_foundation() -> None:
    from src.models.sbert_baseline import SbertBaseline

    frame = _synthetic_frame()
    frame.loc[1, "pair_id"] = frame.loc[0, "pair_id"]

    with pytest.raises(ValueError, match="Duplicate pair_id"):
        SbertBaseline(
            load_model_config(), encoder=FakeEncoder(_known_embeddings()), device="cpu"
        ).fit(frame, label_column="label")


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        ([np.ones((3, 2)), np.ones((3, 3))], "same shape"),
        ([np.ones(2)], "two-dimensional"),
        ([np.ones((2, 2))], "returned 2 rows.*expected 3"),
        ([np.array([[np.nan, 1.0], [1.0, 0.0], [1.0, 0.0]])], "finite"),
        ([np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])], "zero vectors"),
        ([np.empty((3, 0))], "nonzero dimension"),
    ],
)
def test_raw_similarity_rejects_malformed_embeddings(outputs, message) -> None:
    from src.models.sbert_baseline import SbertBaseline

    model = SbertBaseline(
        load_model_config(), encoder=QueueEncoder(outputs), device="cpu"
    )

    with pytest.raises(ValueError, match=message):
        model.raw_similarities(_synthetic_frame().iloc[:3].copy())


def test_calibrated_predictions_are_explicitly_clipped_to_score_range() -> None:
    from src.models.calibration import LinearScoreCalibrator
    from src.models.sbert_baseline import SbertBaseline

    frame = _synthetic_frame()
    model = SbertBaseline(
        load_model_config(), encoder=FakeEncoder(_known_embeddings()), device="cpu"
    ).fit(frame, label_column="label")
    assert model.calibrator is not None
    model.calibrator = LinearScoreCalibrator(
        slope=100.0,
        intercept=-50.0,
        fit_records=model.calibrator.fit_records,
    )

    predictions = model.predict(frame).set_index("pair_id")

    assert predictions.loc["T_SAME", "prediction"] == 10.0
    assert predictions.loc["T_OPPOSITE", "prediction"] == 1.0
    assert predictions["prediction"].between(1.0, 10.0).all()


def test_draft_field_cannot_be_used_as_calibration_label() -> None:
    from src.models.sbert_baseline import SbertBaseline

    frame = _synthetic_frame().assign(draft_lmp_score=lambda value: value["label"])

    with pytest.raises(ValueError, match="draft_lmp_score.*never"):
        SbertBaseline(
            load_model_config(), encoder=FakeEncoder(_known_embeddings()), device="cpu"
        ).fit(frame, label_column="draft_lmp_score")


def test_real_encoder_loading_is_lazy_offline_and_never_trusts_remote_code(
    monkeypatch,
) -> None:
    from src.models import sbert_baseline

    calls = []
    loaded = FakeEncoder(_known_embeddings())

    def constructor(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return loaded

    monkeypatch.setattr(
        sbert_baseline, "_load_sentence_transformer_class", lambda: constructor
    )
    model = sbert_baseline.SbertBaseline(load_model_config(), device="cpu")
    assert calls == []

    model.raw_similarities(_synthetic_frame().iloc[:3].copy())

    assert calls == [
        (
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            {
                "device": "cpu",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]
    assert loaded.max_seq_length == 128


def test_offline_loading_blocks_huggingface_network_environment_and_restores_it(
    monkeypatch,
) -> None:
    from src.models import sbert_baseline

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    observed = []
    loaded = FakeEncoder(_known_embeddings())

    def constructor(model_name, **kwargs):
        observed.append(
            (
                os.environ.get("HF_HUB_OFFLINE"),
                os.environ.get("TRANSFORMERS_OFFLINE"),
            )
        )
        return loaded

    monkeypatch.setattr(
        sbert_baseline, "_load_sentence_transformer_class", lambda: constructor
    )

    sbert_baseline.SbertBaseline(
        load_model_config(), device="cpu"
    ).raw_similarities(_synthetic_frame().iloc[:3].copy())

    assert observed == [("1", "1")]
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_allow_download_explicitly_disables_local_files_only(monkeypatch) -> None:
    from src.models import sbert_baseline

    calls = []
    loaded = FakeEncoder(_known_embeddings())

    def constructor(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return loaded

    monkeypatch.setattr(
        sbert_baseline, "_load_sentence_transformer_class", lambda: constructor
    )
    model = sbert_baseline.SbertBaseline(
        load_model_config(), device="cpu", allow_download=True
    )

    model.raw_similarities(_synthetic_frame().iloc[:3].copy())

    assert calls[0][1]["local_files_only"] is False
    assert calls[0][1]["trust_remote_code"] is False


def test_missing_local_model_has_explicit_identity_and_no_fallback(monkeypatch) -> None:
    from src.models import sbert_baseline

    calls = []

    def unavailable(model_name, **kwargs):
        calls.append((model_name, kwargs))
        raise OSError("not present in local cache")

    monkeypatch.setattr(
        sbert_baseline, "_load_sentence_transformer_class", lambda: unavailable
    )
    model = sbert_baseline.SbertBaseline(load_model_config(), device="cpu")

    with pytest.raises(
        RuntimeError,
        match=(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
            ".*local_files_only=True.*--allow-download"
        ),
    ):
        model.raw_similarities(_synthetic_frame().iloc[:3].copy())

    assert len(calls) == 1
    assert model.encoder is None


def test_sbert_run_rejects_non_boolean_download_policy_before_output(tmp_path) -> None:
    from src.models.sbert_baseline import run_sbert_baseline

    output = tmp_path / "must-not-exist.csv"
    with pytest.raises(ValueError, match="allow_download must be a boolean"):
        run_sbert_baseline(
            _synthetic_frame(),
            load_model_config(),
            output_path=output,
            label_column="label",
            run_mode="smoke_test",
            synthetic=True,
            encoder=FakeEncoder(_known_embeddings()),
            allow_download=1,
            device="cpu",
        )

    assert not output.exists()
    assert not output.with_suffix(".metadata.json").exists()


@pytest.mark.parametrize("collision_target", ["prediction", "sidecar"])
def test_sbert_rejects_authoritative_master_output_collisions_without_writing(
    tmp_path, collision_target
) -> None:
    from src.models.sbert_baseline import run_sbert_baseline

    output = tmp_path / "lmp_pairs.csv"
    master = (
        output
        if collision_target == "prediction"
        else output.with_suffix(".metadata.json")
    )
    sentinel = b"authoritative-input-sentinel\n"
    master.write_bytes(sentinel)
    cfg = load_model_config(
        overrides={
            "paths": {"processed_dir": str(tmp_path)},
            "dataset": {"master_file": master.name},
        }
    )

    with pytest.raises(ValueError, match="authoritative.*master|input artifact"):
        run_sbert_baseline(
            _synthetic_frame(),
            cfg,
            output_path=output,
            label_column="label",
            run_mode="smoke_test",
            synthetic=True,
            encoder=FakeEncoder(_known_embeddings()),
            device="cpu",
        )

    assert master.read_bytes() == sentinel
    if collision_target == "prediction":
        assert not output.with_suffix(".metadata.json").exists()
    else:
        assert not output.exists()


def test_final_result_sidecar_records_exact_false_smoke_test(tmp_path) -> None:
    from src.models.sbert_baseline import run_sbert_baseline

    frame = _synthetic_frame().rename(columns={"label": "human_lmp_score"})
    frame["annotation_status"] = "human_single"
    frame["annotator_id"] = "annotator-1"
    frame["adjudication_notes"] = "verified"

    paths = run_sbert_baseline(
        frame,
        load_model_config(),
        output_path=tmp_path / "sbert-final.csv",
        label_column="human_lmp_score",
        run_mode="final",
        synthetic=False,
        encoder=FakeEncoder(_known_embeddings()),
        device="cpu",
    )

    assert read_json(paths.metadata_json)["smoke_test"] is False


def test_smoke_result_bundle_has_hashes_fit_provenance_and_nonresearch_markers(
    tmp_path,
) -> None:
    from src.models.sbert_baseline import run_sbert_baseline

    frame = _synthetic_frame()
    output = tmp_path / "smoke" / "sbert_predictions.csv"

    paths = run_sbert_baseline(
        frame,
        load_model_config(),
        output_path=output,
        label_column="label",
        run_mode="smoke_test",
        synthetic=True,
        encoder=FakeEncoder(_known_embeddings()),
        device="cpu",
    )

    written = read_csv(paths.predictions_csv)
    metadata = read_json(paths.metadata_json)
    assert paths.predictions_csv == output
    assert paths.metadata_json == output.with_suffix(".metadata.json")
    assert written.columns.tolist() == [
        "pair_id",
        "language",
        "legal_domain",
        "source_group",
        "split",
        "cosine_similarity",
        "prediction",
        "run_mode",
        "data_source",
        "research_use",
    ]
    assert written["pair_id"].tolist() == frame["pair_id"].tolist()
    assert set(written["run_mode"]) == {"smoke_test"}
    assert set(written["data_source"]) == {"synthetic"}
    assert set(written["research_use"]) == {"non-research"}
    assert not {
        "label",
        "human_lmp_score",
        "draft_lmp_score",
        "decision",
        "error_type",
        "annotation_status",
    } & set(written.columns)
    assert metadata["model_identity"] == {
        "library": "sentence-transformers",
        "model_name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "revision": None,
    }
    assert metadata["loading_policy"] == {
        "allow_download": False,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert metadata["encoder_fit"] == {
        "fit_on_corpus": False,
        "fit_records": [],
    }
    assert metadata["encoder_runtime"] == {
        "source": "injected_test_encoder",
        "configured_model_weights_used": False,
    }
    assert metadata["calibration"]["fit_records"] == [
        {"pair_id": pair_id, "split": "train"}
        for pair_id in frame.loc[frame["split"] == "train", "pair_id"]
    ]
    assert metadata["score_range"] == [1.0, 10.0]
    assert metadata["explicit_clipping"] is True
    assert metadata["run_mode"] == "smoke_test"
    assert metadata["synthetic"] is True
    assert metadata["data_source"] == "synthetic"
    assert metadata["research_use"] == "non-research"
    assert "metrics" not in metadata
    assert len(metadata["code_sha256"]) == 64
    assert len(metadata["input_data_sha256"]) == 64
    assert metadata["predictions_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()


def test_smoke_main_with_injected_encoder_leaves_master_byte_identical(
    tmp_path, repo_root
) -> None:
    from src.models.sbert_baseline import main

    master = repo_root / "data" / "processed" / "lmp_pairs.csv"
    before = hashlib.sha256(master.read_bytes()).hexdigest()
    output = tmp_path / "cli-smoke" / "sbert_predictions.csv"

    return_code = main(
        [
            "--config",
            str(repo_root / "configs" / "config.yaml"),
            "--smoke-test",
            "--output",
            str(output),
        ],
        encoder=DeterministicTextEncoder(),
    )

    assert return_code == 0
    assert hashlib.sha256(master.read_bytes()).hexdigest() == before
    written = read_csv(output)
    metadata = read_json(output.with_suffix(".metadata.json"))
    assert len(written) <= 6
    assert set(written["run_mode"]) == {"smoke_test"}
    assert set(written["data_source"]) == {"synthetic"}
    assert set(written["research_use"]) == {"non-research"}
    assert metadata["synthetic"] is True
    assert metadata["research_use"] == "non-research"
    assert "metrics" not in metadata


def test_final_main_runs_gold_loader_before_model_construction_or_output(
    monkeypatch, tmp_path
) -> None:
    from src.models import sbert_baseline

    calls = []

    def blocked_loader(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("TODO_HUMAN_ANNOTATION")

    def forbidden_model(*args, **kwargs):
        raise AssertionError("SentenceTransformer path ran before the final gold gate")

    monkeypatch.setattr(sbert_baseline, "load_model_corpus", blocked_loader)
    monkeypatch.setattr(sbert_baseline, "SbertBaseline", forbidden_model)
    output = tmp_path / "must-not-exist" / "sbert_predictions.csv"

    return_code = sbert_baseline.main(
        ["--config", "configs/config.yaml", "--output", str(output)]
    )

    assert return_code == 1
    assert calls == [
        ((), {"config_path": "configs/config.yaml", "mode": "final"})
    ]
    assert not output.exists()
    assert not output.with_suffix(".metadata.json").exists()
    assert not output.parent.exists()
