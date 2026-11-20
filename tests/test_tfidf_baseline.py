"""Tests for the leakage-safe TF-IDF baseline."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import pandas as pd
import pytest

from src.models.common import load_model_config
from src.utils.io import read_csv, read_json


def _synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": "T_IDENTICAL",
                "language": "en",
                "legal_domain": "contract",
                "source_group": "synthetic::1",
                "split": "train",
                "original_text": "The buyer shall pay within 30 days",
                "simplified_text": "The buyer shall pay within 30 days",
                "label": 10.0,
            },
            {
                "pair_id": "T_PARTIAL",
                "language": "en",
                "legal_domain": "contract",
                "source_group": "synthetic::2",
                "split": "train",
                "original_text": "The seller may cancel the contract after notice",
                "simplified_text": "The seller may cancel",
                "label": 5.0,
            },
            {
                "pair_id": "T_DIFFERENT",
                "language": "hi",
                "legal_domain": "contract",
                "source_group": "synthetic::3",
                "split": "train",
                "original_text": "क्रेता माल लौटाएगा",
                "simplified_text": "न्यायालय अपील सुनेगा",
                "label": 1.0,
            },
            {
                "pair_id": "V_UNSEEN",
                "language": "en",
                "legal_domain": "cyber",
                "source_group": "synthetic::4",
                "split": "validation",
                "original_text": "validationonlytoken remains confidential",
                "simplified_text": "validationonlytoken stays confidential",
                "label": 7.0,
            },
            {
                "pair_id": "X_TEST",
                "language": "hi",
                "legal_domain": "sale_of_goods",
                "source_group": "synthetic::5",
                "split": "test",
                "original_text": "विक्रेता माल देगा",
                "simplified_text": "विक्रेता वस्तु देगा",
                "label": 8.0,
            },
        ]
    )


def _final_frame() -> pd.DataFrame:
    frame = _synthetic_frame().rename(columns={"label": "human_lmp_score"})
    frame["annotation_status"] = "human_single"
    frame["annotator_id"] = "annotator-1"
    frame["adjudication_notes"] = "verified"
    return frame


def test_tfidf_configuration_is_loaded_from_project_config() -> None:
    from src.models.tfidf_baseline import TfidfConfig

    settings = TfidfConfig.from_project_config(load_model_config())

    assert settings.word_ngram_range == (1, 2)
    assert settings.char_ngram_range == (3, 5)
    assert settings.min_df == 1
    assert settings.sublinear_tf is True
    assert settings.use_char_ngrams is True
    assert settings.calibration == "linear_regression"
    assert settings.clip_to_score_range is True


def test_word_and_char_views_compute_rowwise_identical_pair_cosines() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame()
    model = TfidfBaseline(load_model_config())
    model.fit(frame, label_column="label")

    similarities = model.raw_similarities(frame)

    assert similarities.columns.tolist() == [
        "word_cosine_similarity",
        "char_cosine_similarity",
        "cosine_similarity",
    ]
    identical = similarities.iloc[0]
    assert identical["word_cosine_similarity"] == pytest.approx(1.0)
    assert identical["char_cosine_similarity"] == pytest.approx(1.0)
    assert identical["cosine_similarity"] == pytest.approx(1.0)


def test_word_vocabulary_keeps_complete_hindi_content_tokens() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    model = TfidfBaseline(load_model_config()).fit(
        _synthetic_frame(), label_column="label"
    )

    assert "क्रेता" in model.word_vocabulary
    assert "लौटाएगा" in model.word_vocabulary
    assert "क" not in model.word_vocabulary


def test_vocabulary_and_calibration_fit_train_rows_only() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame()
    model = TfidfBaseline(load_model_config()).fit(frame, label_column="label")
    train_ids = tuple(frame.loc[frame["split"] == "train", "pair_id"])

    assert model.vocabulary_fit_pair_ids == train_ids
    assert "validationonlytoken" not in model.word_vocabulary
    assert model.calibrator is not None
    assert model.calibrator.fit_pair_ids == train_ids
    assert set(model.calibrator.fit_splits) == {"train"}


def test_character_branch_can_be_disabled_without_changing_word_branch() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    cfg = load_model_config(
        overrides={"models": {"tfidf": {"use_char_ngrams": False}}}
    )
    model = TfidfBaseline(cfg).fit(_synthetic_frame(), label_column="label")

    assert model.feature_branches == ("word",)
    assert model.char_vectorizer is None
    assert model.raw_similarities(_synthetic_frame()).columns.tolist() == [
        "word_cosine_similarity",
        "cosine_similarity",
    ]


def test_predictions_preserve_pair_metadata_order_and_drop_all_labels() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame().iloc[[3, 0, 4, 1, 2]].reset_index(drop=True)
    model = TfidfBaseline(load_model_config()).fit(frame, label_column="label")

    predictions = model.predict(frame)

    assert predictions.columns.tolist() == [
        "pair_id",
        "language",
        "legal_domain",
        "source_group",
        "split",
        "word_cosine_similarity",
        "char_cosine_similarity",
        "cosine_similarity",
        "prediction",
    ]
    assert predictions["pair_id"].tolist() == frame["pair_id"].tolist()
    assert predictions["language"].tolist() == frame["language"].tolist()
    assert predictions["legal_domain"].tolist() == frame["legal_domain"].tolist()
    assert predictions["split"].tolist() == frame["split"].tolist()
    assert predictions["prediction"].between(1.0, 10.0).all()
    assert not {
        "label",
        "human_lmp_score",
        "draft_lmp_score",
        "error_type",
        "decision",
        "annotation_status",
    } & set(predictions.columns)


def test_predictions_retain_source_group_when_the_input_provides_it() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame()
    predictions = TfidfBaseline(load_model_config()).fit(
        frame, label_column="label"
    ).predict(frame)

    assert predictions["source_group"].tolist() == frame["source_group"].tolist()


@pytest.mark.parametrize("label_column", ["draft_test_label", "curator_draft_label"])
def test_no_draft_field_can_be_used_as_the_calibration_label(label_column) -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame()
    frame[label_column] = frame["label"]

    with pytest.raises(ValueError, match=rf"{label_column}.*never"):
        TfidfBaseline(load_model_config()).fit(
            frame, label_column=label_column
        )


def test_unsupported_tfidf_calibration_has_an_explicit_error() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    cfg = load_model_config(
        overrides={"models": {"tfidf": {"calibration": "isotonic"}}}
    )

    with pytest.raises(ValueError, match="models.tfidf.calibration.*linear_regression"):
        TfidfBaseline(cfg)


@pytest.mark.parametrize("collision_target", ["prediction", "sidecar"])
def test_tfidf_rejects_authoritative_master_output_collisions_without_writing(
    tmp_path, collision_target
) -> None:
    from src.models.tfidf_baseline import run_tfidf_baseline

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
        run_tfidf_baseline(
            _synthetic_frame(),
            cfg,
            output_path=output,
            label_column="label",
            run_mode="smoke_test",
            synthetic=True,
        )

    assert master.read_bytes() == sentinel
    if collision_target == "prediction":
        assert not output.with_suffix(".metadata.json").exists()
    else:
        assert not output.exists()


@pytest.mark.parametrize("alias_target", ["prediction", "sidecar"])
def test_tfidf_rejects_existing_hard_link_alias_to_authoritative_master(
    tmp_path, alias_target
) -> None:
    from src.models.tfidf_baseline import run_tfidf_baseline

    master = tmp_path / "authoritative-master.csv"
    sentinel = b"authoritative-master-sentinel\n"
    master.write_bytes(sentinel)
    output = tmp_path / "tfidf-final.csv"
    target = output if alias_target == "prediction" else output.with_suffix(
        ".metadata.json"
    )
    os.link(master, target)
    cfg = load_model_config(
        overrides={
            "paths": {"processed_dir": str(tmp_path)},
            "dataset": {"master_file": master.name},
        }
    )

    with pytest.raises(ValueError, match="hard-link|authoritative.*master"):
        run_tfidf_baseline(
            _final_frame(),
            cfg,
            output_path=output,
            label_column="human_lmp_score",
            run_mode="final",
            synthetic=False,
        )

    assert master.read_bytes() == sentinel
    assert os.path.samefile(master, target)


@pytest.mark.parametrize("alias_target", ["prediction", "sidecar"])
def test_tfidf_atomic_publication_defeats_hard_link_created_after_validation(
    tmp_path, monkeypatch, alias_target
) -> None:
    from src.models import provenance as model_provenance
    from src.models import tfidf_baseline

    master = tmp_path / "authoritative-master.csv"
    sentinel = b"authoritative-master-sentinel\n"
    master.write_bytes(sentinel)
    output = tmp_path / "tfidf-final.csv"
    target = output if alias_target == "prediction" else output.with_suffix(
        ".metadata.json"
    )
    cfg = load_model_config(
        overrides={
            "paths": {"processed_dir": str(tmp_path)},
            "dataset": {"master_file": master.name},
        }
    )
    real_write_csv = model_provenance.io_utils.write_csv
    injected = False

    def write_csv_after_alias_race(frame, path):
        nonlocal injected
        if not injected:
            os.link(master, target)
            injected = True
        return real_write_csv(frame, path)

    monkeypatch.setattr(
        model_provenance.io_utils, "write_csv", write_csv_after_alias_race
    )

    paths = tfidf_baseline.run_tfidf_baseline(
        _final_frame(),
        cfg,
        output_path=output,
        label_column="human_lmp_score",
        run_mode="final",
        synthetic=False,
    )

    assert injected
    assert master.read_bytes() == sentinel
    assert paths.predictions_csv.is_file()
    assert paths.metadata_json.is_file()
    assert not os.path.samefile(master, target)


def test_final_result_sidecar_records_exact_false_smoke_test(tmp_path) -> None:
    from src.models.tfidf_baseline import run_tfidf_baseline

    frame = _synthetic_frame().rename(columns={"label": "human_lmp_score"})
    frame["annotation_status"] = "human_single"
    frame["annotator_id"] = "annotator-1"
    frame["adjudication_notes"] = "verified"

    paths = run_tfidf_baseline(
        frame,
        load_model_config(),
        output_path=tmp_path / "tfidf-final.csv",
        label_column="human_lmp_score",
        run_mode="final",
        synthetic=False,
    )

    assert read_json(paths.metadata_json)["smoke_test"] is False


def test_smoke_result_bundle_is_label_free_and_has_clear_provenance(tmp_path) -> None:
    from src.models.tfidf_baseline import run_tfidf_baseline

    frame = _synthetic_frame()
    output = tmp_path / "smoke" / "tfidf_predictions.csv"

    paths = run_tfidf_baseline(
        frame,
        load_model_config(),
        output_path=output,
        label_column="label",
        run_mode="smoke_test",
        synthetic=True,
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
        "word_cosine_similarity",
        "char_cosine_similarity",
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
    assert not {"label", "human_lmp_score", "error_type", "decision"} & set(
        written.columns
    )
    assert metadata["run_mode"] == "smoke_test"
    assert metadata["synthetic"] is True
    assert metadata["research_use"] == "non-research"
    assert metadata["data_source"] == "synthetic"
    assert metadata["vocabulary_fit"]["splits"] == ["train"]
    assert metadata["calibration"]["fit_records"] == [
        {"pair_id": pair_id, "split": "train"}
        for pair_id in frame.loc[frame["split"] == "train", "pair_id"]
    ]
    assert "metrics" not in metadata
    assert metadata["predictions_sha256"] == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert len(metadata["input_data_sha256"]) == 64
    assert len(metadata["code_sha256"]) == 64
    assert {
        "generated_at_utc",
        "git_revision",
        "config_fingerprint",
        "project_version",
    } <= metadata["provenance"].keys()


def test_smoke_cli_uses_only_synthetic_data_and_does_not_modify_master(
    tmp_path, repo_root
) -> None:
    master = repo_root / "data" / "processed" / "lmp_pairs.csv"
    before_sha = hashlib.sha256(master.read_bytes()).hexdigest()
    output = tmp_path / "cli-smoke" / "tfidf_predictions.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.models.tfidf_baseline",
            "--config",
            str(repo_root / "configs" / "config.yaml"),
            "--smoke-test",
            "--output",
            str(output),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.exists()
    assert output.with_suffix(".metadata.json").exists()
    assert hashlib.sha256(master.read_bytes()).hexdigest() == before_sha
    written = read_csv(output)
    metadata = read_json(output.with_suffix(".metadata.json"))
    assert len(written) <= 6
    assert set(written["run_mode"]) == {"smoke_test"}
    assert set(written["data_source"]) == {"synthetic"}
    assert set(written["research_use"]) == {"non-research"}
    assert not {
        "label",
        "human_lmp_score",
        "error_type",
        "decision",
        "annotation_status",
    } & set(written.columns)
    assert metadata["synthetic"] is True
    assert metadata["research_use"] == "non-research"
    assert "metrics" not in metadata


def test_final_cli_is_blocked_by_current_gold_gate_before_output_creation(
    tmp_path, repo_root, unannotated_corpus_env
) -> None:
    output = tmp_path / "must-not-exist" / "tfidf_predictions.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.models.tfidf_baseline",
            "--config",
            str(unannotated_corpus_env["cfg_file"]),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode != 0
    assert "TODO_HUMAN_ANNOTATION" in completed.stdout + completed.stderr
    assert not output.exists()
    assert not output.with_suffix(".metadata.json").exists()
    assert not output.parent.exists()


def test_final_cli_succeeds_on_production_gold_dataset(
    tmp_path, repo_root
) -> None:
    """CLI execution on production gold corpus generates valid predictions."""
    output = tmp_path / "production_tfidf" / "tfidf_predictions.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.models.tfidf_baseline",
            "--config",
            str(repo_root / "configs" / "config.yaml"),
            "--output",
            str(output),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0
    assert output.exists()
    assert output.with_suffix(".metadata.json").exists()
    pred_df = pd.read_csv(output)
    assert len(pred_df) == 340
    assert "prediction" in pred_df.columns


def test_calibrated_predictions_are_explicitly_clipped_to_one_and_ten() -> None:
    from src.models.calibration import LinearScoreCalibrator
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame()
    model = TfidfBaseline(load_model_config()).fit(frame, label_column="label")
    assert model.calibrator is not None
    model.calibrator = LinearScoreCalibrator(
        slope=100.0,
        intercept=-50.0,
        fit_records=model.calibrator.fit_records,
    )

    predictions = model.predict(frame).set_index("pair_id")

    assert predictions.loc["T_IDENTICAL", "prediction"] == 10.0
    assert predictions.loc["T_DIFFERENT", "prediction"] == 1.0
    assert predictions["prediction"].between(1.0, 10.0).all()


@pytest.mark.parametrize("bad_id", [None, "", "   "])
def test_fit_rejects_malformed_pair_ids_through_foundation(bad_id) -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame()
    frame.loc[0, "pair_id"] = bad_id

    with pytest.raises(ValueError, match="null or blank pair_id"):
        TfidfBaseline(load_model_config()).fit(frame, label_column="label")


def test_fit_rejects_duplicate_pair_ids_through_foundation() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame()
    frame.loc[1, "pair_id"] = frame.loc[0, "pair_id"]

    with pytest.raises(ValueError, match="Duplicate pair_id"):
        TfidfBaseline(load_model_config()).fit(frame, label_column="label")


def test_fit_without_train_rows_has_an_explicit_error() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame().loc[lambda value: value["split"] != "train"].copy()

    with pytest.raises(ValueError, match="at least one train row"):
        TfidfBaseline(load_model_config()).fit(frame, label_column="label")


def test_predict_before_fit_has_an_explicit_error() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    with pytest.raises(RuntimeError, match="fitted before prediction"):
        TfidfBaseline(load_model_config()).predict(_synthetic_frame())


def test_fit_requires_metadata_needed_for_preserved_output() -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame().drop(columns=["language"])

    with pytest.raises(ValueError, match="missing columns.*language"):
        TfidfBaseline(load_model_config()).fit(frame, label_column="label")


@pytest.mark.parametrize("text_column", ["original_text", "simplified_text"])
@pytest.mark.parametrize("bad_text", [None, "", "   "])
def test_fit_rejects_null_or_blank_pair_text(text_column, bad_text) -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    frame = _synthetic_frame()
    frame.loc[0, text_column] = bad_text

    with pytest.raises(ValueError, match=rf"{text_column}.*null or blank"):
        TfidfBaseline(load_model_config()).fit(frame, label_column="label")


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("word_ngram_range", [0, 2], "positive integers"),
        ("char_ngram_range", [5, 3], "positive integers"),
        ("min_df", 0, "min_df"),
    ],
)
def test_invalid_tfidf_settings_have_explicit_errors(setting, value, message) -> None:
    from src.models.tfidf_baseline import TfidfBaseline

    cfg = load_model_config(
        overrides={"models": {"tfidf": {setting: value}}}
    )

    with pytest.raises(ValueError, match=message):
        TfidfBaseline(cfg)
