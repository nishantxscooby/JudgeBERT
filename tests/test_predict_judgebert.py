"""Download-free tests for JUDGEBERT checkpoint inference."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch import nn

from src.models.common import load_model_config


class RecordingTokenizer:
    """Tiny pair tokenizer that records exact batched text inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        originals, simplified = args
        rows = [
            [
                1,
                2 + len(original) % 11,
                2,
                14 + len(reformulation) % 11,
                2,
            ]
            for original, reformulation in zip(originals, simplified, strict=True)
        ]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 5), dtype=torch.long),
            "token_type_ids": torch.tensor(
                [[0, 0, 0, 1, 1]] * len(rows), dtype=torch.long
            ),
        }


class TinyEncoder(nn.Module):
    """Small initialized encoder compatible with JudgeBertModel."""

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)
        self.forward_batch_sizes: list[int] = []
        self.forward_grad_enabled: list[bool] = []
        self.forward_training: list[bool] = []

    def forward(self, **kwargs):
        self.forward_batch_sizes.append(int(kwargs["input_ids"].shape[0]))
        self.forward_grad_enabled.append(torch.is_grad_enabled())
        self.forward_training.append(self.training)
        return SimpleNamespace(
            last_hidden_state=self.embedding(kwargs["input_ids"]),
            attentions=None,
        )


def _config(*, eval_batch_size: int = 2):
    return load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {
                "epochs": 1,
                "eval_batch_size": eval_batch_size,
                "device": "cpu",
            },
        }
    )


def _model(cfg):
    from src.models.judgebert_model import JudgeBertModel

    model = JudgeBertModel.from_config(cfg, encoder=TinyEncoder())
    with torch.no_grad():
        model.regression_head[-1].weight.zero_()
        model.regression_head[-1].bias.fill_(0.5)
    return model


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": "P3",
                "language": "hi",
                "legal_domain": "contract",
                "source_group": "g3",
                "split": "test",
                "original_text": "  मूल पाठ तीन  ",
                "simplified_text": "  सरल पाठ तीन  ",
                "human_lmp_score": 1.0,
                "draft_lmp_score": 10.0,
                "decision": "Not Preserved",
                "error_type": "Omission",
                "annotation_status": "human_single",
            },
            {
                "pair_id": "P1",
                "language": "en",
                "legal_domain": "cyber",
                "source_group": "g1",
                "split": "train",
                "original_text": "Original one",
                "simplified_text": "Simple one",
                "human_lmp_score": 2.0,
                "draft_lmp_score": 9.0,
                "decision": "Partially Preserved",
                "error_type": "None",
                "annotation_status": "human_consensus",
            },
            {
                "pair_id": "P2",
                "language": "en",
                "legal_domain": "sale_of_goods",
                "source_group": "g2",
                "split": "validation",
                "original_text": "Original two",
                "simplified_text": "Simple two",
                "human_lmp_score": 3.0,
                "draft_lmp_score": 8.0,
                "decision": "Preserved",
                "error_type": "None",
                "annotation_status": "draft",
            },
        ]
    )


def test_predict_dataframe_batches_exact_pairs_preserves_order_and_drops_labels() -> None:
    from src.models.predict_judgebert import predict_judgebert

    cfg = _config(eval_batch_size=2)
    tokenizer = RecordingTokenizer()
    model = _model(cfg)
    frame = _prediction_frame()

    predictions = predict_judgebert(
        frame,
        cfg,
        tokenizer=tokenizer,
        model=model,
        device="cpu",
    )

    assert predictions.columns.tolist() == [
        "pair_id",
        "language",
        "legal_domain",
        "source_group",
        "split",
        "prediction",
    ]
    assert predictions["pair_id"].tolist() == ["P3", "P1", "P2"]
    assert predictions["language"].tolist() == ["hi", "en", "en"]
    assert predictions["split"].tolist() == ["test", "train", "validation"]
    assert predictions["prediction"].tolist() == [5.5, 5.5, 5.5]
    assert predictions["prediction"].between(1.0, 10.0).all()
    assert not {
        "human_lmp_score",
        "draft_lmp_score",
        "decision",
        "error_type",
        "annotation_status",
    } & set(predictions.columns)
    assert [call[0] for call in tokenizer.calls] == [
        (
            frame["original_text"].tolist()[:2],
            frame["simplified_text"].tolist()[:2],
        ),
        (
            frame["original_text"].tolist()[2:],
            frame["simplified_text"].tolist()[2:],
        ),
    ]
    assert model.encoder.forward_batch_sizes == [2, 1]
    assert model.encoder.forward_grad_enabled == [False, False]
    assert model.encoder.forward_training == [False, False]


@torch.no_grad()
def test_prediction_rejects_non_scalar_or_boolean_pair_ids_before_model_use() -> None:
    from src.models.predict_judgebert import predict_judgebert

    cfg = _config()
    tokenizer = RecordingTokenizer()
    model = _model(cfg)

    for bad_id in (False, ["P1"], {"id": "P1"}):
        frame = _prediction_frame().iloc[:1].copy()
        frame["pair_id"] = frame["pair_id"].astype(object)
        frame.at[0, "pair_id"] = bad_id
        try:
            predict_judgebert(
                frame,
                cfg,
                tokenizer=tokenizer,
                model=model,
                device="cpu",
            )
        except ValueError as exc:
            assert "invalid pair_id" in str(exc)
        else:  # pragma: no cover - assertion branch
            raise AssertionError(f"pair_id {bad_id!r} was accepted")

    assert tokenizer.calls == []
    assert model.encoder.forward_batch_sizes == []


def test_prediction_rejects_invalid_identity_and_text_before_model_use() -> None:
    from src.models.predict_judgebert import predict_judgebert

    cfg = _config()
    cases: list[tuple[pd.DataFrame, str]] = []

    missing_id = _prediction_frame().drop(columns=["pair_id"])
    cases.append((missing_id, "missing columns"))

    duplicate_id = _prediction_frame().copy()
    duplicate_id.loc[1, "pair_id"] = duplicate_id.loc[0, "pair_id"]
    cases.append((duplicate_id, "Duplicate pair_id"))

    blank_id = _prediction_frame().copy()
    blank_id.loc[0, "pair_id"] = "  "
    cases.append((blank_id, "null or blank pair_id"))

    blank_text = _prediction_frame().copy()
    blank_text.loc[0, "original_text"] = "  "
    cases.append((blank_text, "original_text"))

    non_string_text = _prediction_frame().copy()
    non_string_text["simplified_text"] = non_string_text["simplified_text"].astype(object)
    non_string_text.at[0, "simplified_text"] = 7
    cases.append((non_string_text, "simplified_text"))

    unknown_split = _prediction_frame().copy()
    unknown_split.loc[0, "split"] = "holdout"
    cases.append((unknown_split, "Unknown split"))

    cases.append((_prediction_frame().iloc[0:0], "at least one row"))

    for frame, message in cases:
        tokenizer = RecordingTokenizer()
        model = _model(cfg)
        with pytest.raises(ValueError, match=message):
            predict_judgebert(
                frame,
                cfg,
                tokenizer=tokenizer,
                model=model,
                device="cpu",
            )
        assert tokenizer.calls == []
        assert model.encoder.forward_batch_sizes == []


def _save_checkpoint(
    path,
    cfg,
    model,
    *,
    run_mode="smoke_test",
    synthetic=True,
):
    from src.models.train_judgebert import TrainingResult, save_judgebert_checkpoint

    if run_mode == "final":
        fit_frame = _prediction_frame().loc[
            lambda rows: rows["split"].isin(["train", "validation"])
        ].copy()
        fit_frame["annotation_status"] = "human_consensus"
    else:
        fit_frame = _prediction_frame().iloc[:2].copy()
    result = TrainingResult(
        history=[
            {
                "epoch": 1,
                "train_loss": 0.1,
                "validation_rmse": 0.2,
                "optimizer_steps": 1,
                "encoder_frozen": False,
            }
        ],
        best_epoch=1,
        best_validation_rmse=0.2,
        stopped_early=False,
        total_optimizer_steps=1,
        validation_predictions=pd.DataFrame(
            {"pair_id": ["P2"], "prediction": [5.5]}
        ),
        train_ids=["P1"],
        validation_ids=["P2"],
        fit_frame=fit_frame,
    )
    return save_judgebert_checkpoint(
        path,
        model,
        cfg,
        result,
        run_mode=run_mode,
        synthetic=synthetic,
    )


def _write_final_manifest(path, checkpoint_path, cfg):
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    payload = {
        "model_identity": "JUDGEBERT-inspired MuRIL cross-encoder",
        "backbone": cfg["models"]["judgebert"]["backbone"],
        "run_mode": "final",
        "smoke_test": False,
        "synthetic": False,
        "checkpoint_sha256": checkpoint_sha256,
        "hashes": {"checkpoint_sha256": checkpoint_sha256},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path, payload


def test_checkpoint_reload_is_safe_strict_and_prediction_identical(
    tmp_path, monkeypatch
) -> None:
    from src.models import train_judgebert
    from src.models.predict_judgebert import (
        load_judgebert_checkpoint,
        predict_judgebert,
        restore_judgebert_model,
    )

    cfg = _config()
    tokenizer = RecordingTokenizer()
    trained = _model(cfg)
    checkpoint_path = _save_checkpoint(tmp_path / "checkpoint.pt", cfg, trained)
    checkpoint_bytes = checkpoint_path.read_bytes()
    expected = predict_judgebert(
        _prediction_frame(),
        cfg,
        tokenizer=tokenizer,
        model=trained,
        device="cpu",
    )
    real_read = train_judgebert._read_checkpoint
    safe_load_calls = []

    def recording_read(source, *, map_location="cpu"):
        safe_load_calls.append((source.getvalue(), map_location))
        return real_read(source, map_location=map_location)

    monkeypatch.setattr(train_judgebert, "_read_checkpoint", recording_read)
    restored = _model(cfg)
    with torch.no_grad():
        restored.regression_head[-1].bias.zero_()

    checkpoint = load_judgebert_checkpoint(
        checkpoint_path,
        config=cfg,
        map_location="cpu",
    )
    restore_judgebert_model(checkpoint, restored)
    actual = predict_judgebert(
        _prediction_frame(),
        cfg,
        tokenizer=RecordingTokenizer(),
        model=restored,
        device="cpu",
    )

    assert safe_load_calls == [(checkpoint_bytes, "cpu")]
    assert checkpoint.format_version == 1
    assert checkpoint.model_identity == "JUDGEBERT-inspired MuRIL cross-encoder"
    assert checkpoint.backbone == "google/muril-base-cased"
    assert checkpoint.project_config["models"]["judgebert"] == cfg["models"]["judgebert"]
    assert torch.equal(
        torch.tensor(actual["prediction"].to_numpy()),
        torch.tensor(expected["prediction"].to_numpy()),
    )
    assert all(
        torch.equal(restored.state_dict()[name], value)
        for name, value in trained.state_dict().items()
    )


def test_checkpoint_loader_hashes_the_same_bytes_that_it_deserializes(
    tmp_path, monkeypatch
) -> None:
    from src.models import train_judgebert
    from src.models.predict_judgebert import load_judgebert_checkpoint

    cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    captured_bytes = checkpoint_path.read_bytes()
    captured_sha256 = hashlib.sha256(captured_bytes).hexdigest()

    replacement_model = _model(cfg)
    with torch.no_grad():
        replacement_model.regression_head[-1].bias.fill_(0.25)
    replacement_path = _save_checkpoint(
        tmp_path / "replacement.pt", cfg, replacement_model
    )
    replacement_bytes = replacement_path.read_bytes()
    assert replacement_bytes != captured_bytes

    real_read = train_judgebert._read_checkpoint

    def replace_after_deserialization(source, *, map_location="cpu"):
        payload = real_read(source, map_location=map_location)
        checkpoint_path.write_bytes(replacement_bytes)
        return payload

    monkeypatch.setattr(
        train_judgebert, "_read_checkpoint", replace_after_deserialization
    )

    checkpoint = load_judgebert_checkpoint(checkpoint_path)

    assert checkpoint.sha256 == captured_sha256
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != checkpoint.sha256


def test_direct_final_checkpoint_rejects_fit_ids_inconsistent_with_rows(
    tmp_path,
) -> None:
    from src.models.train_judgebert import (
        TrainingResult,
        save_judgebert_checkpoint,
    )

    cfg = _config()
    fit_frame = _prediction_frame().loc[
        lambda rows: rows["split"].isin(["train", "validation"])
    ].copy()
    fit_frame["annotation_status"] = "human_consensus"
    result = TrainingResult(
        history=[],
        best_epoch=1,
        best_validation_rmse=0.2,
        stopped_early=False,
        total_optimizer_steps=1,
        validation_predictions=pd.DataFrame(
            {"pair_id": ["P2"], "prediction": [5.5]}
        ),
        train_ids=["P2"],
        validation_ids=["P1"],
        fit_frame=fit_frame,
    )

    with pytest.raises(ValueError, match="fit IDs.*fit rows"):
        save_judgebert_checkpoint(
            tmp_path / "checkpoint.pt",
            _model(cfg),
            cfg,
            result,
            run_mode="final",
            synthetic=False,
        )


def test_direct_final_checkpoint_requires_verified_human_gold_fit_rows(
    tmp_path,
) -> None:
    from src.models.train_judgebert import (
        TrainingResult,
        save_judgebert_checkpoint,
    )

    cfg = _config()
    fit_frame = _prediction_frame().loc[
        lambda rows: rows["split"].isin(["train", "validation"])
    ].copy()
    result = TrainingResult(
        history=[],
        best_epoch=1,
        best_validation_rmse=0.2,
        stopped_early=False,
        total_optimizer_steps=1,
        validation_predictions=pd.DataFrame(
            {"pair_id": ["P2"], "prediction": [5.5]}
        ),
        train_ids=["P1"],
        validation_ids=["P2"],
        fit_frame=fit_frame,
    )

    with pytest.raises(ValueError, match="not human-annotated"):
        save_judgebert_checkpoint(
            tmp_path / "checkpoint.pt",
            _model(cfg),
            cfg,
            result,
            run_mode="final",
            synthetic=False,
        )


def test_direct_final_checkpoint_records_complete_fit_and_hash_lineage(
    tmp_path,
) -> None:
    from src.models.provenance import dataframe_sha256
    from src.models.train_judgebert import (
        FINAL_TRAINING_INPUT_COLUMNS,
        TrainingResult,
        save_judgebert_checkpoint,
    )
    from src.utils.config import config_fingerprint

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {"eval_batch_size": 2, "device": "cpu", "seed": 73},
        }
    )
    fit_frame = _prediction_frame().loc[
        lambda rows: rows["split"].isin(["train", "validation"])
    ].copy()
    fit_frame["annotation_status"] = "human_consensus"
    result = TrainingResult(
        history=[],
        best_epoch=1,
        best_validation_rmse=0.2,
        stopped_early=False,
        total_optimizer_steps=1,
        validation_predictions=pd.DataFrame(
            {"pair_id": ["P2"], "prediction": [5.5]}
        ),
        train_ids=["P1"],
        validation_ids=["P2"],
        fit_frame=fit_frame,
    )

    checkpoint_path = save_judgebert_checkpoint(
        tmp_path / "checkpoint.pt",
        _model(cfg),
        cfg,
        result,
        run_mode="final",
        synthetic=False,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    expected_fit = {
        "splits": ["train", "validation"],
        "ids_by_split": {"train": ["P1"], "validation": ["P2"]},
        "records": [
            {"pair_id": "P1", "split": "train"},
            {"pair_id": "P2", "split": "validation"},
        ],
    }
    assert payload["data_source"] == "human_gold_corpus"
    assert payload["smoke_test"] is False
    assert payload["research_metrics_reported"] is False
    assert payload["fit"] == expected_fit
    assert payload["fit_ids"] == expected_fit["ids_by_split"]
    assert payload["fit_splits"] == expected_fit["splits"]
    assert payload["config_fingerprint"] == config_fingerprint(cfg)
    assert payload["effective_seed"] == 73
    assert payload["input_data_columns"] == list(FINAL_TRAINING_INPUT_COLUMNS)
    assert payload["input_data_sha256"] == dataframe_sha256(
        fit_frame.loc[:, FINAL_TRAINING_INPUT_COLUMNS].reset_index(drop=True)
    )
    for field in ("config_sha256", "code_sha256", "input_data_sha256"):
        assert len(payload[field]) == 64
        int(payload[field], 16)
    assert payload["lineage"] == {
        "config_fingerprint": payload["config_fingerprint"],
        "config_sha256": payload["config_sha256"],
        "effective_seed": payload["effective_seed"],
        "code_sha256": payload["code_sha256"],
        "input_data_columns": list(FINAL_TRAINING_INPUT_COLUMNS),
        "input_data_sha256": payload["input_data_sha256"],
        "fit": expected_fit,
    }


def _gold_training_frame() -> pd.DataFrame:
    frame = _prediction_frame().loc[
        lambda rows: rows["split"].isin(["train", "validation"])
    ].copy()
    frame["annotation_status"] = "human_consensus"
    return frame.reset_index(drop=True)


def _run_configured_final_training(tmp_path, monkeypatch, cfg):
    from src.models import train_judgebert as training_module

    tokenizer = RecordingTokenizer()
    configured_model = _model(cfg)
    monkeypatch.setattr(
        training_module,
        "load_judgebert_tokenizer",
        lambda *args, **kwargs: tokenizer,
    )
    monkeypatch.setattr(
        training_module.JudgeBertModel,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: configured_model),
    )
    return training_module.run_judgebert_training(
        _gold_training_frame(),
        cfg,
        output_dir=tmp_path / "final-training",
        run_mode="final",
        synthetic=False,
        label_column="human_lmp_score",
        epochs=1,
        device="cpu",
    )


def test_final_training_manifest_records_complete_configured_lineage(
    tmp_path, monkeypatch
) -> None:
    from src.models.train_judgebert import FINAL_TRAINING_INPUT_COLUMNS
    from src.utils.config import config_fingerprint

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {
                "epochs": 1,
                "eval_batch_size": 2,
                "device": "cpu",
                "seed": 73,
            },
        }
    )
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        artifacts.checkpoint, map_location="cpu", weights_only=True
    )

    assert {
        "run_mode": manifest["run_mode"],
        "synthetic": manifest["synthetic"],
        "smoke_test": manifest["smoke_test"],
        "data_source": manifest["data_source"],
        "research_use": manifest["research_use"],
        "research_metrics_reported": manifest["research_metrics_reported"],
    } == {
        "run_mode": "final",
        "synthetic": False,
        "smoke_test": False,
        "data_source": "human_gold_corpus",
        "research_use": "research-training-only",
        "research_metrics_reported": False,
    }
    assert manifest["loading_policy"]["tokenizer_source"] == "configured_muril"
    assert manifest["loading_policy"]["model_source"] == "configured_muril"
    assert manifest["fit"] == checkpoint["fit"]
    assert manifest["fit_ids"] == checkpoint["fit_ids"]
    assert manifest["fit_splits"] == ["train", "validation"]
    assert manifest["input_data_columns"] == checkpoint["input_data_columns"] == list(
        FINAL_TRAINING_INPUT_COLUMNS
    )
    assert all(record["split"] != "test" for record in manifest["fit"]["records"])
    assert manifest["effective_seed"] == checkpoint["effective_seed"] == 73
    assert manifest["provenance"]["seed"] == 73
    assert manifest["provenance"]["config_fingerprint"] == config_fingerprint(
        checkpoint["project_config"]
    )
    assert manifest["artifacts"] == {
        "checkpoint": "checkpoint.pt",
        "effective_config": "effective_config.yaml",
        "predictions": "validation_predictions.csv",
    }
    expected_hashes = {
        "config_sha256": checkpoint["config_sha256"],
        "code_sha256": checkpoint["code_sha256"],
        "input_data_sha256": checkpoint["input_data_sha256"],
        "checkpoint_sha256": hashlib.sha256(
            artifacts.checkpoint.read_bytes()
        ).hexdigest(),
        "predictions_sha256": hashlib.sha256(
            artifacts.predictions.read_bytes()
        ).hexdigest(),
        "effective_config_sha256": hashlib.sha256(
            artifacts.effective_config.read_bytes()
        ).hexdigest(),
    }
    assert manifest["hashes"] == expected_hashes
    assert all(manifest[name] == digest for name, digest in expected_hashes.items())


def test_genuine_final_bundle_supports_configured_minimal_label_free_inference(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation.provenance import load_prediction_artifact
    from src.models import predict_judgebert

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    inference_frame = _prediction_frame()[
        ["pair_id", "original_text", "simplified_text"]
    ].copy()
    input_path = tmp_path / "minimal-input.csv"
    inference_frame.to_csv(input_path, index=False, lineterminator="\n")

    paths = predict_judgebert.run_judgebert_prediction(
        inference_frame,
        checkpoint_path=artifacts.checkpoint,
        checkpoint_manifest_path=artifacts.manifest,
        output_path=tmp_path / "final-predictions.csv",
        input_path=input_path,
        device="cpu",
    )

    loaded = load_prediction_artifact(paths.predictions_csv)
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    checkpoint_payload = torch.load(
        artifacts.checkpoint, map_location="cpu", weights_only=True
    )
    assert loaded.frame.columns.tolist() == [
        "pair_id",
        "prediction",
        "run_mode",
        "synthetic",
        "research_use",
    ]
    assert loaded.frame["pair_id"].tolist() == inference_frame["pair_id"].tolist()
    assert metadata["run_mode"] == "inference"
    assert metadata["smoke_test"] is False
    assert metadata["synthetic"] is False
    assert metadata["loading_policy"]["tokenizer_source"] == "configured_muril"
    assert metadata["loading_policy"]["model_source"] == "configured_muril"
    assert metadata["checkpoint_manifest_sha256"] == hashlib.sha256(
        artifacts.manifest.read_bytes()
    ).hexdigest()
    assert metadata["inference_config_identity"] == {
        "config_fingerprint": checkpoint_payload["config_fingerprint"],
        "config_sha256": checkpoint_payload["config_sha256"],
        "effective_seed": checkpoint_payload["effective_seed"],
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("config_fingerprint", "wrong-fingerprint"),
        ("config_sha256", "f" * 64),
        ("effective_seed", 999),
    ],
)
def test_final_prediction_loader_binds_complete_inference_config_identity(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    from src.evaluation.provenance import load_prediction_artifact
    from src.models import predict_judgebert

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    inference_frame = _prediction_frame()[
        ["pair_id", "original_text", "simplified_text"]
    ].copy()
    paths = predict_judgebert.run_judgebert_prediction(
        inference_frame,
        checkpoint_path=artifacts.checkpoint,
        checkpoint_manifest_path=artifacts.manifest,
        output_path=tmp_path / "final-predictions.csv",
        device="cpu",
    )
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    metadata["inference_config_identity"][field] = bad_value
    paths.metadata_json.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="inference config identity"):
        load_prediction_artifact(paths.predictions_csv)


@pytest.mark.parametrize("bad_version", [True, 1.0])
def test_final_prediction_loader_requires_exact_checkpoint_format_version(
    tmp_path, monkeypatch, bad_version
) -> None:
    from src.evaluation.provenance import load_prediction_artifact
    from src.models import predict_judgebert

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    inference_frame = _prediction_frame()[
        ["pair_id", "original_text", "simplified_text"]
    ].copy()
    paths = predict_judgebert.run_judgebert_prediction(
        inference_frame,
        checkpoint_path=artifacts.checkpoint,
        checkpoint_manifest_path=artifacts.manifest,
        output_path=tmp_path / "final-predictions.csv",
        device="cpu",
    )
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    metadata["checkpoint_format_version"] = bad_version
    paths.metadata_json.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="checkpoint_format_version.*integer"):
        load_prediction_artifact(paths.predictions_csv)


@pytest.mark.parametrize("bad_version", [True, 1.0])
def test_checkpoint_format_version_requires_exact_integer_type(
    tmp_path, bad_version
) -> None:
    from src.models.predict_judgebert import load_judgebert_checkpoint

    cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["format_version"] = bad_version
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="format_version.*exact integer 1"):
        load_judgebert_checkpoint(checkpoint_path)


@pytest.mark.parametrize(
    ("configured_seed", "bad_seed"),
    [(42, 42.0), (1, True)],
)
def test_checkpoint_effective_seed_requires_exact_integer_type(
    tmp_path, configured_seed, bad_seed
) -> None:
    from src.models.predict_judgebert import load_judgebert_checkpoint

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {
                "eval_batch_size": 2,
                "device": "cpu",
                "seed": configured_seed,
            },
        }
    )
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["effective_seed"] = bad_seed
    payload["lineage"]["effective_seed"] = bad_seed
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="effective_seed.*integer"):
        load_judgebert_checkpoint(checkpoint_path)


def test_checkpoint_rejects_inconsistent_smoke_and_synthetic_markers(
    tmp_path,
) -> None:
    from src.models.predict_judgebert import load_judgebert_checkpoint

    cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["run_mode"] = "final"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="run_mode and synthetic markers are inconsistent"):
        load_judgebert_checkpoint(checkpoint_path)


def test_missing_checkpoint_fails_without_creating_prediction_artifacts(tmp_path) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    output_path = tmp_path / "new" / "predictions.csv"
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=tmp_path / "missing.pt",
            output_path=output_path,
            config=_config(),
            tokenizer=RecordingTokenizer(),
            model=_model(_config()),
            device="cpu",
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()
    assert not output_path.parent.exists()


def test_final_checkpoint_requires_manifest_before_model_or_output(
    tmp_path, monkeypatch
) -> None:
    from src.models import predict_judgebert

    cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt",
        cfg,
        _model(cfg),
        run_mode="final",
        synthetic=False,
    )
    output_path = tmp_path / "new" / "predictions.csv"
    model_load_attempted = False

    def forbidden_tokenizer_load(*args, **kwargs):
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("model assets must not load before manifest validation")

    monkeypatch.setattr(
        predict_judgebert._training,
        "load_judgebert_tokenizer",
        forbidden_tokenizer_load,
    )

    with pytest.raises(FileNotFoundError, match="checkpoint manifest"):
        predict_judgebert.run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            config=cfg,
            device="cpu",
        )

    assert model_load_attempted is False
    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()
    assert not output_path.parent.exists()


def test_final_inference_rejects_minimal_self_authored_manifest(
    tmp_path,
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt",
        cfg,
        _model(cfg),
        run_mode="final",
        synthetic=False,
    )
    manifest_path, _ = _write_final_manifest(
        tmp_path / "manifest.json", checkpoint_path, cfg
    )
    output_path = tmp_path / "predictions.csv"

    with pytest.raises(ValueError, match="complete final training manifest"):
        run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=checkpoint_path,
            checkpoint_manifest_path=manifest_path,
            output_path=output_path,
            config=cfg,
            tokenizer=RecordingTokenizer(),
            model=_model(cfg),
            device="cpu",
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("data_source", "self-authored", "data_source"),
        ("research_use", "prediction-only", "research_use"),
        ("research_metrics_reported", True, "research_metrics_reported"),
        ("loading_policy.tokenizer_source", "injected_test_tokenizer", "tokenizer_source"),
        ("loading_policy.model_source", "injected_test_model", "model_source"),
    ],
)
def test_final_inference_rejects_nonproduction_training_manifest_flags(
    tmp_path, monkeypatch, field, bad_value, message
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    if field.startswith("loading_policy."):
        manifest["loading_policy"][field.split(".", 1)[1]] = bad_value
    else:
        manifest[field] = bad_value
    artifacts.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    output_path = tmp_path / "predictions.csv"

    with pytest.raises(ValueError, match=message):
        run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=artifacts.checkpoint,
            checkpoint_manifest_path=artifacts.manifest,
            output_path=output_path,
            config=cfg,
            tokenizer=RecordingTokenizer(),
            model=_model(cfg),
            device="cpu",
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()


@pytest.mark.parametrize(
    "field",
    ["effective_seed", "provenance.seed"],
)
def test_final_manifest_seed_fields_require_exact_integer_types(
    tmp_path, monkeypatch, field
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    if field == "provenance.seed":
        manifest["provenance"]["seed"] = float(manifest["provenance"]["seed"])
    else:
        manifest["effective_seed"] = float(manifest["effective_seed"])
    artifacts.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="seed.*integer"):
        run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=artifacts.checkpoint,
            checkpoint_manifest_path=artifacts.manifest,
            output_path=tmp_path / "predictions.csv",
            config=cfg,
            tokenizer=RecordingTokenizer(),
            model=_model(cfg),
            device="cpu",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "test_record",
        "fit_ids_by_split",
        "top_level_fit_ids",
        "fit_splits",
    ],
)
def test_final_inference_rejects_fit_lineage_not_matching_checkpoint(
    tmp_path, monkeypatch, mutation
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    if mutation == "test_record":
        manifest["fit"]["records"][0]["split"] = "test"
    elif mutation == "fit_ids_by_split":
        manifest["fit"]["ids_by_split"]["train"] = ["UNKNOWN"]
    elif mutation == "top_level_fit_ids":
        manifest["fit_ids"]["validation"] = ["UNKNOWN"]
    else:
        manifest["fit_splits"] = ["train", "validation", "test"]
    artifacts.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="fit"):
        run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=artifacts.checkpoint,
            checkpoint_manifest_path=artifacts.manifest,
            output_path=tmp_path / "predictions.csv",
            config=cfg,
            tokenizer=RecordingTokenizer(),
            model=_model(cfg),
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("provenance.config_fingerprint", "bad-fingerprint", "config_fingerprint"),
        ("provenance.seed", 999, "seed"),
        ("effective_seed", 999, "effective_seed"),
        ("config_sha256", "f" * 64, "config_sha256"),
        ("code_sha256", "f" * 64, "code_sha256"),
        ("input_data_sha256", "f" * 64, "input_data_sha256"),
        ("hashes.config_sha256", "f" * 64, "config_sha256"),
        ("hashes.code_sha256", "f" * 64, "code_sha256"),
        ("hashes.input_data_sha256", "f" * 64, "input_data_sha256"),
    ],
)
def test_final_inference_rejects_manifest_lineage_not_matching_checkpoint_config(
    tmp_path, monkeypatch, field, bad_value, message
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    section, _, key = field.partition(".")
    if key:
        manifest[section][key] = bad_value
    else:
        manifest[field] = bad_value
    artifacts.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match=message):
        run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=artifacts.checkpoint,
            checkpoint_manifest_path=artifacts.manifest,
            output_path=tmp_path / "predictions.csv",
            config=cfg,
            tokenizer=RecordingTokenizer(),
            model=_model(cfg),
            device="cpu",
        )


def test_final_inference_rejects_non_boolean_manifest_flags(
    tmp_path, monkeypatch
) -> None:
    from src.models import predict_judgebert

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    manifest["smoke_test"] = 0
    artifacts.manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        predict_judgebert._training,
        "load_judgebert_tokenizer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("model setup must not run")
        ),
    )
    output = tmp_path / "must-not-exist.csv"

    with pytest.raises(ValueError, match="smoke_test does not match the checkpoint"):
        predict_judgebert.run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=artifacts.checkpoint,
            checkpoint_manifest_path=artifacts.manifest,
            output_path=output,
            config=cfg,
            device="cpu",
        )

    assert not output.exists()
    assert not output.with_suffix(".metadata.json").exists()


def test_final_inference_cannot_overwrite_manifest_bound_training_predictions(
    tmp_path, monkeypatch
) -> None:
    from src.models import predict_judgebert

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    predictions_before = artifacts.predictions.read_bytes()
    manifest_before = artifacts.manifest.read_bytes()

    with pytest.raises(ValueError, match="manifest-bound"):
        predict_judgebert.run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=artifacts.checkpoint,
            checkpoint_manifest_path=artifacts.manifest,
            output_path=artifacts.predictions,
            device="cpu",
        )

    assert artifacts.predictions.read_bytes() == predictions_before
    assert artifacts.manifest.read_bytes() == manifest_before
    checkpoint = predict_judgebert.load_judgebert_checkpoint(artifacts.checkpoint)
    lineage = predict_judgebert._load_final_checkpoint_manifest(
        artifacts.manifest, checkpoint
    )
    assert lineage.training_predictions_path == artifacts.predictions.resolve()


@pytest.mark.parametrize(
    "artifact_name",
    ["checkpoint", "manifest", "effective_config", "predictions", "input"],
)
@pytest.mark.parametrize("destination_kind", ["predictions", "sidecar"])
def test_final_inference_rejects_hard_link_aliases_to_every_protected_artifact(
    tmp_path, monkeypatch, artifact_name, destination_kind
) -> None:
    import os

    from src.models import predict_judgebert

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    inference_frame = _prediction_frame()[
        ["pair_id", "original_text", "simplified_text"]
    ].copy()
    input_path = tmp_path / "inference-input.csv"
    inference_frame.to_csv(input_path, index=False, lineterminator="\n")
    protected = (
        input_path if artifact_name == "input" else getattr(artifacts, artifact_name)
    )
    protected_before = protected.read_bytes()
    output_path = tmp_path / "hard-link-output" / "predictions.csv"
    output_path.parent.mkdir(parents=True)
    alias_path = (
        output_path
        if destination_kind == "predictions"
        else output_path.with_suffix(".metadata.json")
    )
    os.link(protected, alias_path)

    with pytest.raises(ValueError, match="must not.*alias|must not overwrite"):
        predict_judgebert.run_judgebert_prediction(
            inference_frame,
            checkpoint_path=artifacts.checkpoint,
            checkpoint_manifest_path=artifacts.manifest,
            output_path=output_path,
            input_path=input_path,
            device="cpu",
        )

    assert protected.read_bytes() == protected_before


@pytest.mark.parametrize("destination_kind", ["predictions", "sidecar"])
def test_atomic_publication_defeats_hard_link_alias_created_after_validation(
    tmp_path, monkeypatch, destination_kind
) -> None:
    import os

    from src.models import predict_judgebert

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    inference_frame = _prediction_frame()[
        ["pair_id", "original_text", "simplified_text"]
    ].copy()
    protected = artifacts.effective_config
    protected_before = protected.read_bytes()
    output_path = tmp_path / "raced-output" / "predictions.csv"
    alias_path = (
        output_path
        if destination_kind == "predictions"
        else output_path.with_suffix(".metadata.json")
    )
    injected = False

    if destination_kind == "predictions":
        real_writer = predict_judgebert.io_utils.write_csv

        def racing_writer(payload, path, **kwargs):
            nonlocal injected
            if not injected:
                alias_path.parent.mkdir(parents=True, exist_ok=True)
                os.link(protected, alias_path)
                injected = True
            return real_writer(payload, path, **kwargs)

        monkeypatch.setattr(predict_judgebert.io_utils, "write_csv", racing_writer)
    else:
        real_writer = predict_judgebert.io_utils.write_json

        def racing_writer(payload, path, **kwargs):
            nonlocal injected
            if not injected:
                alias_path.parent.mkdir(parents=True, exist_ok=True)
                os.link(protected, alias_path)
                injected = True
            return real_writer(payload, path, **kwargs)

        monkeypatch.setattr(predict_judgebert.io_utils, "write_json", racing_writer)

    paths = predict_judgebert.run_judgebert_prediction(
        inference_frame,
        checkpoint_path=artifacts.checkpoint,
        checkpoint_manifest_path=artifacts.manifest,
        output_path=output_path,
        device="cpu",
    )

    assert injected is True
    assert protected.read_bytes() == protected_before
    assert not os.path.samefile(protected, alias_path)
    assert paths.predictions_csv.is_file()
    assert paths.metadata_json.is_file()


@pytest.mark.parametrize(
    ("artifact_name", "message"),
    [
        ("effective_config", "effective_config_sha256"),
        ("predictions", "predictions_sha256"),
    ],
)
def test_final_inference_rejects_tampered_training_artifacts(
    tmp_path, monkeypatch, artifact_name, message
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    artifact_path = getattr(artifacts, artifact_name)
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\ntampered")

    with pytest.raises(ValueError, match=message):
        run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=artifacts.checkpoint,
            checkpoint_manifest_path=artifacts.manifest,
            output_path=tmp_path / "predictions.csv",
            config=cfg,
            tokenizer=RecordingTokenizer(),
            model=_model(cfg),
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("model_identity", "wrong-model", "model_identity"),
        ("backbone", "wrong-backbone", "backbone"),
        ("run_mode", "smoke_test", "run_mode"),
        ("synthetic", True, "synthetic"),
        ("smoke_test", True, "smoke_test"),
        ("checkpoint_sha256", "f" * 64, "checkpoint_sha256"),
        ("hashes.checkpoint_sha256", "f" * 64, "checkpoint_sha256"),
    ],
)
def test_final_manifest_mismatch_fails_before_model_or_output(
    tmp_path, monkeypatch, field, bad_value, message
) -> None:
    from src.models import predict_judgebert

    cfg = _config()
    artifacts = _run_configured_final_training(tmp_path, monkeypatch, cfg)
    checkpoint_path = artifacts.checkpoint
    manifest_path = artifacts.manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "hashes.checkpoint_sha256":
        manifest["hashes"]["checkpoint_sha256"] = bad_value
    else:
        manifest[field] = bad_value
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    output_path = tmp_path / "new" / "predictions.csv"
    model_load_attempted = False

    def forbidden_tokenizer_load(*args, **kwargs):
        nonlocal model_load_attempted
        model_load_attempted = True
        raise AssertionError("model assets must not load after manifest failure")

    monkeypatch.setattr(
        predict_judgebert._training,
        "load_judgebert_tokenizer",
        forbidden_tokenizer_load,
    )

    with pytest.raises(ValueError, match=message):
        predict_judgebert.run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=checkpoint_path,
            checkpoint_manifest_path=manifest_path,
            output_path=output_path,
            config=cfg,
            device="cpu",
        )

    assert model_load_attempted is False
    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()
    assert not output_path.parent.exists()


def test_malformed_checkpoint_fails_without_creating_prediction_artifacts(
    tmp_path,
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    checkpoint_path = tmp_path / "malformed.pt"
    torch.save({"format_version": 1}, checkpoint_path)
    output_path = tmp_path / "new" / "predictions.csv"
    cfg = _config()

    with pytest.raises(ValueError, match="model_identity"):
        run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            config=cfg,
            tokenizer=RecordingTokenizer(),
            model=_model(cfg),
            device="cpu",
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()
    assert not output_path.parent.exists()


def test_divergent_explicit_seed_fails_before_model_or_output(
    tmp_path, monkeypatch
) -> None:
    from src.models import predict_judgebert

    checkpoint_cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", checkpoint_cfg, _model(checkpoint_cfg)
    )
    divergent_cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {
                "eval_batch_size": 2,
                "device": "cpu",
                "seed": 73,
            },
        }
    )
    output_path = tmp_path / "must-not-exist" / "predictions.csv"
    model_attempted = False

    def forbidden_tokenizer(*args, **kwargs):
        nonlocal model_attempted
        model_attempted = True
        raise AssertionError("model/network setup must not run")

    monkeypatch.setattr(
        predict_judgebert._training,
        "load_judgebert_tokenizer",
        forbidden_tokenizer,
    )

    with pytest.raises(ValueError, match="Explicit config.*checkpoint"):
        predict_judgebert.run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            config=divergent_cfg,
            device="cpu",
        )

    assert model_attempted is False
    assert not output_path.parent.exists()


def test_incompatible_config_fails_without_creating_prediction_artifacts(
    tmp_path,
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    checkpoint_cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", checkpoint_cfg, _model(checkpoint_cfg)
    )
    incompatible_cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0, "pooling": "mean"}},
            "training": {"device": "cpu"},
        }
    )
    output_path = tmp_path / "new" / "predictions.csv"

    with pytest.raises(ValueError, match="incompatible"):
        run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            config=incompatible_cfg,
            tokenizer=RecordingTokenizer(),
            model=_model(incompatible_cfg),
            device="cpu",
        )

    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()
    assert not output_path.parent.exists()


def test_unavailable_tokenizer_fails_without_fallback_or_output(
    tmp_path, monkeypatch
) -> None:
    from src.models import predict_judgebert

    cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    output_path = tmp_path / "new" / "predictions.csv"
    model_load_called = False

    def missing_tokenizer(*args, **kwargs):
        raise RuntimeError("exact MuRIL tokenizer is unavailable; no fallback")

    def forbidden_model_load(*args, **kwargs):
        nonlocal model_load_called
        model_load_called = True
        raise AssertionError("model must not load after tokenizer failure")

    monkeypatch.setattr(
        predict_judgebert._training,
        "load_judgebert_tokenizer",
        missing_tokenizer,
    )
    monkeypatch.setattr(
        predict_judgebert.JudgeBertModel,
        "from_config",
        forbidden_model_load,
    )

    with pytest.raises(RuntimeError, match="no fallback"):
        predict_judgebert.run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            config=cfg,
            device="cpu",
        )

    assert model_load_called is False
    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()
    assert not output_path.parent.exists()


def test_unavailable_model_fails_without_fallback_or_output(
    tmp_path, monkeypatch
) -> None:
    from src.models import predict_judgebert

    cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    output_path = tmp_path / "new" / "predictions.csv"
    tokenizer = RecordingTokenizer()

    monkeypatch.setattr(
        predict_judgebert._training,
        "load_judgebert_tokenizer",
        lambda *args, **kwargs: tokenizer,
    )

    def missing_model(*args, **kwargs):
        raise RuntimeError("exact MuRIL model is unavailable; no fallback")

    monkeypatch.setattr(
        predict_judgebert.JudgeBertModel,
        "from_config",
        missing_model,
    )

    with pytest.raises(RuntimeError, match="no fallback"):
        predict_judgebert.run_judgebert_prediction(
            _prediction_frame(),
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            config=cfg,
            device="cpu",
        )

    assert tokenizer.calls == []
    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()
    assert not output_path.parent.exists()


def test_prediction_rejects_caller_frame_that_differs_from_input_path_projection(
    tmp_path,
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    cfg = _config()
    caller_frame = _prediction_frame()
    source_frame = caller_frame.copy()
    source_frame.loc[0, "original_text"] = "Different authoritative source text"
    input_path = tmp_path / "pairs.csv"
    source_frame.to_csv(input_path, index=False, lineterminator="\n")
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    output_path = tmp_path / "predictions.csv"
    tokenizer = RecordingTokenizer()

    with pytest.raises(ValueError, match="caller frame.*input_path"):
        run_judgebert_prediction(
            caller_frame,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            input_path=input_path,
            config=cfg,
            tokenizer=tokenizer,
            model=_model(cfg),
            device="cpu",
        )

    assert tokenizer.calls == []
    assert not output_path.exists()
    assert not output_path.with_suffix(".metadata.json").exists()


def test_prediction_hashes_and_consumes_the_same_captured_input_bytes(
    tmp_path, monkeypatch
) -> None:
    from src.models import predict_judgebert as prediction_module

    cfg = _config()
    frame = _prediction_frame()
    input_path = tmp_path / "pairs.csv"
    frame.to_csv(input_path, index=False, lineterminator="\n")
    captured_bytes = input_path.read_bytes()
    replacement_frame = frame.copy()
    replacement_frame.loc[0, "original_text"] = "Replacement source text"
    replacement_path = tmp_path / "replacement.csv"
    replacement_frame.to_csv(replacement_path, index=False, lineterminator="\n")
    replacement_bytes = replacement_path.read_bytes()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    real_predict = prediction_module.predict_judgebert
    consumed_text: list[str] = []

    def replace_during_prediction(inference_frame, effective_cfg, **kwargs):
        consumed_text.extend(inference_frame["original_text"].tolist())
        input_path.write_bytes(replacement_bytes)
        return real_predict(inference_frame, effective_cfg, **kwargs)

    monkeypatch.setattr(
        prediction_module,
        "predict_judgebert",
        replace_during_prediction,
    )

    paths = prediction_module.run_judgebert_prediction(
        frame,
        checkpoint_path=checkpoint_path,
        output_path=tmp_path / "predictions.csv",
        input_path=input_path,
        config=cfg,
        tokenizer=RecordingTokenizer(),
        model=_model(cfg),
        device="cpu",
    )

    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert consumed_text == frame["original_text"].tolist()
    assert metadata["input_sha256"] == hashlib.sha256(captured_bytes).hexdigest()
    assert input_path.read_bytes() == replacement_bytes
    assert hashlib.sha256(replacement_bytes).hexdigest() != metadata["input_sha256"]


def test_prediction_artifact_is_label_free_hashed_and_provenance_complete(
    tmp_path, repo_root
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    cfg = _config(eval_batch_size=2)
    frame = _prediction_frame()
    input_path = tmp_path / "canonical_input.csv"
    frame.to_csv(input_path, index=False, lineterminator="\n")
    checkpoint_model = _model(cfg)
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, checkpoint_model
    )
    output_path = tmp_path / "predictions" / "judgebert.csv"
    corpus_path = repo_root / "data" / "processed" / "lmp_pairs.csv"
    corpus_before = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    paths = run_judgebert_prediction(
        frame,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        input_path=input_path,
        config=cfg,
        tokenizer=RecordingTokenizer(),
        model=_model(cfg),
        device="cpu",
    )

    assert paths.predictions_csv == output_path
    assert paths.metadata_json == output_path.with_suffix(".metadata.json")
    written = pd.read_csv(paths.predictions_csv)
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert written.columns.tolist() == [
        "pair_id",
        "language",
        "legal_domain",
        "source_group",
        "split",
        "prediction",
        "run_mode",
        "synthetic",
        "research_use",
    ]
    assert written["pair_id"].tolist() == frame["pair_id"].tolist()
    assert written["prediction"].between(1.0, 10.0).all()
    forbidden = {
        "human_lmp_score",
        "draft_lmp_score",
        "decision",
        "error_type",
        "annotation_status",
        "label",
    }
    assert forbidden.isdisjoint(written.columns)
    metadata_text = paths.metadata_json.read_text(encoding="utf-8")
    assert all(name not in metadata_text for name in forbidden)
    assert "metrics" not in metadata
    assert "final_metrics" not in metadata
    assert "test_metrics" not in metadata
    assert metadata["research_metrics_reported"] is False
    assert metadata["model_name"] == "judgebert"
    assert metadata["model_identity"] == "JUDGEBERT-inspired MuRIL cross-encoder"
    assert metadata["backbone"] == "google/muril-base-cased"
    assert metadata["run_mode"] == "smoke_test"
    assert metadata["checkpoint_run_mode"] == "smoke_test"
    assert metadata["checkpoint_synthetic"] is True
    assert metadata["score_range"] == [1.0, 10.0]
    assert metadata["pair_ids"] == frame["pair_id"].tolist()
    assert metadata["splits"] == ["test", "train", "validation"]
    assert metadata["records"] == [
        {"pair_id": row.pair_id, "split": row.split}
        for row in frame.itertuples(index=False)
    ]
    assert metadata["effective_config"]["judgebert"] == {
        "backbone": "google/muril-base-cased",
        "max_seq_length": 192,
        "dropout": 0.0,
        "head_hidden_size": 0,
        "pooling": "cls",
        "target_scaling": "minmax",
        "loss": "smooth_l1",
        "smooth_l1_beta": 0.5,
        "clip_predictions": True,
        "score_min": 1.0,
        "score_max": 10.0,
    }
    assert metadata["effective_config"]["eval_batch_size"] == 2
    assert metadata["effective_config"]["seed"] == 42
    assert metadata["device"] == "cpu"
    assert metadata["loading_policy"] == {
        "allow_download": False,
        "local_files_only": True,
        "trust_remote_code": False,
        "tokenizer_source": "injected_test_tokenizer",
        "model_source": "injected_test_model",
    }
    expected_hashes = {
        "checkpoint_sha256": checkpoint_path,
        "input_sha256": input_path,
        "output_sha256": output_path,
    }
    for key, path in expected_hashes.items():
        assert metadata[key] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(metadata["code_sha256"]) == 64
    int(metadata["code_sha256"], 16)
    assert metadata["provenance"]["config_fingerprint"]
    assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == corpus_before


def test_synthetic_checkpoint_marks_every_prediction_as_smoke_nonresearch(
    tmp_path,
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction

    cfg = _config()
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    output_path = tmp_path / "predictions.csv"

    paths = run_judgebert_prediction(
        _prediction_frame(),
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        config=cfg,
        tokenizer=RecordingTokenizer(),
        model=_model(cfg),
        device="cpu",
    )

    written = pd.read_csv(paths.predictions_csv)
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    assert written["run_mode"].eq("smoke_test").all()
    assert written["synthetic"].all()
    assert written["research_use"].eq("non-research").all()
    assert metadata["run_mode"] == "smoke_test"
    assert metadata["smoke_test"] is True
    assert metadata["synthetic"] is True
    assert metadata["research_use"] == "non-research"
    assert metadata["research_metrics_reported"] is False


def test_prediction_records_paired_text_projection_hash_without_target_columns(
    tmp_path,
) -> None:
    from src.models.predict_judgebert import run_judgebert_prediction
    from src.models.provenance import dataframe_sha256

    cfg = _config()
    frame = _prediction_frame().drop(
        columns=[
            "human_lmp_score",
            "draft_lmp_score",
            "decision",
            "error_type",
            "annotation_status",
        ]
    )
    input_path = tmp_path / "pairs.csv"
    frame.to_csv(input_path, index=False, lineterminator="\n")
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )

    paths = run_judgebert_prediction(
        frame,
        checkpoint_path=checkpoint_path,
        output_path=tmp_path / "predictions.csv",
        input_path=input_path,
        config=cfg,
        tokenizer=RecordingTokenizer(),
        model=_model(cfg),
        device="cpu",
    )

    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    projection_columns = [
        "pair_id",
        "original_text",
        "simplified_text",
        "language",
        "legal_domain",
        "source_group",
        "split",
    ]
    assert metadata["input_data_columns"] == projection_columns
    assert metadata["input_data_sha256"] == dataframe_sha256(
        frame[projection_columns]
    )
    assert metadata["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert metadata["input_hash_kind"] == "file_sha256"


def test_cli_with_injected_components_reads_canonical_csv_without_label_leakage(
    tmp_path, repo_root, capsys
) -> None:
    from src.models.predict_judgebert import main

    config_path = repo_root / "configs" / "config.yaml"
    cfg = load_model_config(str(config_path))
    input_path = tmp_path / "canonical.csv"
    _prediction_frame().to_csv(input_path, index=False, lineterminator="\n")
    checkpoint_path = _save_checkpoint(
        tmp_path / "checkpoint.pt", cfg, _model(cfg)
    )
    output_path = tmp_path / "cli" / "predictions.csv"
    corpus_path = repo_root / "data" / "processed" / "lmp_pairs.csv"
    corpus_before = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    return_code = main(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
            "--device",
            "cpu",
        ],
        tokenizer=RecordingTokenizer(),
        model=_model(cfg),
    )

    captured = capsys.readouterr()
    assert return_code == 0, captured.err
    assert "label-free JUDGEBERT predictions" in captured.out
    assert output_path.is_file()
    assert output_path.with_suffix(".metadata.json").is_file()
    written = pd.read_csv(output_path)
    assert written["pair_id"].tolist() == ["P3", "P1", "P2"]
    assert not {
        "human_lmp_score",
        "draft_lmp_score",
        "decision",
        "error_type",
        "annotation_status",
    } & set(written.columns)
    assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == corpus_before
