"""Offline unit tests for JUDGEBERT-inspired training and smoke artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import pandas as pd
from torch import nn
from torch.utils.data import RandomSampler, SequentialSampler


class RecordingTokenizer:
    """Tiny paired tokenizer that records the exact Hugging Face call contract."""

    def __init__(self, *, include_token_type_ids: bool = True) -> None:
        self.include_token_type_ids = include_token_type_ids
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, dict(kwargs)))
        originals, simplified = args
        rows = []
        for original, reformulation in zip(originals, simplified, strict=True):
            rows.append(
                [
                    1,
                    2 + len(str(original)) % 11,
                    2,
                    14 + len(str(reformulation)) % 11,
                    2,
                ]
            )
        result = {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 5), dtype=torch.long),
        }
        if self.include_token_type_ids:
            result["token_type_ids"] = torch.tensor(
                [[0, 0, 0, 1, 1]] * len(rows), dtype=torch.long
            )
        return result


class TinyEncoder(nn.Module):
    """Small trainable encoder satisfying the JudgeBertModel contract."""

    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)

    def forward(self, **kwargs):
        hidden = self.embedding(kwargs["input_ids"])
        return SimpleNamespace(last_hidden_state=hidden, attentions=None)


def _smoke_frame(*, include_test: bool = True) -> pd.DataFrame:
    rows = [
        ("TRAIN_1", "train", "Original train one", "Simple train one", 2.0),
        ("TRAIN_2", "train", "Original train two", "Simple train two", 4.0),
        ("TRAIN_3", "train", "Original train three", "Simple train three", 7.0),
        ("TRAIN_4", "train", "Original train four", "Simple train four", 9.0),
        ("VALID_1", "validation", "Original valid one", "Simple valid one", 3.0),
        ("VALID_2", "validation", "Original valid two", "Simple valid two", 8.0),
    ]
    if include_test:
        rows.append(("TEST_MUST_NOT_FIT", "test", "Secret test", "Secret simple", 6.0))
    return pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "language": "en",
                "legal_domain": "contract",
                "source_group": f"smoke::{pair_id}",
                "split": split,
                "original_text": original,
                "simplified_text": simplified,
                "label": label,
            }
            for pair_id, split, original, simplified, label in rows
        ]
    )


def _training_config(**overrides):
    from src.models.common import load_model_config

    training = {
        "epochs": 2,
        "batch_size": 2,
        "eval_batch_size": 2,
        "learning_rate": 0.02,
        "weight_decay": 0.01,
        "adam_epsilon": 1.0e-8,
        "warmup_ratio": 0.0,
        "max_grad_norm": 0.5,
        "scheduler": "linear",
        "gradient_accumulation_steps": 2,
        "freeze_encoder_epochs": 0,
        "early_stopping": {
            "enabled": False,
            "patience": 2,
            "metric": "val_rmse",
            "mode": "min",
            "min_delta": 0.0,
        },
    }
    training.update(overrides)
    return load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": training,
        }
    )


def _tiny_model(cfg):
    from src.models.judgebert_model import JudgeBertModel

    return JudgeBertModel.from_config(cfg, encoder=TinyEncoder())


def test_encode_text_pairs_uses_hf_pair_contract_and_optional_segments() -> None:
    from src.models.train_judgebert import encode_text_pairs

    originals = ["Original one", "मूल दो"]
    simplified = ["Simple one", "सरल दो"]
    tokenizer = RecordingTokenizer()

    encoded = encode_text_pairs(
        tokenizer,
        originals,
        simplified,
        max_length=19,
        padding="longest",
    )

    assert tokenizer.calls == [
        (
            (originals, simplified),
            {
                "truncation": "longest_first",
                "padding": "longest",
                "max_length": 19,
                "return_tensors": "pt",
            },
        )
    ]
    assert set(encoded) == {"input_ids", "attention_mask", "token_type_ids"}
    assert encoded["input_ids"].shape == (2, 5)

    no_segments = RecordingTokenizer(include_token_type_ids=False)
    encoded_without_segments = encode_text_pairs(
        no_segments,
        originals,
        simplified,
        max_length=19,
        padding=True,
    )
    assert set(encoded_without_segments) == {"input_ids", "attention_mask"}


def test_tokenizer_initialization_is_exactly_muril_local_only_and_restores_offline_state(
    monkeypatch,
) -> None:
    from src.models import train_judgebert
    from src.models.common import load_model_config

    calls = []
    observed_environment = []
    imported_constants = SimpleNamespace(HF_HUB_OFFLINE=True)
    tokenizer = RecordingTokenizer()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            sys.modules["huggingface_hub.constants"] = imported_constants
            observed_environment.append(
                (os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE"))
            )
            return tokenizer

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    monkeypatch.setattr(
        train_judgebert,
        "_load_auto_tokenizer_class",
        lambda: FakeAutoTokenizer,
        raising=False,
    )

    loaded = train_judgebert.load_judgebert_tokenizer(load_model_config())

    assert loaded is tokenizer
    assert calls == [
        (
            "google/muril-base-cased",
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]
    assert observed_environment == [("1", "1")]
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert imported_constants.HF_HUB_OFFLINE is False


def test_missing_tokenizer_fails_with_identity_and_never_falls_back(monkeypatch) -> None:
    from src.models import train_judgebert
    from src.models.common import load_model_config
    from src.models.judgebert_model import MODEL_IDENTITY

    calls = []

    class MissingAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            raise OSError("tokenizer not cached")

    monkeypatch.setattr(
        train_judgebert,
        "_load_auto_tokenizer_class",
        lambda: MissingAutoTokenizer,
    )

    with pytest.raises(RuntimeError) as caught:
        train_judgebert.load_judgebert_tokenizer(load_model_config())

    message = str(caught.value)
    assert MODEL_IDENTITY in message
    assert "tokenizer 'google/muril-base-cased'" in message
    assert "local_files_only=True" in message
    assert "No fallback tokenizer will be used" in message
    assert "--allow-download" in message
    assert calls == [
        (
            "google/muril-base-cased",
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]


def test_training_dataloaders_shuffle_only_train_use_zero_workers_and_exclude_test() -> None:
    from src.models.train_judgebert import build_training_dataloaders

    frame = _smoke_frame()
    tokenizer = RecordingTokenizer()
    cfg = _training_config()

    data = build_training_dataloaders(
        frame,
        cfg,
        tokenizer,
        label_column="label",
        seed=123,
    )

    assert isinstance(data.train_loader.sampler, RandomSampler)
    assert isinstance(data.validation_loader.sampler, SequentialSampler)
    assert data.train_loader.num_workers == 0
    assert data.validation_loader.num_workers == 0
    assert data.train_ids == ["TRAIN_1", "TRAIN_2", "TRAIN_3", "TRAIN_4"]
    assert data.validation_ids == ["VALID_1", "VALID_2"]
    encoded_texts = [text for call, _ in tokenizer.calls for text in call[0]]
    assert "Secret test" not in encoded_texts
    assert "TEST_MUST_NOT_FIT" not in data.train_ids + data.validation_ids


def test_one_epoch_training_is_finite_and_accumulates_then_clips_gradients(
    monkeypatch,
) -> None:
    from src.models.train_judgebert import train_judgebert

    cfg = _training_config(epochs=1, batch_size=1, gradient_accumulation_steps=2)
    model = _tiny_model(cfg)
    tokenizer = RecordingTokenizer()
    clip_calls = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        materialized = list(parameters)
        clip_calls.append((len(materialized), float(max_norm)))
        return real_clip(materialized, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)

    result = train_judgebert(
        _smoke_frame(),
        cfg,
        tokenizer=tokenizer,
        model=model,
        label_column="label",
        epochs=1,
        device="cpu",
    )

    assert result.total_optimizer_steps == 2
    assert clip_calls == [(3, 0.5), (3, 0.5)]
    assert len(result.history) == 1
    assert result.history[0]["optimizer_steps"] == 2
    assert torch.isfinite(torch.tensor(result.history[0]["train_loss"]))
    assert torch.isfinite(torch.tensor(result.history[0]["validation_rmse"]))
    scores = torch.tensor(result.validation_predictions["prediction"].tolist())
    assert torch.isfinite(scores).all()
    assert ((scores >= 1.0) & (scores <= 10.0)).all()
    assert result.train_ids == ["TRAIN_1", "TRAIN_2", "TRAIN_3", "TRAIN_4"]
    assert result.validation_ids == ["VALID_1", "VALID_2"]


def test_encoder_is_frozen_for_configured_first_epoch_then_unfrozen() -> None:
    from src.models.train_judgebert import train_judgebert

    cfg = _training_config(
        epochs=2,
        batch_size=4,
        gradient_accumulation_steps=1,
        freeze_encoder_epochs=1,
    )
    model = _tiny_model(cfg)

    result = train_judgebert(
        _smoke_frame(),
        cfg,
        tokenizer=RecordingTokenizer(),
        model=model,
        label_column="label",
        device="cpu",
    )

    assert [row["encoder_frozen"] for row in result.history] == [True, False]
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())


def test_early_stopping_restores_best_validation_state(monkeypatch) -> None:
    from src.models import train_judgebert as training_module

    early = {
        "enabled": True,
        "patience": 1,
        "metric": "val_rmse",
        "mode": "min",
        "min_delta": 0.0,
    }
    cfg = _training_config(
        epochs=5,
        batch_size=4,
        gradient_accumulation_steps=1,
        early_stopping=early,
    )
    model = _tiny_model(cfg)
    best_state = {}
    evaluation_count = 0

    def controlled_evaluation(candidate, loader, device):
        nonlocal evaluation_count, best_state
        current = {
            name: value.detach().cpu().clone()
            for name, value in candidate.state_dict().items()
        }
        if evaluation_count == 0:
            best_state = current
            answer = (2.0, [3.0, 8.0])
        elif evaluation_count == 1:
            assert any(
                not torch.equal(current[name], best_state[name]) for name in current
            )
            answer = (3.0, [4.0, 9.0])
        else:
            assert all(
                torch.equal(current[name], best_state[name]) for name in current
            )
            answer = (2.0, [3.0, 8.0])
        evaluation_count += 1
        return answer

    monkeypatch.setattr(training_module, "_evaluate_model", controlled_evaluation)

    result = training_module.train_judgebert(
        _smoke_frame(),
        cfg,
        tokenizer=RecordingTokenizer(),
        model=model,
        label_column="label",
        device="cpu",
    )

    assert result.stopped_early is True
    assert result.best_epoch == 1
    assert result.best_validation_rmse == 2.0
    assert len(result.history) == 2
    assert evaluation_count == 3
    assert all(
        torch.equal(model.state_dict()[name].cpu(), value)
        for name, value in best_state.items()
    )


def test_optimizer_and_scheduler_use_effective_configuration(monkeypatch) -> None:
    from src.models import train_judgebert as training_module

    cfg = _training_config(
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=0.0123,
        weight_decay=0.045,
        adam_epsilon=2.0e-7,
        scheduler="cosine",
        warmup_ratio=0.25,
    )
    optimizer_calls = []
    scheduler_calls = []
    real_adamw = torch.optim.AdamW
    real_scheduler = training_module._build_scheduler

    def recording_adamw(parameters, **kwargs):
        optimizer_calls.append(dict(kwargs))
        return real_adamw(parameters, **kwargs)

    def recording_scheduler(optimizer, **kwargs):
        scheduler_calls.append(dict(kwargs))
        return real_scheduler(optimizer, **kwargs)

    monkeypatch.setattr(torch.optim, "AdamW", recording_adamw)
    monkeypatch.setattr(training_module, "_build_scheduler", recording_scheduler)

    training_module.train_judgebert(
        _smoke_frame(),
        cfg,
        tokenizer=RecordingTokenizer(),
        model=_tiny_model(cfg),
        label_column="label",
        device="cpu",
    )

    assert optimizer_calls == [
        {"lr": 0.0123, "weight_decay": 0.045, "eps": 2.0e-7}
    ]
    assert scheduler_calls == [
        {"name": "cosine", "total_steps": 2, "warmup_ratio": 0.25}
    ]


def test_checkpoint_load_into_injected_compatible_model_has_prediction_parity(
    tmp_path,
) -> None:
    from src.models.train_judgebert import (
        encode_text_pairs,
        load_checkpoint_into_model,
        save_judgebert_checkpoint,
        train_judgebert,
    )

    cfg = _training_config(epochs=1, batch_size=4, gradient_accumulation_steps=1)
    tokenizer = RecordingTokenizer()
    trained = _tiny_model(cfg)
    result = train_judgebert(
        _smoke_frame(),
        cfg,
        tokenizer=tokenizer,
        model=trained,
        label_column="label",
        device="cpu",
    )
    checkpoint = tmp_path / "checkpoint.pt"

    save_judgebert_checkpoint(
        checkpoint,
        trained,
        cfg,
        result,
        run_mode="smoke_test",
        synthetic=True,
    )
    restored = _tiny_model(cfg)
    payload = load_checkpoint_into_model(checkpoint, restored, map_location="cpu")

    encoded = encode_text_pairs(
        tokenizer,
        ["Parity original"],
        ["Parity simplified"],
        max_length=cfg["models"]["judgebert"]["max_seq_length"],
    )
    expected = trained.predict_scores(**encoded)
    actual = restored.predict_scores(**encoded)

    assert torch.equal(actual, expected)
    assert payload["model_identity"] == "JUDGEBERT-inspired MuRIL cross-encoder"
    assert payload["backbone"] == "google/muril-base-cased"
    assert payload["run_mode"] == "smoke_test"
    assert payload["synthetic"] is True
    assert payload["project_config"]["training"]["epochs"] == 1
    assert payload["best_epoch"] == result.best_epoch
    assert all(row["research_use"] == "non-research" for row in payload["history"])


def test_synthetic_smoke_fixture_uses_only_generic_labels_and_honors_train_limit() -> None:
    from src.models.train_judgebert import build_synthetic_smoke_fixture

    frame = build_synthetic_smoke_fixture(max_train_rows=3)

    assert frame["split"].value_counts().to_dict() == {"train": 3, "validation": 2}
    assert "label" in frame.columns
    forbidden = {
        "human_lmp_score",
        "draft_lmp_score",
        "annotation_status",
        "decision",
        "error_type",
    }
    assert forbidden.isdisjoint(frame.columns)
    assert frame["pair_id"].is_unique
    assert frame["label"].between(1.0, 10.0).all()


def test_smoke_run_writes_hashed_label_free_nonresearch_artifacts_without_touching_corpus(
    tmp_path,
    repo_root,
) -> None:
    from src.models.train_judgebert import run_judgebert_training

    cfg = _training_config(epochs=1, batch_size=4, gradient_accumulation_steps=1)
    frame = _smoke_frame()
    output_dir = tmp_path / "judgebert-smoke"
    corpus_path = repo_root / "data" / "processed" / "lmp_pairs.csv"
    corpus_before = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    artifacts = run_judgebert_training(
        frame,
        cfg,
        output_dir=output_dir,
        run_mode="smoke_test",
        synthetic=True,
        label_column="label",
        tokenizer=RecordingTokenizer(),
        model=_tiny_model(cfg),
        epochs=1,
        device="cpu",
    )

    corpus_after = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    assert corpus_after == corpus_before
    assert artifacts.checkpoint == output_dir / "checkpoint.pt"
    assert artifacts.predictions == output_dir / "smoke_predictions.csv"
    assert artifacts.manifest == output_dir / "manifest.json"
    assert artifacts.effective_config == output_dir / "effective_config.yaml"
    assert all(path.is_file() for path in artifacts)

    predictions = pd.read_csv(artifacts.predictions)
    assert predictions.columns.tolist() == [
        "pair_id",
        "split",
        "prediction",
        "run_mode",
        "synthetic",
        "research_use",
    ]
    assert not any("label" in column.casefold() for column in predictions.columns)
    assert set(predictions["pair_id"]) == {
        "TRAIN_1",
        "TRAIN_2",
        "TRAIN_3",
        "TRAIN_4",
        "VALID_1",
        "VALID_2",
    }
    assert "TEST_MUST_NOT_FIT" not in set(predictions["pair_id"])
    assert predictions["prediction"].between(1.0, 10.0).all()
    assert predictions["run_mode"].eq("smoke_test").all()
    assert predictions["synthetic"].all()
    assert predictions["research_use"].eq("non-research").all()

    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    assert manifest["model_identity"] == "JUDGEBERT-inspired MuRIL cross-encoder"
    assert manifest["backbone"] == "google/muril-base-cased"
    assert manifest["run_mode"] == "smoke_test"
    assert manifest["smoke_test"] is True
    assert manifest["synthetic"] is True
    assert manifest["research_use"] == "non-research"
    assert manifest["research_metrics_reported"] is False
    assert manifest["fit"]["splits"] == ["train", "validation"]
    assert manifest["fit"]["ids_by_split"] == {
        "train": ["TRAIN_1", "TRAIN_2", "TRAIN_3", "TRAIN_4"],
        "validation": ["VALID_1", "VALID_2"],
    }
    assert [record["split"] for record in manifest["fit"]["records"]] == [
        "train",
        "train",
        "train",
        "train",
        "validation",
        "validation",
    ]
    assert "test" not in manifest["fit"]["splits"]
    assert "TEST_MUST_NOT_FIT" not in artifacts.manifest.read_text(encoding="utf-8")
    assert manifest["best_epoch"] >= 1
    assert all(row["research_use"] == "non-research" for row in manifest["history"])

    expected_hashes = {
        "code_sha256": None,
        "config_sha256": None,
        "input_data_sha256": None,
        "checkpoint_sha256": artifacts.checkpoint,
        "predictions_sha256": artifacts.predictions,
        "effective_config_sha256": artifacts.effective_config,
    }
    assert set(manifest["hashes"]) == set(expected_hashes)
    assert manifest["fit_ids"] == manifest["fit"]["ids_by_split"]
    assert manifest["fit_splits"] == ["train", "validation"]
    for name, path in expected_hashes.items():
        digest = manifest["hashes"][name]
        assert manifest[name] == digest
        assert len(digest) == 64
        int(digest, 16)
        if path is not None:
            assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "final_metrics" not in manifest
    assert "test_metrics" not in manifest


def test_training_manifest_hashes_the_checkpoint_bytes_actually_restored(
    tmp_path, monkeypatch
) -> None:
    from src.models import train_judgebert as training_module

    cfg = _training_config(epochs=1, batch_size=4, gradient_accumulation_steps=1)
    output_dir = tmp_path / "checkpoint-toctou"
    captured: list[bytes] = []
    replacement = b"replacement-after-deserialization"
    real_read_checkpoint = training_module._read_checkpoint

    def replace_after_deserialization(source, *, map_location="cpu"):
        if hasattr(source, "getvalue"):
            captured.append(source.getvalue())
        else:
            captured.append(source.read_bytes())
        payload = real_read_checkpoint(source, map_location=map_location)
        (output_dir / "checkpoint.pt").write_bytes(replacement)
        return payload

    monkeypatch.setattr(
        training_module,
        "_read_checkpoint",
        replace_after_deserialization,
    )

    artifacts = training_module.run_judgebert_training(
        _smoke_frame(),
        cfg,
        output_dir=output_dir,
        run_mode="smoke_test",
        synthetic=True,
        label_column="label",
        tokenizer=RecordingTokenizer(),
        model=_tiny_model(cfg),
        epochs=1,
        device="cpu",
    )

    manifest = json.loads(artifacts.manifest.read_text(encoding="utf-8"))
    restored_digest = hashlib.sha256(captured[0]).hexdigest()
    assert manifest["checkpoint_sha256"] == restored_digest
    assert manifest["hashes"]["checkpoint_sha256"] == restored_digest
    assert artifacts.checkpoint.read_bytes() == replacement
    assert restored_digest != hashlib.sha256(replacement).hexdigest()


def test_missing_production_backbone_is_explicit_and_creates_no_output(
    tmp_path,
    monkeypatch,
) -> None:
    from src.models import judgebert_model
    from src.models.train_judgebert import (
        build_synthetic_smoke_fixture,
        run_judgebert_training,
    )

    class MissingAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            raise OSError("MuRIL weights absent")

    monkeypatch.setattr(
        judgebert_model,
        "_load_auto_model_class",
        lambda: MissingAutoModel,
    )
    output_dir = tmp_path / "must-not-exist"
    cfg = _training_config(epochs=1)

    with pytest.raises(RuntimeError) as caught:
        run_judgebert_training(
            build_synthetic_smoke_fixture(max_train_rows=2),
            cfg,
            output_dir=output_dir,
            run_mode="smoke_test",
            synthetic=True,
            label_column="label",
            tokenizer=RecordingTokenizer(),
            epochs=1,
            device="cpu",
        )

    message = str(caught.value)
    assert "JUDGEBERT-inspired MuRIL cross-encoder backbone" in message
    assert "'google/muril-base-cased'" in message
    assert "local_files_only=True" in message
    assert "No fallback model will be used" in message
    assert not output_dir.exists()


def test_final_cli_calls_authoritative_gold_loader_before_assets_or_output(
    tmp_path,
    repo_root,
    monkeypatch,
    capsys,
) -> None:
    from src.models import train_judgebert

    events = []
    config_path = repo_root / "configs" / "config.yaml"
    output_dir = tmp_path / "blocked-final"

    def blocked_gold_loader(*args, **kwargs):
        events.append(("gold_gate", args, kwargs))
        raise ValueError("TODO_HUMAN_ANNOTATION — gold data unavailable")

    def forbidden_tokenizer(*args, **kwargs):
        events.append(("tokenizer", args, kwargs))
        raise AssertionError("tokenizer must not load before the gold gate")

    monkeypatch.setattr(train_judgebert, "load_model_corpus", blocked_gold_loader, raising=False)
    monkeypatch.setattr(train_judgebert, "load_judgebert_tokenizer", forbidden_tokenizer)

    return_code = train_judgebert.main(
        ["--config", str(config_path), "--output-dir", str(output_dir)]
    )

    captured = capsys.readouterr()
    assert return_code == 1
    assert events == [
        (
            "gold_gate",
            (),
            {"config_path": str(config_path), "mode": "final"},
        )
    ]
    assert "TODO_HUMAN_ANNOTATION" in captured.err
    assert not output_dir.exists()


def test_smoke_cli_supports_one_epoch_eight_rows_with_injected_tiny_components(
    tmp_path,
    repo_root,
    monkeypatch,
    capsys,
) -> None:
    from src.models import train_judgebert
    from src.models.common import load_model_config

    config_path = repo_root / "configs" / "config.yaml"
    output_dir = tmp_path / "cli-smoke"
    cfg = load_model_config(str(config_path))

    def forbidden_corpus_load(*args, **kwargs):
        raise AssertionError("smoke CLI must not read the canonical corpus")

    monkeypatch.setattr(train_judgebert, "load_model_corpus", forbidden_corpus_load)

    return_code = train_judgebert.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--smoke-test",
            "--epochs",
            "1",
            "--max-train-rows",
            "8",
        ],
        tokenizer=RecordingTokenizer(),
        model=_tiny_model(cfg),
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert "synthetic non-research smoke checkpoint" in captured.out
    assert (output_dir / "checkpoint.pt").is_file()
    predictions = pd.read_csv(output_dir / "smoke_predictions.csv")
    assert len(predictions) == 10
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["fit"]["ids_by_split"]["train"]) == 8
    assert manifest["run_mode"] == "smoke_test"


def test_missing_production_tokenizer_is_explicit_and_creates_no_output(
    tmp_path,
    monkeypatch,
) -> None:
    from src.models import train_judgebert

    class MissingAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            raise OSError("MuRIL tokenizer absent")

    monkeypatch.setattr(
        train_judgebert,
        "_load_auto_tokenizer_class",
        lambda: MissingAutoTokenizer,
    )
    cfg = _training_config(epochs=1)
    output_dir = tmp_path / "missing-tokenizer-output"

    with pytest.raises(RuntimeError) as caught:
        train_judgebert.run_judgebert_training(
            train_judgebert.build_synthetic_smoke_fixture(max_train_rows=2),
            cfg,
            output_dir=output_dir,
            run_mode="smoke_test",
            synthetic=True,
            label_column="label",
            epochs=1,
            device="cpu",
        )

    message = str(caught.value)
    assert "JUDGEBERT-inspired MuRIL cross-encoder tokenizer" in message
    assert "'google/muril-base-cased'" in message
    assert "local_files_only=True" in message
    assert "No fallback tokenizer will be used" in message
    assert not output_dir.exists()


def test_allow_download_is_explicitly_forwarded_to_exact_tokenizer(monkeypatch) -> None:
    from src.models import train_judgebert

    calls = []
    tokenizer = RecordingTokenizer()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            return tokenizer

    monkeypatch.setattr(
        train_judgebert,
        "_load_auto_tokenizer_class",
        lambda: FakeAutoTokenizer,
    )

    loaded = train_judgebert.load_judgebert_tokenizer(
        _training_config(), allow_download=True
    )

    assert loaded is tokenizer
    assert calls == [
        (
            "google/muril-base-cased",
            {"local_files_only": False, "trust_remote_code": False},
        )
    ]
