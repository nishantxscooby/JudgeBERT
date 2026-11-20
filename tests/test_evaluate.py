"""Contract and integration tests for final evaluation reporting."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.utils.config import config_fingerprint


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pair_digest(pair_ids) -> str:
    payload = json.dumps(list(pair_ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config() -> dict:
    return {
        "project": {"name": "test", "version": "1.0", "seed": 17},
        "dataset": {
            "score_min": 1,
            "score_max": 10,
            "gold_status_values": [
                "human_single",
                "human_multi_pending",
                "human_consensus",
            ],
        },
        "decision": {"partial_min": 5.0, "preserved_min": 8.0},
        "evaluation": {
            "metrics": ["pearson", "spearman", "rmse", "mae", "r2"],
            "group_by": ["language", "legal_domain"],
            "report_decision_metrics": True,
            "bootstrap": {"enabled": True, "n_resamples": 20, "confidence": 0.90},
        },
    }


def _gold() -> pd.DataFrame:
    splits = ["train"] * 3 + ["validation"] * 3 + ["test"] * 3
    return pd.DataFrame(
        {
            "pair_id": [f"P{i}" for i in range(1, 10)],
            "original_text": [f"Original canonical text P{i}" for i in range(1, 10)],
            "simplified_text": [
                f"Simplified canonical text P{i}" for i in range(1, 10)
            ],
            "language": ["en", "hi", "en"] * 3,
            "legal_domain": ["contract", "cyber", "sale_of_goods"] * 3,
            "source_group": [f"g{i}" for i in range(1, 10)],
            "split": splits,
            "human_lmp_score": [1.0, 5.0, 9.0, 2.0, 6.0, 9.0, 3.0, 6.0, 10.0],
            "decision": [
                "Not Preserved",
                "Partially Preserved",
                "Preserved",
            ]
            * 3,
            "annotation_status": ["human_consensus"] * 9,
        }
    )


class _ConfiguredSentenceEncoder:
    def __init__(self) -> None:
        self.max_seq_length = None

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


class _ConfiguredPairTokenizer:
    def __call__(self, originals, simplified, **kwargs):
        rows = [
            [1, 2 + len(original) % 11, 2, 14 + len(rewrite) % 11, 2]
            for original, rewrite in zip(originals, simplified, strict=True)
        ]
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.ones((len(rows), 5), dtype=torch.long),
            "token_type_ids": torch.tensor(
                [[0, 0, 0, 1, 1]] * len(rows), dtype=torch.long
            ),
        }


class _ConfiguredTinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = nn.Embedding(32, 4)

    def forward(self, **kwargs):
        return SimpleNamespace(
            last_hidden_state=self.embedding(kwargs["input_ids"]),
            attentions=None,
        )


def _judgebert_fixture_model(cfg):
    from src.models.judgebert_model import JudgeBertConfig, JudgeBertModel

    model = JudgeBertModel(
        JudgeBertConfig.from_project_config(cfg),
        encoder=_ConfiguredTinyEncoder(),
    )
    with torch.no_grad():
        model.regression_head[-1].weight.zero_()
        model.regression_head[-1].bias.fill_(0.5)
    return model


def _write_final_checkpoint_bundle(tmp_path, cfg, frame, monkeypatch):
    from src.models import train_judgebert

    configured_model = _judgebert_fixture_model(cfg)
    monkeypatch.setattr(
        train_judgebert,
        "load_judgebert_tokenizer",
        lambda *args, **kwargs: _ConfiguredPairTokenizer(),
    )
    monkeypatch.setattr(
        train_judgebert.JudgeBertModel,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: configured_model),
    )
    artifacts = train_judgebert.run_judgebert_training(
        frame.loc[frame["split"] != "test"].copy(),
        cfg,
        output_dir=tmp_path / "judgebert-training",
        run_mode="final",
        synthetic=False,
        label_column="human_lmp_score",
        epochs=1,
        device="cpu",
    )
    return artifacts.checkpoint, artifacts.manifest


def _write_baseline_artifact(tmp_path, cfg, gold, *, model="tfidf", offset=0.0):
    prediction_map = {
        pair_id: min(10.0, max(1.0, score + offset))
        for pair_id, score in zip(
            gold["pair_id"], gold["human_lmp_score"], strict=True
        )
    }
    frame = gold[["pair_id", "language", "legal_domain", "split"]].iloc[::-1].copy()
    frame["prediction"] = frame["pair_id"].map(prediction_map)
    frame["run_mode"] = "final"
    frame["data_source"] = "human_gold_corpus"
    frame["research_use"] = "research"
    path = tmp_path / f"{model}_predictions.csv"
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    metadata = {
        "provenance": {
            "generated_at_utc": "2026-01-01T00:00:00Z",
            "git_revision": "unknown",
            "config_fingerprint": config_fingerprint(cfg),
            "project_version": "1.0",
            "seed": 17,
        },
        "model_name": model,
        "run_mode": "final",
        "seed": 17,
        "device": "cpu",
        "n_predictions": len(frame),
        "splits": list(dict.fromkeys(frame["split"].tolist())),
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
            "slope": 1.0,
            "intercept": 0.0,
            "score_min": 1.0,
            "score_max": 10.0,
            "fit_records": [
                {"pair_id": pair_id, "split": "train"}
                for pair_id in gold.loc[gold["split"] == "train", "pair_id"]
            ],
        },
        "predictions_sha256": _sha256(path),
        "input_data_sha256": "c" * 64,
    }
    if model == "sbert":
        metadata.update(
            {
                "model_identity": {
                    "library": "sentence-transformers",
                    "model_name": "example/configured-model",
                    "revision": "abc123",
                },
                "encoder_runtime": {
                    "source": "configured_sentence_transformer",
                    "configured_model_weights_used": True,
                },
                "loading_policy": {
                    "allow_download": False,
                    "local_files_only": True,
                    "trust_remote_code": False,
                },
            }
        )
    path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return path


def test_real_producer_bundles_are_consumed_by_loader_and_evaluator(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_prediction_artifact
    from src.models import predict_judgebert, sbert_baseline
    from src.models.common import load_model_config
    from src.models.tfidf_baseline import run_tfidf_baseline

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {"epochs": 1, "device": "cpu", "eval_batch_size": 3},
            "evaluation": {
                "bootstrap": {"enabled": False},
                "report_decision_metrics": True,
            },
        }
    )
    gold = _gold().assign(
        original_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Original legal clause {pair_id} shall apply"
        ),
        simplified_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Legal clause {pair_id} applies"
        ),
    )

    tfidf_paths = run_tfidf_baseline(
        gold,
        cfg,
        output_path=tmp_path / "tfidf_predictions.csv",
        label_column="human_lmp_score",
        run_mode="final",
        synthetic=False,
    )

    sbert_constructor_calls = []

    def configured_sbert_constructor(model_name, **kwargs):
        sbert_constructor_calls.append((model_name, kwargs))
        return _ConfiguredSentenceEncoder()

    monkeypatch.setattr(
        sbert_baseline,
        "_load_sentence_transformer_class",
        lambda: configured_sbert_constructor,
    )
    sbert_paths = sbert_baseline.run_sbert_baseline(
        gold,
        cfg,
        output_path=tmp_path / "sbert_predictions.csv",
        label_column="human_lmp_score",
        run_mode="final",
        synthetic=False,
        device="cpu",
    )

    checkpoint_path, manifest_path = _write_final_checkpoint_bundle(
        tmp_path, cfg, gold, monkeypatch
    )
    tokenizer_loads = []
    model_loads = []

    def configured_tokenizer_loader(config, *, allow_download=False):
        tokenizer_loads.append((config, allow_download))
        return _ConfiguredPairTokenizer()

    def configured_model_factory(cls, config, *, allow_download=False):
        model_loads.append((config, allow_download))
        return _judgebert_fixture_model(config)

    monkeypatch.setattr(
        predict_judgebert._training,
        "load_judgebert_tokenizer",
        configured_tokenizer_loader,
    )
    monkeypatch.setattr(
        predict_judgebert.JudgeBertModel,
        "from_config",
        classmethod(configured_model_factory),
    )
    inference_frame = gold[
        ["pair_id", "original_text", "simplified_text"]
    ].copy()
    judgebert_paths = predict_judgebert.run_judgebert_prediction(
        inference_frame,
        checkpoint_path=checkpoint_path,
        output_path=tmp_path / "judgebert_predictions.csv",
        device="cpu",
    )

    producer_paths = [
        tfidf_paths.predictions_csv,
        sbert_paths.predictions_csv,
        judgebert_paths.predictions_csv,
    ]
    loaded = [load_prediction_artifact(path) for path in producer_paths]
    assert [artifact.model_name for artifact in loaded] == [
        "tfidf",
        "sbert",
        "judgebert",
    ]
    assert loaded[1].metadata["encoder_runtime"]["source"] == (
        "configured_sentence_transformer"
    )
    judgebert_artifact = loaded[2]
    assert judgebert_artifact.metadata["checkpoint_manifest_sha256"] == _sha256(
        manifest_path
    )
    assert judgebert_artifact.checkpoint_fit_lineage is not None
    assert judgebert_artifact.checkpoint_fit_lineage.checkpoint_path == checkpoint_path.resolve()
    assert judgebert_artifact.checkpoint_fit_lineage.manifest_path == manifest_path.resolve()
    assert judgebert_artifact.metadata["checkpoint_artifact"] == {
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha256(checkpoint_path),
    }
    assert judgebert_artifact.metadata["checkpoint_manifest_artifact"] == {
        "path": str(manifest_path.resolve()),
        "sha256": _sha256(manifest_path),
    }
    expected_run_identity = hashlib.sha256(
        json.dumps(
            {
                "model_name": judgebert_artifact.metadata["model_name"],
                "model_identity": judgebert_artifact.metadata["model_identity"],
                "run_mode": judgebert_artifact.metadata["run_mode"],
                "seed": judgebert_artifact.metadata["seed"],
                "config_fingerprint": judgebert_artifact.metadata["provenance"][
                    "config_fingerprint"
                ],
                "config_sha256": judgebert_artifact.metadata[
                    "inference_config_identity"
                ]["config_sha256"],
                "code_sha256": judgebert_artifact.metadata["code_sha256"],
                "input_data_sha256": judgebert_artifact.metadata["input_data_sha256"],
                "calibration_identity": None,
                "checkpoint_identity": judgebert_artifact.checkpoint_identity,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert judgebert_artifact.run_identity == expected_run_identity
    assert {
        "language",
        "legal_domain",
        "source_group",
        "split",
    }.isdisjoint(loaded[2].frame.columns)
    assert sbert_constructor_calls
    assert tokenizer_loads and model_loads

    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    result = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=producer_paths,
        output_dir=tmp_path / "evaluation",
    )

    assert set(result["models"]) == {"tfidf", "sbert", "judgebert"}
    assert result["evaluated_gold"]["pair_ids"] == ["P7", "P8", "P9"]


@pytest.mark.parametrize("artifact_name", ["checkpoint", "manifest"])
@pytest.mark.parametrize("mutation", ["alter", "delete"])
def test_evaluator_revalidates_published_judgebert_artifacts_before_output(
    tmp_path, monkeypatch, artifact_name, mutation
) -> None:
    from src.evaluation import evaluate
    from src.models import predict_judgebert
    from src.models.common import load_model_config

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {"epochs": 1, "device": "cpu", "eval_batch_size": 3},
            "evaluation": {
                "bootstrap": {"enabled": False},
                "report_decision_metrics": False,
            },
        }
    )
    gold = _gold().assign(
        original_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Original legal clause {pair_id} shall apply"
        ),
        simplified_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Legal clause {pair_id} applies"
        ),
    )
    checkpoint_path, manifest_path = _write_final_checkpoint_bundle(
        tmp_path, cfg, gold, monkeypatch
    )
    prediction_paths = predict_judgebert.run_judgebert_prediction(
        gold[["pair_id", "original_text", "simplified_text"]].copy(),
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=manifest_path,
        output_path=tmp_path / "published_predictions.csv",
        device="cpu",
    )
    target = checkpoint_path if artifact_name == "checkpoint" else manifest_path
    if mutation == "delete":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b"\n ")
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    output_dir = tmp_path / f"blocked-{artifact_name}-{mutation}"

    with pytest.raises((FileNotFoundError, ValueError)):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_paths.predictions_csv],
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_evaluator_rejects_checkpoint_fit_lineage_not_matching_canonical_gold(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.models import predict_judgebert
    from src.models.common import load_model_config

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {"epochs": 1, "device": "cpu", "eval_batch_size": 3},
            "evaluation": {
                "bootstrap": {"enabled": False},
                "report_decision_metrics": False,
            },
        }
    )
    gold = _gold().assign(
        original_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Original legal clause {pair_id} shall apply"
        ),
        simplified_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Legal clause {pair_id} applies"
        ),
    )
    checkpoint_path, manifest_path = _write_final_checkpoint_bundle(
        tmp_path, cfg, gold, monkeypatch
    )
    inference_frame = gold[
        ["pair_id", "original_text", "simplified_text"]
    ].copy()
    prediction_paths = predict_judgebert.run_judgebert_prediction(
        inference_frame,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=manifest_path,
        output_path=tmp_path / "judgebert_predictions.csv",
        device="cpu",
    )
    changed_gold = gold.copy()
    changed_gold.loc[changed_gold["pair_id"] == "P1", "human_lmp_score"] = 9.5
    monkeypatch.setattr(
        evaluate, "load_master", lambda *, cfg, subset: changed_gold.copy()
    )
    output_dir = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="checkpoint fit.*canonical gold"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_paths.predictions_csv],
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_evaluator_accepts_complete_fit_lineage_from_interleaved_canonical_splits(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.models import predict_judgebert
    from src.models.common import load_model_config

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {"epochs": 1, "device": "cpu", "eval_batch_size": 3},
            "evaluation": {
                "bootstrap": {"enabled": False},
                "report_decision_metrics": False,
            },
        }
    )
    gold = _gold().iloc[[0, 3, 6, 1, 4, 7, 2, 5, 8]].reset_index(drop=True)
    checkpoint_path, manifest_path = _write_final_checkpoint_bundle(
        tmp_path, cfg, gold, monkeypatch
    )
    prediction_paths = predict_judgebert.run_judgebert_prediction(
        gold[["pair_id", "original_text", "simplified_text"]].copy(),
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=manifest_path,
        output_path=tmp_path / "interleaved_predictions.csv",
        device="cpu",
    )
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    result = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_paths.predictions_csv],
        output_dir=tmp_path / "interleaved-evaluation",
    )

    assert result["evaluated_gold"]["pair_ids"] == ["P7", "P8", "P9"]


def test_evaluator_rejects_checkpoint_fit_subset_of_canonical_train_validation(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.models import predict_judgebert
    from src.models.common import load_model_config

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {"epochs": 1, "device": "cpu", "eval_batch_size": 3},
            "evaluation": {
                "bootstrap": {"enabled": False},
                "report_decision_metrics": False,
            },
        }
    )
    gold = _gold()
    subset_for_training = gold.loc[gold["pair_id"] != "P1"].copy()
    checkpoint_path, manifest_path = _write_final_checkpoint_bundle(
        tmp_path, cfg, subset_for_training, monkeypatch
    )
    inference_frame = gold[
        ["pair_id", "original_text", "simplified_text"]
    ].copy()
    prediction_paths = predict_judgebert.run_judgebert_prediction(
        inference_frame,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=manifest_path,
        output_path=tmp_path / "judgebert_predictions.csv",
        device="cpu",
    )
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    output_dir = tmp_path / "must-not-exist-fit-subset"

    with pytest.raises(ValueError, match="complete canonical train/validation"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_paths.predictions_csv],
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_evaluator_accepts_enriched_label_free_judgebert_input_projection(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.models import predict_judgebert
    from src.models.common import load_model_config

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {"epochs": 1, "device": "cpu", "eval_batch_size": 3},
            "evaluation": {
                "bootstrap": {"enabled": False},
                "report_decision_metrics": False,
            },
        }
    )
    gold = _gold()
    checkpoint_path, manifest_path = _write_final_checkpoint_bundle(
        tmp_path, cfg, gold, monkeypatch
    )
    input_columns = [
        "pair_id",
        "original_text",
        "simplified_text",
        "language",
        "legal_domain",
        "source_group",
        "split",
    ]
    prediction_paths = predict_judgebert.run_judgebert_prediction(
        gold[input_columns].copy(),
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=manifest_path,
        output_path=tmp_path / "enriched_predictions.csv",
        device="cpu",
    )
    metadata = json.loads(
        prediction_paths.metadata_json.read_text(encoding="utf-8")
    )
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    result = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_paths.predictions_csv],
        output_dir=tmp_path / "enriched-evaluation",
    )

    assert metadata["input_data_columns"] == input_columns
    assert result["evaluated_gold"]["pair_ids"] == ["P7", "P8", "P9"]


@pytest.mark.parametrize("mutation", ["duplicate", "reorder"])
def test_evaluator_rejects_malformed_enriched_input_column_lineage(
    tmp_path, monkeypatch, mutation
) -> None:
    from src.evaluation import evaluate
    from src.models import predict_judgebert
    from src.models.common import load_model_config

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {"epochs": 1, "device": "cpu", "eval_batch_size": 3},
            "evaluation": {
                "bootstrap": {"enabled": False},
                "report_decision_metrics": False,
            },
        }
    )
    gold = _gold()
    checkpoint_path, manifest_path = _write_final_checkpoint_bundle(
        tmp_path, cfg, gold, monkeypatch
    )
    input_columns = [
        "pair_id",
        "original_text",
        "simplified_text",
        "language",
        "legal_domain",
        "source_group",
        "split",
    ]
    prediction_paths = predict_judgebert.run_judgebert_prediction(
        gold[input_columns].copy(),
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=manifest_path,
        output_path=tmp_path / "enriched_predictions.csv",
        device="cpu",
    )
    metadata = json.loads(
        prediction_paths.metadata_json.read_text(encoding="utf-8")
    )
    if mutation == "duplicate":
        metadata["input_data_columns"].append("split")
    else:
        metadata["input_data_columns"][3:5] = reversed(
            metadata["input_data_columns"][3:5]
        )
    prediction_paths.metadata_json.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    output_dir = tmp_path / f"blocked-input-columns-{mutation}"

    with pytest.raises(ValueError, match="input_data_columns"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_paths.predictions_csv],
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_evaluator_rejects_judgebert_input_text_not_matching_canonical_gold(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.models import predict_judgebert
    from src.models.common import load_model_config

    cfg = load_model_config(
        overrides={
            "models": {"judgebert": {"dropout": 0.0}},
            "training": {"epochs": 1, "device": "cpu", "eval_batch_size": 3},
            "evaluation": {
                "bootstrap": {"enabled": False},
                "report_decision_metrics": False,
            },
        }
    )
    gold = _gold().assign(
        original_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Original legal clause {pair_id} shall apply"
        ),
        simplified_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Legal clause {pair_id} applies"
        ),
    )
    checkpoint_path, manifest_path = _write_final_checkpoint_bundle(
        tmp_path, cfg, gold, monkeypatch
    )
    inference_frame = gold[
        ["pair_id", "original_text", "simplified_text"]
    ].copy()
    prediction_paths = predict_judgebert.run_judgebert_prediction(
        inference_frame,
        checkpoint_path=checkpoint_path,
        checkpoint_manifest_path=manifest_path,
        output_path=tmp_path / "judgebert_predictions.csv",
        device="cpu",
    )
    changed_gold = gold.copy()
    changed_gold.loc[
        changed_gold["pair_id"] == "P7", "original_text"
    ] = "Different authoritative test text"
    monkeypatch.setattr(
        evaluate, "load_master", lambda *, cfg, subset: changed_gold.copy()
    )
    output_dir = tmp_path / "must-not-exist-input-mismatch"

    with pytest.raises(ValueError, match="prediction input.*canonical gold"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_paths.predictions_csv],
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_divergent_model_seed_bundle_uses_training_seed_while_evaluation_keeps_project_seed(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.models.common import load_model_config
    from src.models.tfidf_baseline import run_tfidf_baseline

    cfg = load_model_config(
        overrides={
            "project": {"seed": 17},
            "training": {"seed": 42},
            "evaluation": {
                "bootstrap": {"enabled": False},
                "report_decision_metrics": False,
            },
        }
    )
    gold = _gold().assign(
        original_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Original legal clause {pair_id} shall apply"
        ),
        simplified_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Legal clause {pair_id} applies"
        ),
    )
    paths = run_tfidf_baseline(
        gold,
        cfg,
        output_path=tmp_path / "tfidf_predictions.csv",
        label_column="human_lmp_score",
        run_mode="final",
        synthetic=False,
    )
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))

    assert metadata["seed"] == 42
    assert metadata["provenance"]["seed"] == 42

    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    result = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[paths.predictions_csv],
        output_dir=tmp_path / "evaluation",
    )

    assert result["models"]["tfidf"]["artifact"]["run_provenance"]["seed"] == 42
    assert result["provenance"]["seed"] == 17
    assert result["bootstrap_policy"]["seed"] == 17


def test_final_evaluator_calls_authoritative_gold_seam_before_artifacts_or_output(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate

    events: list[object] = []
    gold = pd.DataFrame({"pair_id": []})
    cfg = {"project": {"seed": 42}}
    output_dir = tmp_path / "must-not-exist"

    def fake_load_master(*, cfg, subset):
        events.append(("load_master", cfg, subset))
        return gold

    def fake_assert_gold(frame, context):
        events.append(("assert_gold", frame, context))
        raise ValueError("TODO_HUMAN_ANNOTATION — blocked by test")

    def forbidden_artifact_read(*args, **kwargs):
        events.append("artifact_read")
        raise AssertionError("prediction artifacts must not be read before gold gate")

    monkeypatch.setattr(evaluate, "load_master", fake_load_master)
    monkeypatch.setattr(evaluate, "assert_gold", fake_assert_gold)
    monkeypatch.setattr(evaluate, "load_prediction_artifact", forbidden_artifact_read)

    with pytest.raises(ValueError, match="TODO_HUMAN_ANNOTATION"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[tmp_path / "missing.csv"],
            output_dir=output_dir,
        )

    assert events == [
        ("load_master", cfg, "gold"),
        ("assert_gold", gold, "final evaluation"),
    ]
    assert not output_dir.exists()


def test_final_evaluator_writes_gold_only_metrics_and_auditable_envelope(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "evaluation"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    result = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )

    metrics_path = output_dir / "metrics.json"
    provenance_path = output_dir / "metrics.metadata.json"
    assert metrics_path.is_file()
    assert provenance_path.is_file()
    written = json.loads(metrics_path.read_text(encoding="utf-8"))
    linkage = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert result == written
    assert written["schema_version"] == 1
    assert written["evaluation_split"] == "test"
    assert written["includes_fit_data"] is False
    assert written["canonical_gold"]["n_rows"] == 9
    assert written["canonical_gold"]["pair_id_sha256"] == _pair_digest(
        gold["pair_id"]
    )
    assert written["evaluated_gold"]["n_rows"] == 3
    assert written["evaluated_gold"]["pair_ids"] == ["P7", "P8", "P9"]
    assert written["label_policy"] == {
        "score": "human_lmp_score",
        "decision": "canonical_human_gold_decision",
        "accepted_annotation_statuses": [
            "human_consensus",
            "human_multi_pending",
            "human_single",
        ],
        "draft_labels_used": False,
    }
    assert written["provenance"]["config_fingerprint"] == config_fingerprint(cfg)
    assert written["provenance"]["seed"] == 17
    assert len(written["provenance"]["evaluation_code_sha256"]) == 64

    model = written["models"]["tfidf"]
    assert model["artifact"]["prediction_file"]["sha256"] == _sha256(
        prediction_path
    )
    assert model["artifact"]["run_identity"]
    assert model["artifact"]["calibration_identity"]
    assert model["artifact"]["checkpoint_identity"] is None
    assert model["score_range"] == [1, 10]
    assert model["regression"]["overall"] == {
        "n_rows": 3,
        "pearson": pytest.approx(1.0),
        "spearman": pytest.approx(1.0),
        "rmse": pytest.approx(0.0),
        "mae": pytest.approx(0.0),
        "r2": pytest.approx(1.0),
    }
    assert set(model["regression"]) == {
        "overall",
        "by_language",
        "by_legal_domain",
        "by_split",
    }
    assert set(model["bootstrap"]) == {
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "r2",
    }
    assert all(
        interval["requested_replicates"] == 20
        and interval["effective_replicates"] == 20
        for interval in model["bootstrap"].values()
    )
    assert model["decision"]["thresholds"]["fit_split"] == "validation"
    assert [
        record["pair_id"]
        for record in model["decision"]["thresholds"]["fit_records"]
    ] == ["P4", "P5", "P6"]
    assert {
        record["source_group"]
        for record in model["decision"]["thresholds"]["fit_records"]
    } == {"g4", "g5", "g6"}
    assert model["decision"]["metrics"] == {
        "accuracy": pytest.approx(1.0),
        "macro_f1": pytest.approx(1.0),
    }
    assert written["paired_comparisons"] == {}
    assert written["output_linkage"] == {
        "metrics_file": "metrics.json",
        "provenance_file": "metrics.metadata.json",
    }
    assert linkage["result_file"] == "metrics.json"
    assert linkage["result_sha256"] == _sha256(metrics_path)
    assert linkage["canonical_pair_id_sha256"] == _pair_digest(gold["pair_id"])
    assert linkage["model_run_identities"] == {"tfidf": model["artifact"]["run_identity"]}


def test_gold_value_provenance_binds_text_labels_and_metadata(tmp_path, monkeypatch) -> None:
    from src.evaluation import evaluate
    from src.models.provenance import dataframe_sha256

    cfg = _config()
    gold = _gold().assign(
        original_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Original canonical text {pair_id}"
        ),
        simplified_text=lambda value: value["pair_id"].map(
            lambda pair_id: f"Simplified canonical text {pair_id}"
        ),
    )
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    current_gold = {"frame": gold}
    monkeypatch.setattr(
        evaluate,
        "load_master",
        lambda *, cfg, subset: current_gold["frame"].copy(),
    )

    first = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=tmp_path / "gold-first",
    )
    changed = gold.copy()
    changed.loc[changed["pair_id"] == "P7", "human_lmp_score"] = 4.0
    changed.loc[changed["pair_id"] == "P7", "decision"] = "Not Preserved"
    current_gold["frame"] = changed
    second = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=tmp_path / "gold-second",
    )

    columns = [
        "pair_id",
        "original_text",
        "simplified_text",
        "human_lmp_score",
        "decision",
        "annotation_status",
        "split",
        "source_group",
        "language",
        "legal_domain",
    ]
    assert first["canonical_gold"]["columns"] == columns
    assert first["canonical_gold"]["sha256"] == dataframe_sha256(gold[columns])
    assert first["evaluated_gold"]["columns"] == columns
    assert first["evaluated_gold"]["sha256"] == dataframe_sha256(
        gold.loc[gold["split"] == "test", columns].reset_index(drop=True)
    )
    assert first["canonical_gold"]["sha256"] != second["canonical_gold"]["sha256"]
    first_linkage = json.loads(
        (tmp_path / "gold-first" / "metrics.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    second_linkage = json.loads(
        (tmp_path / "gold-second" / "metrics.metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert first_linkage["canonical_gold_sha256"] == first["canonical_gold"]["sha256"]
    assert second_linkage["canonical_gold_sha256"] == second["canonical_gold"]["sha256"]
    assert first_linkage["canonical_gold_sha256"] != second_linkage["canonical_gold_sha256"]


@pytest.mark.parametrize("interrupted_target", ["metrics.json", "metrics.metadata.json"])
def test_interrupt_immediately_after_publication_rolls_back_owned_target(
    tmp_path, monkeypatch, interrupted_target
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / f"interrupt-after-{interrupted_target}"
    real_atomic_write = evaluate._atomic_write_json

    def interrupt_after_publication(payload, destination, *, ownership=None):
        real_atomic_write(payload, destination, ownership=ownership)
        if destination.name == interrupted_target:
            raise KeyboardInterrupt("injected interruption after publication")

    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    monkeypatch.setattr(
        evaluate, "_atomic_write_json", interrupt_after_publication
    )

    with pytest.raises(KeyboardInterrupt, match="after publication"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_path],
            output_dir=output_dir,
        )

    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "metrics.metadata.json").exists()
    assert not (output_dir / ".evaluation.lock").exists()
    assert not list(output_dir.glob("*.tmp"))


def test_evaluator_does_not_delete_foreign_lock_after_lock_takeover(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "foreign-lock-takeover"
    lock_path = output_dir / ".evaluation.lock"
    foreign_bytes = b"pid=foreign-writer\n"

    def fail_after_lock_takeover(path):
        lock_path.unlink()
        lock_path.write_bytes(foreign_bytes)
        raise RuntimeError("injected read-back failure after lock takeover")

    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    monkeypatch.setattr(evaluate, "load_evaluation_result", fail_after_lock_takeover)

    with pytest.raises(RuntimeError, match="lock takeover"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_path],
            output_dir=output_dir,
        )

    assert lock_path.read_bytes() == foreign_bytes
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "metrics.metadata.json").exists()
    assert not list(output_dir.glob("*.tmp"))


def test_second_artifact_write_failure_rolls_back_all_evaluation_outputs(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "second-write-failure"
    real_write_json = evaluate.write_json

    def fail_after_writing_second_temp(payload, path):
        written = real_write_json(payload, path)
        if ".metrics.metadata.json." in path.name:
            raise OSError("injected second artifact write failure")
        return written

    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    monkeypatch.setattr(evaluate, "write_json", fail_after_writing_second_temp)

    with pytest.raises(OSError, match="second artifact write failure"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_path],
            output_dir=output_dir,
        )

    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "metrics.metadata.json").exists()
    assert not (output_dir / ".evaluation.lock").exists()
    assert not list(output_dir.glob("*.tmp"))


@pytest.mark.parametrize("foreign_target", ["metrics.json", "metrics.metadata.json"])
def test_evaluation_publication_never_overwrites_or_deletes_foreign_race_winner(
    tmp_path, monkeypatch, foreign_target
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / f"foreign-race-{foreign_target}"
    output_dir.mkdir()
    foreign_path = output_dir / foreign_target
    foreign_bytes = b"foreign concurrent writer\n"
    real_write_json = evaluate.write_json
    injected = False

    def write_json_with_foreign_race(payload, path):
        nonlocal injected
        real_write_json(payload, path)
        if not injected and path.name.startswith(f".{foreign_target}."):
            foreign_path.write_bytes(foreign_bytes)
            injected = True

    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    monkeypatch.setattr(evaluate, "write_json", write_json_with_foreign_race)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_path],
            output_dir=output_dir,
        )

    assert injected
    assert foreign_path.read_bytes() == foreign_bytes
    other_target = (
        output_dir / "metrics.metadata.json"
        if foreign_target == "metrics.json"
        else output_dir / "metrics.json"
    )
    assert not other_target.exists()
    assert not (output_dir / ".evaluation.lock").exists()
    assert not list(output_dir.glob("*.tmp"))


def test_sequential_evaluation_refuses_overwrite_without_changing_first_result(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "sequential-output"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    sidecar_path = output_dir / "metrics.metadata.json"
    original_metrics = metrics_path.read_bytes()
    original_sidecar = sidecar_path.read_bytes()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_path],
            output_dir=output_dir,
        )

    assert metrics_path.read_bytes() == original_metrics
    assert sidecar_path.read_bytes() == original_sidecar
    assert not (output_dir / ".evaluation.lock").exists()
    assert not list(output_dir.glob("*.tmp"))


def test_lock_publication_failure_leaves_no_evaluation_residue(
    tmp_path, monkeypatch
) -> None:
    from pathlib import Path

    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "lock-write-failure"
    lock_path = output_dir / ".evaluation.lock"
    real_open = Path.open

    class _FailingLockHandle:
        def __init__(self, handle):
            self._handle = handle

        def fileno(self):
            return self._handle.fileno()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            self._handle.close()
            return False

        def write(self, _value):
            raise OSError("injected lock write failure")

    def failing_lock_open(self, mode="r", *args, **kwargs):
        if self == lock_path and mode == "x":
            return _FailingLockHandle(real_open(self, mode, *args, **kwargs))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    monkeypatch.setattr(Path, "open", failing_lock_open)

    with pytest.raises(OSError, match="injected lock write failure"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_path],
            output_dir=output_dir,
        )

    assert not lock_path.exists()
    assert not (output_dir / "metrics.json").exists()
    assert not (output_dir / "metrics.metadata.json").exists()
    assert not list(output_dir.glob("*.tmp"))


def test_concurrent_evaluation_writers_have_exactly_one_atomic_winner(
    tmp_path, monkeypatch
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Lock

    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "concurrent-evaluation"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    ensure_barrier = Barrier(2, timeout=20)
    write_barrier = Barrier(2, timeout=20)
    real_ensure = evaluate.ensure_output_directory
    real_write_json = evaluate.write_json
    barrier_lock = Lock()
    metrics_writes = {"count": 0}

    def synchronized_ensure(path):
        directory = real_ensure(path)
        ensure_barrier.wait()
        return directory

    def synchronized_write(obj, path, *args, **kwargs):
        if path.name == "metrics.json":
            with barrier_lock:
                metrics_writes["count"] += 1
            write_barrier.wait()
        return real_write_json(obj, path, *args, **kwargs)

    monkeypatch.setattr(evaluate, "ensure_output_directory", synchronized_ensure)
    monkeypatch.setattr(evaluate, "write_json", synchronized_write)

    def run_once():
        try:
            return evaluate.evaluate_prediction_files(
                cfg=cfg,
                prediction_paths=[prediction_path],
                output_dir=output_dir,
            )
        except Exception as exc:  # asserted below
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: run_once(), range(2)))

    successes = [outcome for outcome in outcomes if isinstance(outcome, dict)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, FileExistsError)]
    assert len(successes) == 1
    assert len(conflicts) == 1
    assert "evaluation output" in str(conflicts[0]).lower()
    assert load_evaluation_result(output_dir / "metrics.json") == successes[0]
    assert not [
        path
        for path in output_dir.iterdir()
        if path.name.endswith(".tmp") or path.name == ".evaluation.lock"
    ]


def test_final_evaluator_reports_paired_model_deltas_with_shared_replicates(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    tfidf_path = _write_baseline_artifact(
        tmp_path, cfg, gold, model="tfidf", offset=0.5
    )
    sbert_path = _write_baseline_artifact(
        tmp_path, cfg, gold, model="sbert", offset=0.0
    )
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    result = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[tfidf_path, sbert_path],
        output_dir=tmp_path / "paired",
    )

    comparison = result["paired_comparisons"]["tfidf_minus_sbert"]
    assert comparison["model_a"] == "tfidf"
    assert comparison["model_b"] == "sbert"
    assert set(comparison["metrics"]) == {
        "pearson",
        "spearman",
        "rmse",
        "mae",
        "r2",
    }
    assert comparison["metrics"]["rmse"]["delta_definition"] == (
        "model_a_minus_model_b"
    )
    assert comparison["metrics"]["rmse"]["delta_estimate"] == pytest.approx(
        (0.5 / 3.0) ** 0.5
    )
    assert all(
        metric["requested_replicates"] == 20
        and metric["effective_replicates"] == 20
        and metric["seed"] == 17
        for metric in comparison["metrics"].values()
    )
    assert result["bootstrap_policy"]["paired_models_use_identical_resamples"] is True


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("allow_fit_data", "allow_fit_data must be a boolean"),
        ("bootstrap_enabled", "bootstrap.enabled must be a boolean"),
        ("report_decisions", "report_decision_metrics must be a boolean"),
        ("score_min", "dataset.score_min must be a number, not a boolean"),
        ("score_max", "dataset.score_max must be a number, not a boolean"),
    ],
)
def test_evaluator_rejects_boolean_surrogates_before_output(
    tmp_path, monkeypatch, case, message
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    evaluation_split = "test"
    allow_fit_data = False
    if case == "allow_fit_data":
        evaluation_split = "validation"
        allow_fit_data = "false"
    elif case == "bootstrap_enabled":
        cfg["evaluation"]["bootstrap"]["enabled"] = "false"
    elif case == "report_decisions":
        cfg["evaluation"]["report_decision_metrics"] = 0
    else:
        cfg["dataset"][case] = True
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / f"invalid-{case}"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    with pytest.raises(ValueError, match=message):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_path],
            output_dir=output_dir,
            evaluation_split=evaluation_split,
            allow_fit_data=allow_fit_data,
        )

    assert not output_dir.exists()


def test_final_evaluator_blocks_fit_split_without_explicit_permission(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "blocked-fit"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    with pytest.raises(ValueError, match="fit data is blocked"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_path],
            output_dir=output_dir,
            evaluation_split="validation",
        )

    assert not output_dir.exists()


def test_final_evaluator_records_explicit_fit_split_evaluation(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    result = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=tmp_path / "allowed-fit",
        evaluation_split="validation",
        allow_fit_data=True,
    )

    assert result["evaluation_split"] == "validation"
    assert result["includes_fit_data"] is True
    assert result["fit_data_evaluation_explicitly_allowed"] is True
    assert result["evaluated_gold"]["pair_ids"] == ["P4", "P5", "P6"]


def test_cli_gold_gate_precedes_missing_prediction_files_and_output_creation(
    tmp_path, monkeypatch, capsys
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    output_dir = tmp_path / "cli-output"
    missing = tmp_path / "missing_predictions.csv"
    events: list[object] = []

    monkeypatch.setattr(evaluate, "load_config", lambda path: cfg)

    def fake_load_master(*, cfg, subset):
        events.append(("load_master", subset))
        return _gold().iloc[0:0].copy()

    monkeypatch.setattr(evaluate, "load_master", fake_load_master)

    exit_code = evaluate.main(
        [
            "--config",
            str(tmp_path / "config.yaml"),
            "--predictions",
            str(missing),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "TODO_HUMAN_ANNOTATION" in captured.err
    assert events == [("load_master", "gold")]
    assert not output_dir.exists()


def test_evaluation_result_loader_reads_valid_linked_envelope(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "result-validation"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    expected = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )

    loaded = load_evaluation_result(output_dir / "metrics.json")

    assert loaded == expected


def test_evaluation_result_loader_rejects_tampered_metrics_file(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "tampered-result"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        metrics_path.read_text(encoding="utf-8").replace(
            '"schema_version": 1', '"schema_version": 2', 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale or tampered.*result_sha256"):
        load_evaluation_result(metrics_path)


def test_evaluation_loader_hashes_the_same_result_bytes_that_it_parses(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate, provenance
    from src.utils import io as io_utils

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "read-once-result"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    sidecar_path = output_dir / "metrics.metadata.json"
    original_bytes = metrics_path.read_bytes()
    mutated_bytes = original_bytes.replace(
        b'"schema_version": 1', b'"schema_version": 2', 1
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["result_sha256"] = hashlib.sha256(mutated_bytes).hexdigest()
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    real_read_json = io_utils.read_json

    def mutate_result_after_parse(target):
        parsed = real_read_json(target)
        if target == sidecar_path:
            metrics_path.write_bytes(mutated_bytes)
        return parsed

    monkeypatch.setattr(io_utils, "read_json", mutate_result_after_parse)

    with pytest.raises(ValueError, match="stale or tampered.*result_sha256"):
        provenance.load_evaluation_result(metrics_path)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("canonical_pair_id_sha256", "f" * 64),
        ("config_fingerprint", "wrong-config"),
        ("seed", 999),
        ("seed", 17.0),
        ("evaluation_split", "validation"),
        ("model_run_identities", {"tfidf": "0" * 64}),
    ],
)
def test_evaluation_result_loader_rejects_semantically_mismatched_sidecar(
    tmp_path, monkeypatch, field, bad_value
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "mismatched-sidecar"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    sidecar_path = output_dir / "metrics.metadata.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar[field] = bad_value
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="sidecar.*does not match"):
        load_evaluation_result(metrics_path)


@pytest.mark.parametrize(
    ("target", "bad_version"),
    [
        ("result", True),
        ("result", 1.0),
        ("sidecar", True),
        ("sidecar", 1.0),
    ],
)
def test_evaluation_result_loader_requires_exact_integer_schema_versions(
    tmp_path, monkeypatch, target, bad_version
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / f"bad-{target}-schema-{type(bad_version).__name__}"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    sidecar_path = output_dir / "metrics.metadata.json"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if target == "result":
        result["schema_version"] = bad_version
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        sidecar["result_sha256"] = _sha256(metrics_path)
    else:
        sidecar["schema_version"] = bad_version
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version.*integer 1"):
        load_evaluation_result(metrics_path)


@pytest.mark.parametrize(
    "case",
    [
        "malformed_canonical_sha",
        "reordered_canonical_columns",
        "malformed_evaluated_sha",
        "reordered_evaluated_columns",
    ],
)
def test_evaluation_result_loader_validates_gold_projection_envelopes(
    tmp_path, monkeypatch, case
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / f"malformed-gold-projection-{case}"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    sidecar_path = output_dir / "metrics.metadata.json"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if case == "malformed_canonical_sha":
        result["canonical_gold"]["sha256"] = "not-a-sha256"
        sidecar["canonical_gold_sha256"] = "not-a-sha256"
    elif case == "reordered_canonical_columns":
        result["canonical_gold"]["columns"] = list(
            reversed(result["canonical_gold"]["columns"])
        )
    elif case == "malformed_evaluated_sha":
        result["evaluated_gold"]["sha256"] = "not-a-sha256"
    else:
        result["evaluated_gold"]["columns"] = list(
            reversed(result["evaluated_gold"]["columns"])
        )
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sidecar["result_sha256"] = _sha256(metrics_path)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="gold projection"):
        load_evaluation_result(metrics_path)


@pytest.mark.parametrize(
    "case",
    [
        "includes_fit_data_string",
        "explicit_fit_permission_integer",
        "draft_labels_integer",
        "bootstrap_enabled_integer",
        "paired_resamples_integer",
        "canonical_count_below_evaluated",
    ],
)
def test_evaluation_result_loader_rejects_invalid_control_scalars_and_counts(
    tmp_path, monkeypatch, case
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / f"invalid-result-control-{case}"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    sidecar_path = output_dir / "metrics.metadata.json"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if case == "includes_fit_data_string":
        result["includes_fit_data"] = "false"
    elif case == "explicit_fit_permission_integer":
        result["fit_data_evaluation_explicitly_allowed"] = 0
    elif case == "draft_labels_integer":
        result["label_policy"]["draft_labels_used"] = 0
    elif case == "bootstrap_enabled_integer":
        result["bootstrap_policy"]["enabled"] = 0
    elif case == "paired_resamples_integer":
        result["bootstrap_policy"]["paired_models_use_identical_resamples"] = 1
    else:
        result["canonical_gold"]["n_rows"] = 0
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sidecar["result_sha256"] = _sha256(metrics_path)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Evaluation result"):
        load_evaluation_result(metrics_path)


@pytest.mark.parametrize(
    "case",
    [
        "draft_score_label",
        "draft_annotation_status",
        "bootstrap_row_unit",
        "enabled_bootstrap_missing_model_reports",
        "disabled_bootstrap_retains_model_reports",
        "disabled_bootstrap_retains_paired_comparison",
        "paired_comparisons_list",
        "model_row_count_mismatch",
        "regression_metric_string",
        "decision_metric_string",
        "decision_threshold_boolean",
        "unknown_model_identity",
        "model_key_artifact_mismatch",
        "unknown_bootstrap_metric_identity",
        "wrong_sidecar_artifact",
    ],
)
def test_evaluation_result_loader_rejects_semantically_inconsistent_envelope(
    tmp_path, monkeypatch, case
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / f"semantic-result-envelope-{case}"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    sidecar_path = output_dir / "metrics.metadata.json"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    result_changed = True
    if case == "draft_score_label":
        result["label_policy"]["score"] = "draft_lmp_score"
    elif case == "draft_annotation_status":
        result["label_policy"]["accepted_annotation_statuses"] = ["draft"]
    elif case == "bootstrap_row_unit":
        result["bootstrap_policy"]["unit"] = "row"
    elif case == "enabled_bootstrap_missing_model_reports":
        result["models"]["tfidf"]["bootstrap"] = {}
    elif case == "disabled_bootstrap_retains_model_reports":
        result["bootstrap_policy"].update(
            {"enabled": False, "n_resamples": 0, "confidence": None}
        )
    elif case == "disabled_bootstrap_retains_paired_comparison":
        result["bootstrap_policy"].update(
            {"enabled": False, "n_resamples": 0, "confidence": None}
        )
        result["models"]["tfidf"]["bootstrap"] = {}
        result["paired_comparisons"] = {"forged": {}}
    elif case == "paired_comparisons_list":
        result["paired_comparisons"] = []
    elif case == "model_row_count_mismatch":
        result["models"]["tfidf"]["regression"]["overall"]["n_rows"] = 2
    elif case == "regression_metric_string":
        result["models"]["tfidf"]["regression"]["overall"]["rmse"] = "forged"
    elif case == "decision_metric_string":
        result["models"]["tfidf"]["decision"]["metrics"]["accuracy"] = "forged"
    elif case == "decision_threshold_boolean":
        result["models"]["tfidf"]["decision"]["thresholds"]["partial_min"] = True
    elif case == "unknown_model_identity":
        result["models"]["forged_model"] = result["models"].pop("tfidf")
        sidecar["model_run_identities"]["forged_model"] = sidecar[
            "model_run_identities"
        ].pop("tfidf")
    elif case == "model_key_artifact_mismatch":
        result["models"]["sbert"] = result["models"].pop("tfidf")
        sidecar["model_run_identities"]["sbert"] = sidecar[
            "model_run_identities"
        ].pop("tfidf")
    elif case == "unknown_bootstrap_metric_identity":
        bootstrap = result["models"]["tfidf"]["bootstrap"]
        forged_interval = dict(next(iter(bootstrap.values())))
        forged_interval["metric"] = "forged_metric"
        bootstrap["forged_metric"] = forged_interval
    else:
        sidecar["artifact"] = "unrelated_artifact"
        result_changed = False
    if result_changed:
        metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        sidecar["result_sha256"] = _sha256(metrics_path)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_evaluation_result(metrics_path)


@pytest.mark.parametrize(
    "bad_version",
    [1.0, "1", None, True, False, 0, 2, [], {}],
    ids=[
        "float-one",
        "string-one",
        "null",
        "true",
        "false",
        "zero",
        "unsupported-two",
        "list",
        "object",
    ],
)
def test_evaluation_result_loader_requires_exact_decision_schema_version(
    tmp_path, monkeypatch, bad_version
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / f"decision-schema-{type(bad_version).__name__}"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )
    metrics_path = output_dir / "metrics.json"
    sidecar_path = output_dir / "metrics.metadata.json"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    result["models"]["tfidf"]["decision"]["thresholds"][
        "schema_version"
    ] = bad_version
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sidecar["result_sha256"] = _sha256(metrics_path)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema_version.*integer 1"):
        load_evaluation_result(metrics_path)


def test_evaluation_result_loader_accepts_integer_decision_schema_version_one(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "decision-schema-valid"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())
    evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )

    loaded = load_evaluation_result(output_dir / "metrics.json")
    version = loaded["models"]["tfidf"]["decision"]["thresholds"][
        "schema_version"
    ]
    assert type(version) is int
    assert version == 1


def test_final_evaluator_reads_back_exact_written_result_before_returning(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate
    from src.evaluation.provenance import load_evaluation_result as real_loader

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    output_dir = tmp_path / "read-back"
    observed: list[object] = []
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    def recording_loader(path):
        observed.append(path)
        return real_loader(path)

    monkeypatch.setattr(evaluate, "load_evaluation_result", recording_loader)

    result = evaluate.evaluate_prediction_files(
        cfg=cfg,
        prediction_paths=[prediction_path],
        output_dir=output_dir,
    )

    assert observed == [(output_dir / "metrics.json").resolve()]
    assert result == real_loader(output_dir / "metrics.json")


def test_evaluation_package_exposes_final_entry_point() -> None:
    from src.evaluation import evaluate_prediction_files

    assert callable(evaluate_prediction_files)


def test_final_evaluator_rejects_calibration_id_disguised_as_non_test(
    tmp_path, monkeypatch
) -> None:
    from src.evaluation import evaluate

    cfg = _config()
    gold = _gold()
    prediction_path = _write_baseline_artifact(tmp_path, cfg, gold)
    sidecar_path = prediction_path.with_suffix(".metadata.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["calibration"]["fit_records"][0] = {
        "pair_id": "P7",
        "split": "train",
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    output_dir = tmp_path / "leaked-calibration"
    monkeypatch.setattr(evaluate, "load_master", lambda *, cfg, subset: gold.copy())

    with pytest.raises(ValueError, match="calibration.*P7.*canonical split.*test"):
        evaluate.evaluate_prediction_files(
            cfg=cfg,
            prediction_paths=[prediction_path],
            output_dir=output_dir,
        )

    assert not output_dir.exists()
