"""Unit tests for the JUDGEBERT-inspired MuRIL cross-encoder core."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.common import load_model_config


class TinyEncoder(nn.Module):
    """Deterministic download-free encoder with the Hugging Face call contract."""

    def __init__(self, hidden_size: int = 3) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls: list[dict[str, object]] = []

    def forward(self, **kwargs):
        self.calls.append(dict(kwargs))
        input_ids = kwargs["input_ids"]
        basis = torch.arange(
            1, self.config.hidden_size + 1, device=input_ids.device, dtype=torch.float32
        )
        hidden = input_ids.to(torch.float32).unsqueeze(-1) * basis * self.scale
        attentions = None
        if kwargs.get("output_attentions"):
            batch, length = input_ids.shape
            attentions = (
                torch.eye(length, device=input_ids.device)
                .expand(batch, 1, length, length)
                .clone(),
            )
        return SimpleNamespace(last_hidden_state=hidden, attentions=attentions)


def _config(**judgebert_overrides):
    return load_model_config(
        overrides={"models": {"judgebert": judgebert_overrides}}
    )


def test_config_identity_and_target_scaling_round_trip_are_exact() -> None:
    from src.models.judgebert_model import (
        MODEL_IDENTITY,
        JudgeBertConfig,
        scale_targets,
        unscale_targets,
    )

    settings = JudgeBertConfig.from_project_config(load_model_config())
    targets = torch.tensor([1.0, 5.5, 10.0], dtype=torch.float64)

    scaled = scale_targets(targets)

    assert MODEL_IDENTITY == "JUDGEBERT-inspired MuRIL cross-encoder"
    assert settings.backbone == "google/muril-base-cased"
    assert scaled.dtype == torch.float64
    assert torch.equal(scaled, torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64))
    assert torch.equal(unscale_targets(scaled), targets)
    for invalid in (torch.tensor([0.99]), torch.tensor([10.01]), torch.tensor([float("nan")])):
        with pytest.raises(ValueError, match="finite|within.*1.*10"):
            scale_targets(invalid)
    with pytest.raises(ValueError, match="0.*1"):
        unscale_targets(torch.tensor([-0.01]))


def test_cls_forward_returns_structured_output_and_passes_pair_segments() -> None:
    from src.models.judgebert_model import JudgeBertModel, JudgeBertOutput

    encoder = TinyEncoder()
    model = JudgeBertModel.from_config(
        _config(dropout=0.0), encoder=encoder
    )
    linear = model.regression_head[-1]
    assert isinstance(linear, nn.Linear)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()

    input_ids = torch.tensor([[1, 2, 9], [3, 4, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.tensor([[0, 0, 1], [0, 1, 0]])
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=torch.tensor([1.0, 1.0]),
        output_attentions=True,
    )

    expected_cls = torch.tensor([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]])
    assert isinstance(output, JudgeBertOutput)
    assert torch.equal(output.pooled_output, expected_cls)
    assert output.logits.shape == (2,)
    assert torch.equal(output.scores, torch.tensor([1.0, 1.0]))
    assert output.loss is not None and output.loss.item() == pytest.approx(0.0)
    assert output.attentions is not None and len(output.attentions) == 1
    assert torch.equal(encoder.calls[-1]["token_type_ids"], token_type_ids)
    assert encoder.calls[-1]["return_dict"] is True
    assert encoder.calls[-1]["output_attentions"] is True


def test_masked_mean_pooling_excludes_padding_and_omits_segment_ids() -> None:
    from src.models.judgebert_model import JudgeBertModel

    encoder = TinyEncoder()
    model = JudgeBertModel.from_config(
        _config(pooling="mean", dropout=0.0), encoder=encoder
    )

    output = model(
        input_ids=torch.tensor([[2, 4, 100], [3, 9, 12]]),
        attention_mask=torch.tensor([[1, 1, 0], [1, 0, 0]]),
    )

    assert torch.equal(
        output.pooled_output,
        torch.tensor([[3.0, 6.0, 9.0], [3.0, 6.0, 9.0]]),
    )
    assert "token_type_ids" not in encoder.calls[-1]


@pytest.mark.parametrize(
    ("hidden_size", "linear_shapes"),
    [(0, [(3, 1)]), (5, [(3, 5), (5, 1)])],
)
def test_regression_head_matches_configured_shape(hidden_size, linear_shapes) -> None:
    from src.models.judgebert_model import JudgeBertModel

    model = JudgeBertModel.from_config(
        _config(head_hidden_size=hidden_size), encoder=TinyEncoder()
    )

    linears = [layer for layer in model.regression_head if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == linear_shapes
    output = model(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
    )
    assert output.logits.shape == (1,)


@pytest.mark.parametrize(
    ("loss_name", "expected"), [("smooth_l1", 0.75), ("mse", 1.0)]
)
def test_configured_loss_has_exact_value_and_finite_gradients(loss_name, expected) -> None:
    from src.models.judgebert_model import JudgeBertModel

    encoder = TinyEncoder()
    model = JudgeBertModel.from_config(
        _config(loss=loss_name, dropout=0.0, smooth_l1_beta=0.5),
        encoder=encoder,
    )
    linear = model.regression_head[-1]
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()

    output = model(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
        labels=torch.tensor([10.0]),
    )
    assert output.loss is not None
    assert output.loss.item() == pytest.approx(expected)

    output.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_eval_mode_is_deterministic_with_configured_dropout() -> None:
    from src.models.judgebert_model import JudgeBertModel

    model = JudgeBertModel.from_config(
        _config(dropout=0.8, head_hidden_size=4), encoder=TinyEncoder()
    ).eval()
    inputs = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
    }

    first = model(**inputs)
    second = model(**inputs)

    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first.scores, second.scores)


def test_predict_scores_is_finite_detached_and_clipped_to_one_through_ten() -> None:
    from src.models.judgebert_model import JudgeBertModel

    model = JudgeBertModel.from_config(
        _config(target_scaling="none", dropout=0.0), encoder=TinyEncoder()
    )
    linear = model.regression_head[-1]
    with torch.no_grad():
        linear.weight.zero_()
        linear.weight[0, 0] = 10.0
        linear.bias.fill_(-5.0)

    scores = model.predict_scores(
        input_ids=torch.tensor([[0, 2], [2, 0]]),
        attention_mask=torch.ones((2, 2), dtype=torch.long),
    )

    assert torch.equal(scores, torch.tensor([1.0, 10.0]))
    assert torch.isfinite(scores).all()
    assert scores.requires_grad is False


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("enabled", False),
        ("backbone", ""),
        ("max_seq_length", 0),
        ("dropout", 1.0),
        ("head_hidden_size", -1),
        ("pooling", "max"),
        ("target_scaling", "zscore"),
        ("loss", "mae"),
        ("smooth_l1_beta", 0.0),
        ("clip_predictions", 1),
    ],
)
def test_invalid_judgebert_configuration_is_rejected(setting, value) -> None:
    from src.models.judgebert_model import JudgeBertConfig

    cfg = load_model_config()
    cfg["models"]["judgebert"][setting] = value

    with pytest.raises(ValueError, match=f"judgebert\\.{setting}"):
        JudgeBertConfig.from_project_config(cfg)


@pytest.mark.parametrize("bound", ["score_min", "score_max"])
def test_judgebert_rejects_boolean_dataset_score_bounds(bound) -> None:
    from src.models.judgebert_model import JudgeBertConfig

    cfg = load_model_config()
    cfg["dataset"][bound] = True

    with pytest.raises(ValueError, match=f"dataset\\.{bound}"):
        JudgeBertConfig.from_project_config(cfg)


@pytest.mark.parametrize(
    "bad_mask",
    [
        torch.tensor([1, 1, 1]),
        torch.tensor([[1, 1]]),
        torch.tensor([[1, 2, 0]]),
        torch.tensor([[0, 0, 0]]),
    ],
)
def test_invalid_attention_masks_are_rejected_before_encoder_use(bad_mask) -> None:
    from src.models.judgebert_model import JudgeBertModel

    encoder = TinyEncoder()
    model = JudgeBertModel.from_config(
        _config(pooling="mean"), encoder=encoder
    )

    with pytest.raises(ValueError, match="attention_mask"):
        model(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=bad_mask,
        )

    assert encoder.calls == []


def test_mismatched_token_type_ids_are_rejected_before_encoder_use() -> None:
    from src.models.judgebert_model import JudgeBertModel

    encoder = TinyEncoder()
    model = JudgeBertModel.from_config(load_model_config(), encoder=encoder)

    with pytest.raises(ValueError, match="token_type_ids"):
        model(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones((1, 3), dtype=torch.long),
            token_type_ids=torch.zeros((1, 2), dtype=torch.long),
        )

    assert encoder.calls == []


def test_nonfinite_regression_output_fails_instead_of_emitting_a_score() -> None:
    from src.models.judgebert_model import MODEL_IDENTITY, JudgeBertModel

    model = JudgeBertModel.from_config(load_model_config(), encoder=TinyEncoder())
    with torch.no_grad():
        model.regression_head[-1].bias.fill_(float("nan"))

    with pytest.raises(RuntimeError, match=f"{MODEL_IDENTITY}.*non-finite"):
        model(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )


@pytest.mark.parametrize(
    "bad_labels",
    [
        torch.tensor([5.0, 6.0]),
        torch.tensor([[5.0, 6.0]]),
        torch.tensor([0.0]),
        torch.tensor([float("inf")]),
    ],
)
def test_labels_must_match_batch_and_remain_finite_one_to_ten(bad_labels) -> None:
    from src.models.judgebert_model import JudgeBertModel

    encoder = TinyEncoder()
    model = JudgeBertModel.from_config(
        _config(target_scaling="none"), encoder=encoder
    )

    with pytest.raises(ValueError, match="Labels|labels"):
        model(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            labels=bad_labels,
        )

    assert encoder.calls == []


def test_auto_model_loading_is_lazy_local_only_and_never_trusts_remote_code(
    monkeypatch,
) -> None:
    from src.models import judgebert_model

    calls = []
    observed_offline_environment = []
    imported_constants = SimpleNamespace(HF_HUB_OFFLINE=True)

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            sys.modules["huggingface_hub.constants"] = imported_constants
            observed_offline_environment.append(
                (
                    os.environ.get("HF_HUB_OFFLINE"),
                    os.environ.get("TRANSFORMERS_OFFLINE"),
                )
            )
            return TinyEncoder()

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    monkeypatch.setattr(
        judgebert_model, "_load_auto_model_class", lambda: FakeAutoModel, raising=False
    )

    model = judgebert_model.JudgeBertModel.from_config(load_model_config())
    assert model.encoder is None
    assert calls == []

    output = model(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
    )

    assert output.scores.shape == (1,)
    assert calls == [
        (
            "google/muril-base-cased",
            {"local_files_only": True, "trust_remote_code": False},
        )
    ]
    assert observed_offline_environment == [("1", "1")]
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert imported_constants.HF_HUB_OFFLINE is False


def test_lazily_created_regression_head_inherits_eval_mode(monkeypatch) -> None:
    from src.models import judgebert_model

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return TinyEncoder()

    monkeypatch.setattr(
        judgebert_model, "_load_auto_model_class", lambda: FakeAutoModel
    )
    model = judgebert_model.JudgeBertModel.from_config(
        _config(dropout=0.8)
    ).eval()

    model(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
    )

    assert model.regression_head is not None
    assert model.regression_head.training is False


def test_allow_download_must_be_explicit_and_disables_local_only(monkeypatch) -> None:
    from src.models import judgebert_model

    calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            return TinyEncoder()

    monkeypatch.setattr(
        judgebert_model, "_load_auto_model_class", lambda: FakeAutoModel, raising=False
    )
    model = judgebert_model.JudgeBertModel.from_config(
        load_model_config(), allow_download=True
    )

    model(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones((1, 2), dtype=torch.long),
    )

    assert calls == [
        (
            "google/muril-base-cased",
            {"local_files_only": False, "trust_remote_code": False},
        )
    ]


def test_missing_local_muril_names_exact_identity_and_never_falls_back(
    monkeypatch,
) -> None:
    from src.models import judgebert_model

    calls = []

    class MissingAutoModel:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((model_name, kwargs))
            raise OSError("not present in local cache")

    monkeypatch.setattr(
        judgebert_model,
        "_load_auto_model_class",
        lambda: MissingAutoModel,
        raising=False,
    )
    model = judgebert_model.JudgeBertModel.from_config(load_model_config())

    with pytest.raises(RuntimeError) as caught:
        model(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )

    assert str(caught.value) == (
        "JUDGEBERT-inspired MuRIL cross-encoder backbone "
        "'google/muril-base-cased' is unavailable with local_files_only=True; "
        "cache the exact model locally or pass allow_download=True explicitly. "
        "No fallback model will be used. Original error: not present in local cache"
    )
    assert len(calls) == 1
    assert model.encoder is None
