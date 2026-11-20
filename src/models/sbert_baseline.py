"""Offline-safe multilingual Sentence-BERT similarity baseline."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from src.models.calibration import LinearScoreCalibrator
from src.models.common import (
    load_model_config,
    load_model_corpus,
    require_gold,
    select_device,
    setup_model_seed,
    validate_model_corpus,
    validate_predictions,
)
from src.models.provenance import (
    ResultPaths,
    atomic_write_csv,
    atomic_write_json,
    build_result_metadata,
    dataframe_sha256,
    file_sha256,
    validate_prediction_output_paths,
)


@contextmanager
def _huggingface_offline_guard(enabled: bool):
    """Block Hub traffic during default local-only model discovery."""
    if not enabled:
        yield
        return

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = "1"

    constants = sys.modules.get("huggingface_hub.constants")
    if constants is not None:
        constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        constants = sys.modules.get("huggingface_hub.constants")
        if constants is not None:
            truthy = {"1", "ON", "TRUE", "YES"}
            constants.HF_HUB_OFFLINE = any(
                (previous[name] or "").upper() in truthy for name in names
            )


def _load_sentence_transformer_class() -> Any:
    """Import SentenceTransformer only when an encoder is actually needed."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


@dataclass(frozen=True)
class SbertConfig:
    """Validated Sentence-BERT settings read from the project configuration."""

    model_name: str
    batch_size: int
    normalize_embeddings: bool
    max_seq_length: int
    calibration: str
    clip_to_score_range: bool

    @classmethod
    def from_project_config(cls, cfg: Mapping[str, Any]) -> "SbertConfig":
        """Build settings from ``models.sbert``."""
        models_cfg = cfg.get("models")
        if not isinstance(models_cfg, Mapping):
            raise ValueError("Project config is missing the 'models' mapping")
        model_cfg = models_cfg.get("sbert")
        if not isinstance(model_cfg, Mapping):
            raise ValueError("Project config is missing the 'models.sbert' mapping")
        required = {
            "enabled",
            "model_name",
            "batch_size",
            "normalize_embeddings",
            "max_seq_length",
            "calibration",
            "clip_to_score_range",
        }
        missing = sorted(required - set(model_cfg))
        if missing:
            raise ValueError(f"models.sbert is missing setting(s): {missing}")
        if model_cfg["enabled"] is not True:
            raise ValueError("models.sbert.enabled must be true to run this baseline")

        model_name = model_cfg["model_name"]
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("models.sbert.model_name must be a non-empty string")
        for name in ("batch_size", "max_seq_length"):
            value = model_cfg[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"models.sbert.{name} must be a positive integer")
        for name in ("normalize_embeddings", "clip_to_score_range"):
            if not isinstance(model_cfg[name], bool):
                raise ValueError(f"models.sbert.{name} must be a boolean")
        if model_cfg["calibration"] != "linear_regression":
            raise ValueError(
                "models.sbert.calibration must be 'linear_regression'"
            )

        return cls(
            model_name=model_name.strip(),
            batch_size=model_cfg["batch_size"],
            normalize_embeddings=model_cfg["normalize_embeddings"],
            max_seq_length=model_cfg["max_seq_length"],
            calibration=model_cfg["calibration"],
            clip_to_score_range=model_cfg["clip_to_score_range"],
        )


class SbertBaseline:
    """Pretrained Sentence-BERT pair similarity with injected-encoder support."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        encoder: Any | None = None,
        device: str | None = None,
        allow_download: bool = False,
    ) -> None:
        self.project_config = dict(cfg)
        self.settings = SbertConfig.from_project_config(cfg)
        requested_device = device or str((cfg.get("training") or {}).get("device", "auto"))
        self.device = select_device(requested_device)
        self.allow_download = bool(allow_download)
        self.encoder = encoder
        self.calibrator: LinearScoreCalibrator | None = None
        if self.encoder is not None:
            self.encoder.max_seq_length = self.settings.max_seq_length

    @staticmethod
    def _validate_text_columns(frame: pd.DataFrame, *, context: str) -> None:
        for column in ("original_text", "simplified_text"):
            if column not in frame.columns:
                raise ValueError(f"{context} is missing required column {column!r}")
            values = frame[column]
            invalid = values.isna() | ~values.map(lambda value: isinstance(value, str))
            blank = values.astype("string").str.strip().eq("").fillna(True)
            bad = (invalid | blank).fillna(True)
            if bad.any():
                raise ValueError(
                    f"{context} {column} contains {int(bad.sum())} null or blank value(s)"
                )

    def _ensure_encoder(self) -> Any:
        if self.encoder is None:
            local_files_only = not self.allow_download
            try:
                with _huggingface_offline_guard(local_files_only):
                    transformer_class = _load_sentence_transformer_class()
                    loaded = transformer_class(
                        self.settings.model_name,
                        device=str(self.device),
                        local_files_only=local_files_only,
                        trust_remote_code=False,
                    )
            except Exception as exc:
                if local_files_only:
                    policy = (
                        "local_files_only=True. Place the weights in the local "
                        "Hugging Face cache or pass --allow-download explicitly"
                    )
                else:
                    policy = "downloads allowed (local_files_only=False)"
                raise RuntimeError(
                    f"Sentence-BERT model {self.settings.model_name!r} is unavailable "
                    f"with {policy}: {exc}"
                ) from exc
            loaded.max_seq_length = self.settings.max_seq_length
            self.encoder = loaded
        return self.encoder

    @staticmethod
    def _validated_embeddings(
        values: Any, *, side: str, expected_rows: int
    ) -> np.ndarray:
        embeddings = np.asarray(values, dtype=float)
        if embeddings.ndim != 2:
            raise ValueError(f"{side} embeddings must be a two-dimensional array")
        if embeddings.shape[0] != expected_rows:
            raise ValueError(
                f"{side} embeddings returned {embeddings.shape[0]} rows; "
                f"expected {expected_rows}"
            )
        if embeddings.shape[1] == 0:
            raise ValueError(f"{side} embeddings must have a nonzero dimension")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{side} embeddings must all be finite")
        norms = np.linalg.norm(embeddings, axis=1)
        if (norms <= 0.0).any():
            raise ValueError(f"{side} embeddings must contain no zero vectors")
        return embeddings

    def raw_similarities(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Encode each text side independently and return paired row cosine."""
        validate_model_corpus(frame)
        self._validate_text_columns(frame, context="Sentence-BERT prediction data")
        encoder = self._ensure_encoder()
        encode_kwargs = {
            "batch_size": self.settings.batch_size,
            "convert_to_numpy": True,
            "normalize_embeddings": self.settings.normalize_embeddings,
            "show_progress_bar": False,
        }
        original = self._validated_embeddings(
            encoder.encode(frame["original_text"].tolist(), **encode_kwargs),
            side="original",
            expected_rows=len(frame),
        )
        simplified = self._validated_embeddings(
            encoder.encode(frame["simplified_text"].tolist(), **encode_kwargs),
            side="simplified",
            expected_rows=len(frame),
        )
        if original.shape != simplified.shape:
            raise ValueError(
                "Original and simplified embeddings must have the same shape; "
                f"got {original.shape} and {simplified.shape}"
            )
        numerator = np.einsum("ij,ij->i", original, simplified)
        denominator = np.linalg.norm(original, axis=1) * np.linalg.norm(
            simplified, axis=1
        )
        similarities = np.clip(numerator / denominator, -1.0, 1.0)
        if not np.isfinite(similarities).all():
            raise ValueError("Sentence-BERT cosine similarities must all be finite")
        return pd.DataFrame({"cosine_similarity": similarities.astype(float)})

    def fit(
        self, frame: pd.DataFrame, *, label_column: str = "label"
    ) -> "SbertBaseline":
        """Fit only the affine score calibration and only on train rows."""
        if "draft" in str(label_column).strip().casefold():
            raise ValueError(
                f"{label_column} is a draft field and must never be used as a label"
            )
        validate_model_corpus(frame)
        required = {
            "language",
            "legal_domain",
            "original_text",
            "simplified_text",
            label_column,
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Sentence-BERT fit data is missing columns: {missing}")
        self._validate_text_columns(frame, context="Sentence-BERT fit data")
        train = frame.loc[frame["split"].astype(str).str.strip().eq("train")].copy()
        if train.empty:
            raise ValueError("Sentence-BERT fit requires at least one train row")

        similarities = self.raw_similarities(train)
        calibration_frame = pd.DataFrame(
            {
                "pair_id": train["pair_id"].tolist(),
                "split": ["train"] * len(train),
                "raw_score": similarities["cosine_similarity"].to_numpy(),
                label_column: train[label_column].tolist(),
            }
        )
        self.calibrator = LinearScoreCalibrator.fit(
            calibration_frame,
            raw_score_column="raw_score",
            label_column=label_column,
            fit_splits=("train",),
        )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return calibrated scores with pair metadata in input row order."""
        required = {"pair_id", "language", "legal_domain", "split"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Sentence-BERT prediction data is missing columns: {missing}"
            )
        if self.calibrator is None:
            raise RuntimeError("Sentence-BERT baseline must be fitted before prediction")

        similarities = self.raw_similarities(frame)
        calibrated = self.calibrator.transform(
            similarities["cosine_similarity"].to_numpy(),
            clip=self.settings.clip_to_score_range,
        )
        checked = validate_predictions(
            pd.DataFrame(
                {"pair_id": frame["pair_id"].tolist(), "prediction": calibrated}
            ),
            frame["pair_id"].tolist(),
        )
        metadata_columns = [
            column
            for column in (
                "pair_id",
                "language",
                "legal_domain",
                "source_group",
                "split",
            )
            if column in frame.columns
        ]
        output = frame[metadata_columns].copy().reset_index(drop=True)
        output = pd.concat([output, similarities.reset_index(drop=True)], axis=1)
        output["prediction"] = checked["prediction"].to_numpy()
        return output


def run_sbert_baseline(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    output_path: str | Path,
    label_column: str,
    run_mode: str,
    synthetic: bool,
    encoder: Any | None = None,
    allow_download: bool = False,
    device: str | None = None,
    input_paths: Iterable[str | Path] = (),
) -> ResultPaths:
    """Fit, predict, and write one auditable Sentence-BERT result bundle."""
    if run_mode not in {"smoke_test", "final"}:
        raise ValueError("Sentence-BERT run_mode must be 'smoke_test' or 'final'")
    if run_mode == "final":
        require_gold(frame, context="Sentence-BERT final run")
        if synthetic or label_column != "human_lmp_score":
            raise ValueError(
                "Final mode requires synthetic=False and the human_lmp_score label"
            )
        data_source = "human_gold_corpus"
        research_use = "research"
    else:
        if not synthetic or label_column != "label":
            raise ValueError(
                "Smoke mode requires synthetic=True and the generic test-only 'label' column"
            )
        data_source = "synthetic"
        research_use = "non-research"

    if not isinstance(allow_download, bool):
        raise ValueError("allow_download must be a boolean")

    destination = Path(output_path)
    if destination.suffix.casefold() != ".csv" or not destination.name:
        raise ValueError("Sentence-BERT --output must be a .csv file path")
    destination, metadata_path = validate_prediction_output_paths(
        destination,
        cfg,
        model_name="Sentence-BERT",
        input_paths=input_paths,
    )

    seed = setup_model_seed(cfg)
    injected_encoder = encoder is not None
    model = SbertBaseline(
        cfg,
        encoder=encoder,
        device=device,
        allow_download=allow_download,
    ).fit(frame, label_column=label_column)
    predictions = model.predict(frame)
    predictions["run_mode"] = run_mode
    predictions["data_source"] = data_source
    predictions["research_use"] = research_use

    metadata = build_result_metadata(
        cfg,
        model_name="sbert",
        run_mode=run_mode,
        pair_ids=predictions["pair_id"].tolist(),
        splits=predictions["split"].tolist(),
        seed=seed,
        device=str(model.device),
    )
    if model.calibrator is None:  # defensive; fit() guarantees this
        raise RuntimeError("Sentence-BERT calibration was not fitted")
    metadata.update(
        {
            "synthetic": bool(synthetic),
            "smoke_test": run_mode == "smoke_test",
            "data_source": data_source,
            "research_use": research_use,
            "research_metrics_reported": False,
            "model_identity": {
                "library": "sentence-transformers",
                "model_name": model.settings.model_name,
                "revision": None,
            },
            "loading_policy": {
                "allow_download": bool(allow_download),
                "local_files_only": not bool(allow_download),
                "trust_remote_code": False,
            },
            "sbert_config": asdict(model.settings),
            "encoder_fit": {
                "fit_on_corpus": False,
                "fit_records": [],
            },
            "encoder_runtime": {
                "source": (
                    "injected_test_encoder"
                    if injected_encoder
                    else "configured_sentence_transformer"
                ),
                "configured_model_weights_used": not injected_encoder,
            },
            "calibration": model.calibrator.to_dict(),
            "score_range": [
                model.calibrator.score_min,
                model.calibrator.score_max,
            ],
            "explicit_clipping": model.settings.clip_to_score_range,
            "input_data_sha256": dataframe_sha256(frame),
        }
    )

    atomic_write_csv(predictions, destination)
    metadata["predictions_sha256"] = file_sha256(destination)
    atomic_write_json(metadata, metadata_path)
    return ResultPaths(destination, metadata_path)


def build_synthetic_smoke_fixture() -> pd.DataFrame:
    """Return a small in-memory fixture containing only generic test labels."""
    rows = [
        (
            "SBERT_SMOKE_T_IDENTICAL",
            "en",
            "contract",
            "sbert-smoke::1",
            "train",
            "The buyer shall pay within 30 days",
            "The buyer shall pay within 30 days",
            10.0,
        ),
        (
            "SBERT_SMOKE_T_PARTIAL",
            "en",
            "contract",
            "sbert-smoke::2",
            "train",
            "The seller may cancel the contract after written notice",
            "The seller may cancel after notice",
            5.0,
        ),
        (
            "SBERT_SMOKE_T_DIFFERENT",
            "hi",
            "contract",
            "sbert-smoke::3",
            "train",
            "क्रेता माल लौटाएगा",
            "न्यायालय अपील सुनेगा",
            1.0,
        ),
        (
            "SBERT_SMOKE_V_001",
            "en",
            "cyber",
            "sbert-smoke::4",
            "validation",
            "A password must remain confidential",
            "The password stays confidential",
            8.0,
        ),
        (
            "SBERT_SMOKE_X_001",
            "hi",
            "sale_of_goods",
            "sbert-smoke::5",
            "test",
            "विक्रेता माल देगा",
            "विक्रेता वस्तु देगा",
            8.0,
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "pair_id",
            "language",
            "legal_domain",
            "source_group",
            "split",
            "original_text",
            "simplified_text",
            "label",
        ],
    )


def main(argv: list[str] | None = None, *, encoder: Any | None = None) -> int:
    """Run the gold-gated final baseline or isolated synthetic smoke test."""
    parser = argparse.ArgumentParser(
        description="Run train-only-calibrated multilingual Sentence-BERT baseline."
    )
    parser.add_argument("--config", default=None, help="Path to project config YAML")
    parser.add_argument(
        "--output",
        default="results/sbert_predictions.csv",
        help="Prediction CSV path (metadata is written beside it)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use only the small in-memory synthetic non-research fixture",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Explicitly allow downloading configured Sentence-BERT weights",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_model_config(args.config)
        if args.smoke_test:
            frame = build_synthetic_smoke_fixture()
            paths = run_sbert_baseline(
                frame,
                cfg,
                output_path=args.output,
                label_column="label",
                run_mode="smoke_test",
                synthetic=True,
                encoder=encoder,
                allow_download=args.allow_download,
                input_paths=(args.config,) if args.config is not None else (),
            )
        else:
            # The authoritative loader requests only gold rows and applies the
            # gold assertion before model construction, weight loading, or IO.
            frame = load_model_corpus(config_path=args.config, mode="final")
            paths = run_sbert_baseline(
                frame,
                cfg,
                output_path=args.output,
                label_column="human_lmp_score",
                run_mode="final",
                synthetic=False,
                encoder=encoder,
                allow_download=args.allow_download,
                input_paths=(args.config,) if args.config is not None else (),
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.smoke_test:
        print(
            "Wrote smoke_test synthetic non-research Sentence-BERT predictions to "
            f"{paths.predictions_csv}"
        )
    else:
        print(f"Wrote final Sentence-BERT predictions to {paths.predictions_csv}")
    print(f"Wrote provenance metadata to {paths.metadata_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
