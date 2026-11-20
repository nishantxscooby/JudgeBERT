"""Leakage-safe word and character TF-IDF baseline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.preprocess import content_tokens, normalize_text
from src.models.calibration import LinearScoreCalibrator
from src.models.common import (
    load_model_config,
    load_model_corpus,
    require_gold,
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


@dataclass(frozen=True)
class TfidfConfig:
    """Validated TF-IDF settings read from the project configuration."""

    word_ngram_range: tuple[int, int]
    char_ngram_range: tuple[int, int]
    min_df: int | float
    sublinear_tf: bool
    use_char_ngrams: bool
    calibration: str
    clip_to_score_range: bool

    @classmethod
    def from_project_config(cls, cfg: Mapping[str, Any]) -> "TfidfConfig":
        """Build and validate settings from ``models.tfidf``."""
        models_cfg = cfg.get("models")
        if not isinstance(models_cfg, Mapping):
            raise ValueError("Project config is missing the 'models' mapping")
        model_cfg = models_cfg.get("tfidf")
        if not isinstance(model_cfg, Mapping):
            raise ValueError("Project config is missing the 'models.tfidf' mapping")
        required = {
            "enabled",
            "word_ngram_range",
            "char_ngram_range",
            "min_df",
            "sublinear_tf",
            "use_char_ngrams",
            "calibration",
            "clip_to_score_range",
        }
        missing = sorted(required - set(model_cfg))
        if missing:
            raise ValueError(f"models.tfidf is missing setting(s): {missing}")
        if model_cfg["enabled"] is not True:
            raise ValueError("models.tfidf.enabled must be true to run this baseline")

        def ngram_range(name: str) -> tuple[int, int]:
            value = model_cfg[name]
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(isinstance(part, bool) or not isinstance(part, int) for part in value)
                or value[0] < 1
                or value[0] > value[1]
            ):
                raise ValueError(
                    f"models.tfidf.{name} must be two increasing positive integers"
                )
            return int(value[0]), int(value[1])

        min_df = model_cfg["min_df"]
        valid_min_df = (
            isinstance(min_df, int)
            and not isinstance(min_df, bool)
            and min_df >= 1
        ) or (
            isinstance(min_df, float) and 0.0 < min_df <= 1.0
        )
        if not valid_min_df:
            raise ValueError(
                "models.tfidf.min_df must be an integer >= 1 or a float in (0, 1]"
            )
        for name in ("sublinear_tf", "use_char_ngrams", "clip_to_score_range"):
            if not isinstance(model_cfg[name], bool):
                raise ValueError(f"models.tfidf.{name} must be a boolean")
        if model_cfg["calibration"] != "linear_regression":
            raise ValueError(
                "models.tfidf.calibration must be 'linear_regression'"
            )

        return cls(
            word_ngram_range=ngram_range("word_ngram_range"),
            char_ngram_range=ngram_range("char_ngram_range"),
            min_df=min_df,
            sublinear_tf=model_cfg["sublinear_tf"],
            use_char_ngrams=model_cfg["use_char_ngrams"],
            calibration=model_cfg["calibration"],
            clip_to_score_range=model_cfg["clip_to_score_range"],
        )


class TfidfBaseline:
    """Word/character TF-IDF pair similarity with train-only calibration."""

    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.project_config = dict(cfg)
        self.settings = TfidfConfig.from_project_config(cfg)
        self.preprocessing_config = dict(cfg.get("preprocessing") or {})
        self.word_vectorizer: TfidfVectorizer | None = None
        self.char_vectorizer: TfidfVectorizer | None = None
        self.calibrator: LinearScoreCalibrator | None = None
        self.vocabulary_fit_pair_ids: tuple[str, ...] = ()

    @property
    def feature_branches(self) -> tuple[str, ...]:
        """Feature views enabled by the effective project configuration."""
        return ("word", "char") if self.settings.use_char_ngrams else ("word",)

    @property
    def word_vocabulary(self) -> frozenset[str]:
        """The fitted word features, exposed for leakage/provenance checks."""
        if self.word_vectorizer is None:
            raise RuntimeError("TF-IDF baseline must be fitted before reading vocabulary")
        return frozenset(self.word_vectorizer.vocabulary_)

    def _normalise(self, values: pd.Series) -> list[str]:
        return [
            normalize_text(value, self.preprocessing_config)
            for value in values.tolist()
        ]

    @staticmethod
    def _validate_text_columns(frame: pd.DataFrame, *, context: str) -> None:
        for column in ("original_text", "simplified_text"):
            values = frame[column]
            invalid = values.isna() | ~values.map(lambda value: isinstance(value, str))
            blank = values.astype("string").str.strip().eq("").fillna(True)
            bad = (invalid | blank).fillna(True)
            if bad.any():
                raise ValueError(
                    f"{context} {column} contains {int(bad.sum())} null or blank value(s)"
                )

    def fit(
        self, frame: pd.DataFrame, *, label_column: str = "label"
    ) -> "TfidfBaseline":
        """Fit both vocabularies and the score calibrator on train rows only."""
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
            raise ValueError(f"TF-IDF fit data is missing columns: {missing}")
        self._validate_text_columns(frame, context="TF-IDF fit data")
        train = frame.loc[frame["split"].astype(str).str.strip().eq("train")].copy()
        if train.empty:
            raise ValueError("TF-IDF fit requires at least one train row")

        original = self._normalise(train["original_text"])
        simplified = self._normalise(train["simplified_text"])
        fit_documents = original + simplified

        self.word_vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=content_tokens,
            token_pattern=None,
            lowercase=False,
            ngram_range=self.settings.word_ngram_range,
            min_df=self.settings.min_df,
            sublinear_tf=self.settings.sublinear_tf,
        )
        self.word_vectorizer.fit(fit_documents)

        if self.settings.use_char_ngrams:
            self.char_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                lowercase=False,
                ngram_range=self.settings.char_ngram_range,
                min_df=self.settings.min_df,
                sublinear_tf=self.settings.sublinear_tf,
            )
            self.char_vectorizer.fit(fit_documents)

        similarities = self._similarities_for_documents(original, simplified)
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
        self.vocabulary_fit_pair_ids = tuple(
            train["pair_id"].astype(str).str.strip().tolist()
        )
        return self

    @staticmethod
    def _paired_cosine(left: Any, right: Any) -> np.ndarray:
        values = np.asarray(left.multiply(right).sum(axis=1)).ravel()
        return np.clip(values.astype(float, copy=False), -1.0, 1.0)

    def _similarities_for_documents(
        self, original: list[str], simplified: list[str]
    ) -> pd.DataFrame:
        if self.word_vectorizer is None:
            raise RuntimeError("TF-IDF baseline must be fitted before prediction")
        original_word = self.word_vectorizer.transform(original)
        simplified_word = self.word_vectorizer.transform(simplified)
        word = self._paired_cosine(original_word, simplified_word)
        payload: dict[str, np.ndarray] = {"word_cosine_similarity": word}

        if self.settings.use_char_ngrams:
            if self.char_vectorizer is None:
                raise RuntimeError("Character TF-IDF vectorizer is not fitted")
            original_char = self.char_vectorizer.transform(original)
            simplified_char = self.char_vectorizer.transform(simplified)
            char = self._paired_cosine(original_char, simplified_char)
            payload["char_cosine_similarity"] = char
            payload["cosine_similarity"] = (word + char) / 2.0
        else:
            payload["cosine_similarity"] = word
        return pd.DataFrame(payload)

    def raw_similarities(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform original/simplified text separately and compare each row."""
        validate_model_corpus(frame)
        required = {"original_text", "simplified_text"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"TF-IDF prediction data is missing columns: {missing}")
        self._validate_text_columns(frame, context="TF-IDF prediction data")
        return self._similarities_for_documents(
            self._normalise(frame["original_text"]),
            self._normalise(frame["simplified_text"]),
        )

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return clipped calibrated scores while preserving input row order."""
        required = {"pair_id", "language", "legal_domain", "split"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"TF-IDF prediction data is missing columns: {missing}")
        if self.calibrator is None:
            raise RuntimeError("TF-IDF baseline must be fitted before prediction")

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


def run_tfidf_baseline(
    frame: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    output_path: str | Path,
    label_column: str,
    run_mode: str,
    synthetic: bool,
    input_paths: Iterable[str | Path] = (),
) -> ResultPaths:
    """Fit, predict, and write one auditable TF-IDF result bundle."""
    if run_mode not in {"smoke_test", "final"}:
        raise ValueError("TF-IDF run_mode must be 'smoke_test' or 'final'")
    if run_mode == "smoke_test":
        if not synthetic or label_column != "label":
            raise ValueError(
                "Smoke mode requires synthetic=True and the generic test-only 'label' column"
            )
        data_source = "synthetic"
        research_use = "non-research"
    else:
        if synthetic or label_column != "human_lmp_score":
            raise ValueError(
                "Final mode requires synthetic=False and the human_lmp_score label"
            )
        require_gold(frame, context="TF-IDF final run")
        data_source = "human_gold_corpus"
        research_use = "research"

    destination = Path(output_path)
    if destination.suffix.casefold() != ".csv" or not destination.name:
        raise ValueError("TF-IDF --output must be a .csv file path")
    destination, metadata_path = validate_prediction_output_paths(
        destination,
        cfg,
        model_name="TF-IDF",
        input_paths=input_paths,
    )

    seed = setup_model_seed(cfg)
    model = TfidfBaseline(cfg).fit(frame, label_column=label_column)
    predictions = model.predict(frame)
    predictions["run_mode"] = run_mode
    predictions["data_source"] = data_source
    predictions["research_use"] = research_use

    metadata = build_result_metadata(
        cfg,
        model_name="tfidf",
        run_mode=run_mode,
        pair_ids=predictions["pair_id"].tolist(),
        splits=predictions["split"].tolist(),
        seed=seed,
        device="cpu",
    )
    if model.calibrator is None:  # defensive; fit() guarantees this
        raise RuntimeError("TF-IDF calibration was not fitted")
    metadata.update(
        {
            "synthetic": bool(synthetic),
            "smoke_test": run_mode == "smoke_test",
            "data_source": data_source,
            "research_use": research_use,
            "research_metrics_reported": False,
            "tfidf_config": asdict(model.settings),
            "feature_branches": list(model.feature_branches),
            "word_analyzer": "content_tokens",
            "char_analyzer": "char_wb" if model.settings.use_char_ngrams else None,
            "similarity_aggregation": (
                "mean_word_char"
                if model.settings.use_char_ngrams
                else "word_only"
            ),
            "vocabulary_fit": {
                "splits": ["train"],
                "pair_ids": list(model.vocabulary_fit_pair_ids),
                "documents_per_pair": 2,
                "word_vocabulary_size": len(model.word_vocabulary),
                "char_vocabulary_size": (
                    len(model.char_vectorizer.vocabulary_)
                    if model.char_vectorizer is not None
                    else 0
                ),
            },
            "calibration": model.calibrator.to_dict(),
            "score_range": [model.calibrator.score_min, model.calibrator.score_max],
            "explicit_clipping": model.settings.clip_to_score_range,
            "input_data_sha256": dataframe_sha256(frame),
        }
    )

    atomic_write_csv(predictions, destination)
    metadata["predictions_sha256"] = file_sha256(destination)
    atomic_write_json(metadata, metadata_path)
    return ResultPaths(destination, metadata_path)


def build_synthetic_smoke_fixture() -> pd.DataFrame:
    """Return a tiny in-memory, non-research fixture with generic labels only."""
    rows = [
        (
            "SMOKE_T_IDENTICAL",
            "en",
            "contract",
            "smoke::1",
            "train",
            "The buyer shall pay within 30 days",
            "The buyer shall pay within 30 days",
            10.0,
        ),
        (
            "SMOKE_T_PARTIAL",
            "en",
            "contract",
            "smoke::2",
            "train",
            "The seller may cancel the contract after written notice",
            "The seller may cancel",
            5.0,
        ),
        (
            "SMOKE_T_DIFFERENT",
            "hi",
            "contract",
            "smoke::3",
            "train",
            "क्रेता माल लौटाएगा",
            "न्यायालय अपील सुनेगा",
            1.0,
        ),
        (
            "SMOKE_V_001",
            "en",
            "cyber",
            "smoke::4",
            "validation",
            "A password must remain confidential",
            "The password stays confidential",
            8.0,
        ),
        (
            "SMOKE_X_001",
            "hi",
            "sale_of_goods",
            "smoke::5",
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


def main(argv: list[str] | None = None) -> int:
    """Run the blocked final baseline or the isolated synthetic smoke test."""
    parser = argparse.ArgumentParser(
        description="Run train-only-calibrated word/character TF-IDF baseline."
    )
    parser.add_argument("--config", default=None, help="Path to project config YAML")
    parser.add_argument(
        "--output",
        default="results/tfidf_predictions.csv",
        help="Prediction CSV path (metadata is written beside it)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use only the small in-memory synthetic non-research fixture",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_model_config(args.config)
        if args.smoke_test:
            frame = build_synthetic_smoke_fixture()
            paths = run_tfidf_baseline(
                frame,
                cfg,
                output_path=args.output,
                label_column="label",
                run_mode="smoke_test",
                synthetic=True,
                input_paths=(args.config,) if args.config is not None else (),
            )
        else:
            # This authoritative path calls load_master(subset='gold') and
            # assert_gold before fitting or creating any output.
            frame = load_model_corpus(config_path=args.config, mode="final")
            paths = run_tfidf_baseline(
                frame,
                cfg,
                output_path=args.output,
                label_column="human_lmp_score",
                run_mode="final",
                synthetic=False,
                input_paths=(args.config,) if args.config is not None else (),
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.smoke_test:
        print(
            "Wrote smoke_test synthetic non-research TF-IDF predictions to "
            f"{paths.predictions_csv}"
        )
    else:
        print(f"Wrote final TF-IDF predictions to {paths.predictions_csv}")
    print(f"Wrote provenance metadata to {paths.metadata_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
