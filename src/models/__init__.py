"""Shared, model-agnostic foundations for JudgeBERT experiments."""

from src.models.calibration import CalibrationFitRecord, LinearScoreCalibrator
from src.models.common import (
    DeviceName,
    ModelRunContext,
    RunMode,
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
    ResultPaths,
    build_result_metadata,
    dataframe_sha256,
    ensure_output_directory,
    file_sha256,
    source_tree_sha256,
    validate_prediction_output_paths,
    write_result_bundle,
)

__all__ = [
    "CalibrationFitRecord",
    "DeviceName",
    "LinearScoreCalibrator",
    "ModelRunContext",
    "ResultPaths",
    "RunMode",
    "build_result_metadata",
    "dataframe_sha256",
    "ensure_output_directory",
    "file_sha256",
    "load_model_config",
    "load_model_corpus",
    "prepare_model_run",
    "require_gold",
    "select_device",
    "setup_model_seed",
    "source_tree_sha256",
    "validate_prediction_output_paths",
    "validate_model_corpus",
    "validate_predictions",
    "write_result_bundle",
]
