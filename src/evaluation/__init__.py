"""Gold-safe metrics, validation, and final evaluation entry points."""

from __future__ import annotations

from typing import Any

__all__ = ["evaluate_prediction_files"]


def __getattr__(name: str) -> Any:
    """Load the CLI-bearing evaluator lazily so ``python -m`` stays clean."""
    if name == "evaluate_prediction_files":
        from src.evaluation.evaluate import evaluate_prediction_files

        return evaluate_prediction_files
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

