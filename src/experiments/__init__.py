"""Experiment harness for Phase 6 (Experiments 1–5).

This module provides the execution framework for evaluating baselines,
cross-lingual performance, domain generalisation, behavioral sanity checks,
and qualitative error analyses.

Strict Data Integrity:
    Experiments 1, 2, 3, and 5 evaluate models against authoritative human gold
    ground truth. When human annotations are pending, final execution is blocked
    with 'Blocked: TODO_HUMAN_ANNOTATION'.
    Experiment 4 evaluates model behavior on controlled synthetic sanity checks
    and does not claim human gold evaluation.
"""

from __future__ import annotations

from src.experiments.common import (
    ExperimentBlockedError,
    build_experiment_provenance,
    load_experiment_config,
    load_gold_or_block,
)

__all__ = [
    "ExperimentBlockedError",
    "build_experiment_provenance",
    "load_experiment_config",
    "load_gold_or_block",
]
