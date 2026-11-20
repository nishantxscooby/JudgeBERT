"""Master orchestrator for Phase 6 experiments (Experiments 1–5).

Executes experiments in dependency order:
1. Experiment 1: Baselines vs JUDGEBERT
2. Experiment 2: Cross-lingual Comparison (English vs Hindi)
3. Experiment 3: Held-Out Domain Generalisation
4. Experiment 4: Behavioral Sanity Checks
5. Experiment 5: Qualitative Error Analysis

Integrity and Status Handling:
- Evaluates against authoritative human gold annotations where required (Exp 1, 2, 3, 5).
- Clearly records BLOCKED status with 'TODO_HUMAN_ANNOTATION' when human gold is absent.
- Never reports blocked experiments as successful.
- Preserves individual experiment provenance and writes summary manifests atomically.
- Allows running individual experiments via `--experiments`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.experiments.common import (
    ExperimentBlockedError,
    build_experiment_provenance,
    ensure_output_directory,
    load_experiment_config,
)
from src.experiments.experiment1_baselines import run_experiment1
from src.experiments.experiment2_language import run_experiment2
from src.experiments.experiment3_domain_generalization import run_experiment3
from src.experiments.experiment4_sanity_checks import run_experiment4
from src.experiments.experiment5_error_analysis import run_experiment5
from src.models.provenance import atomic_write_json
from src.utils.io import TODO_HUMAN_ANNOTATION

EXPERIMENT_REGISTRY: dict[str, tuple[str, Callable[..., dict[str, Any]]]] = {
    "experiment1": ("Experiment 1 — Baselines vs JUDGEBERT", run_experiment1),
    "experiment2": ("Experiment 2 — Cross-lingual Comparison", run_experiment2),
    "experiment3": ("Experiment 3 — Domain Generalisation", run_experiment3),
    "experiment4": ("Experiment 4 — Behavioral Sanity Checks", run_experiment4),
    "experiment5": ("Experiment 5 — Qualitative Error Analysis", run_experiment5),
}


def _normalize_experiment_key(key: str) -> str:
    cleaned = key.strip().lower()
    if cleaned in EXPERIMENT_REGISTRY:
        return cleaned
    cleaned_num = cleaned.replace("experiment", "").replace("exp", "").strip()
    candidate = f"experiment{cleaned_num}"
    if candidate in EXPERIMENT_REGISTRY:
        return candidate
    raise ValueError(
        f"Unknown experiment {key!r}. Valid options: {list(EXPERIMENT_REGISTRY.keys())} or 1..5."
    )


def _generate_summary_markdown(
    results: Mapping[str, dict[str, Any]],
    is_smoke: bool = False,
) -> str:
    lines = [
        "# Phase 6 Experiments — Orchestration Summary",
        "",
    ]
    if is_smoke:
        lines.extend([
            "> [!WARNING]",
            "> **NON-RESEARCH SMOKE TEST RUN.**",
            "> Executed with synthetic fixtures to verify end-to-end pipeline integrity.",
            "",
        ])

    lines.extend([
        "| Experiment | Key | Status | Details / Block Reason |",
        "| :--- | :---: | :---: | :--- |",
    ])

    for key, info in results.items():
        title = info.get("title", key)
        status = info.get("status", "UNKNOWN")
        detail = info.get("detail", "")
        status_badge = f"`{status}`"
        lines.append(f"| {title} | `{key}` | {status_badge} | {detail} |")

    lines.append("")
    return "\n".join(lines)


def run_all_experiments(
    cfg: Mapping[str, Any] | None = None,
    output_dir: str | Path = "results/experiments",
    selected_experiments: Sequence[str] | None = None,
    *,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Execute all or selected experiments in sequence.

    Args:
        cfg: Project configuration mapping.
        output_dir: Root output directory for experiments.
        selected_experiments: Optional list of experiment identifiers (e.g. ['1', '4']).
        smoke_test: If True, executes all experiments in smoke test mode.

    Returns:
        Structured summary dictionary of all experiment runs.
    """
    effective_cfg = dict(cfg) if cfg is not None else load_experiment_config()
    out_root = ensure_output_directory(output_dir)

    # Determine which experiments to run
    if selected_experiments and "all" not in [x.lower() for x in selected_experiments]:
        keys_to_run = [_normalize_experiment_key(k) for k in selected_experiments]
    else:
        keys_to_run = list(EXPERIMENT_REGISTRY.keys())

    summary_results: dict[str, dict[str, Any]] = {}
    has_unexpected_failure = False

    for exp_key in keys_to_run:
        title, runner_func = EXPERIMENT_REGISTRY[exp_key]
        exp_out_dir = out_root / exp_key

        try:
            res = runner_func(
                cfg=effective_cfg,
                output_dir=exp_out_dir,
                smoke_test=smoke_test,
            )
            status = "SUCCESS_SMOKE" if smoke_test else "SUCCESS"
            summary_results[exp_key] = {
                "title": title,
                "status": status,
                "output_dir": str(exp_out_dir),
                "detail": f"Completed successfully ({status}).",
            }
        except ExperimentBlockedError as exc:
            summary_results[exp_key] = {
                "title": title,
                "status": "BLOCKED",
                "output_dir": str(exp_out_dir),
                "detail": f"Blocked: {TODO_HUMAN_ANNOTATION} (Human gold annotations pending)",
            }
        except Exception as exc:
            has_unexpected_failure = True
            summary_results[exp_key] = {
                "title": title,
                "status": "FAILED",
                "output_dir": str(exp_out_dir),
                "detail": f"Unexpected error: {exc}",
            }

    overall_provenance = build_experiment_provenance(
        effective_cfg,
        experiment_name="Phase 6 — Experiments Suite",
        model_identity="orchestrator",
        is_smoke=smoke_test,
    )

    summary_payload = {
        "suite_name": "Phase 6 Experiments",
        "smoke_test": smoke_test,
        "executed_experiments": keys_to_run,
        "summary": summary_results,
        "provenance": overall_provenance,
    }

    # Write summary manifest and report atomically
    atomic_write_json(summary_payload, out_root / "summary_manifest.json")

    summary_md = _generate_summary_markdown(summary_results, is_smoke=smoke_test)
    report_file = out_root / "experiments_summary.md"
    temp_report = report_file.with_name(f".{report_file.name}.tmp")
    temp_report.write_text(summary_md, encoding="utf-8")
    temp_report.replace(report_file)

    if has_unexpected_failure:
        raise RuntimeError("One or more experiments suffered an unexpected failure.")

    return summary_payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for master experiment orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run Phase 6 Experiments (1–5) in dependency order."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="results/experiments", help="Output directory")
    parser.add_argument(
        "--experiments",
        default="all",
        help="Comma-separated experiment keys to run (e.g. '1,4' or 'all')",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run orchestration smoke test across selected experiments",
    )
    args = parser.parse_args(argv)

    selected = [x.strip() for x in args.experiments.split(",") if x.strip()]

    try:
        cfg = load_experiment_config(args.config)
        summary = run_all_experiments(
            cfg=cfg,
            output_dir=args.output_dir,
            selected_experiments=selected,
            smoke_test=args.smoke_test,
        )

        print("==================================================")
        print("EXPERIMENT ORCHESTRATION SUMMARY")
        print("==================================================")
        for key, info in summary["summary"].items():
            print(f"- {info['title']}: [{info['status']}] {info['detail']}")
        print(f"\nManifest written to: {Path(args.output_dir) / 'summary_manifest.json'}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
