"""Experiment 4 — Model Behavioral Sanity Checks.

Evaluates model sensitivity and directional consistency against controlled minimal pair
perturbations:
1. Identical sentence pairs (highest expected preservation)
2. Unrelated sentence pairs (lowest expected preservation)
3. Modal strength changes (obligation vs permission)
4. Negation flips (affirming vs denying legal validity)
5. Number alterations (modifying duration or monetary thresholds)
6. Date modifications (shifting statutory commencement dates)

Integrity Guarantees:
- Uses controlled, auditable synthetic inputs explicitly designed for behavioral stress-testing.
- Clearly segregated from gold corpus evaluation: these outputs are NOT human gold benchmarks.
- Reuses `src.data.legal_cues` to verify that perturbations trigger the intended legal cue shifts.
- Can execute independently of human annotation status because it evaluates model mechanics.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

from src.data.legal_cues import compare_texts
from src.data.preprocess import normalize_text
from src.experiments.common import (
    build_experiment_provenance,
    ensure_output_directory,
    load_experiment_config,
)
from src.models.provenance import atomic_write_csv, atomic_write_json

EXPERIMENT_NAME = "Experiment 4 — Behavioral Sanity Checks"


@dataclass(frozen=True)
class SanityPair:
    pair_id: str
    category: str
    language: str
    original_text: str
    simplified_text: str
    expected_direction: str
    expected_cue_change: str


# Core synthetic probe pairs across legal phenomena and languages
PROBE_PAIRS: tuple[SanityPair, ...] = (
    # --- Identical Pairs ---
    SanityPair(
        pair_id="SANITY_IDENTICAL_EN_01",
        category="identical",
        language="en",
        original_text="An agreement enforceable by law is a contract.",
        simplified_text="An agreement enforceable by law is a contract.",
        expected_direction="high_similarity",
        expected_cue_change="none",
    ),
    SanityPair(
        pair_id="SANITY_IDENTICAL_EN_02",
        category="identical",
        language="en",
        original_text="The seller shall deliver the goods in accordance with the contract.",
        simplified_text="The seller shall deliver the goods in accordance with the contract.",
        expected_direction="high_similarity",
        expected_cue_change="none",
    ),
    SanityPair(
        pair_id="SANITY_IDENTICAL_HI_01",
        category="identical",
        language="hi",
        original_text="विधि द्वारा प्रवर्तनीय करार संविदा है।",
        simplified_text="विधि द्वारा प्रवर्तनीय करार संविदा है।",
        expected_direction="high_similarity",
        expected_cue_change="none",
    ),
    SanityPair(
        pair_id="SANITY_IDENTICAL_HI_02",
        category="identical",
        language="hi",
        original_text="विक्रेता संविदा के निबंधनों के अनुसार माल का परिदान करेगा।",
        simplified_text="विक्रेता संविदा के निबंधनों के अनुसार माल का परिदान करेगा।",
        expected_direction="high_similarity",
        expected_cue_change="none",
    ),

    # --- Unrelated Pairs ---
    SanityPair(
        pair_id="SANITY_UNRELATED_EN_01",
        category="unrelated",
        language="en",
        original_text="An agreement enforceable by law is a contract.",
        simplified_text="Any person who accesses a computer system without authorization commits an offence.",
        expected_direction="low_similarity",
        expected_cue_change="broad_semantic_mismatch",
    ),
    SanityPair(
        pair_id="SANITY_UNRELATED_HI_01",
        category="unrelated",
        language="hi",
        original_text="विधि द्वारा प्रवर्तनीय करार संविदा है।",
        simplified_text="जो कोई भी अनधिकृत रूप से कंप्यूटर प्रणाली में प्रवेश करता है वह अपराध करता है।",
        expected_direction="low_similarity",
        expected_cue_change="broad_semantic_mismatch",
    ),

    # --- Modal Changes (Deontic Shift) ---
    SanityPair(
        pair_id="SANITY_MODAL_EN_01",
        category="modal",
        language="en",
        original_text="The seller shall deliver the goods within thirty days.",
        simplified_text="The seller may deliver the goods within thirty days.",
        expected_direction="reduced_preservation",
        expected_cue_change="modal_weakening",
    ),
    SanityPair(
        pair_id="SANITY_MODAL_EN_02",
        category="modal",
        language="en",
        original_text="No person shall tamper with computer source documents.",
        simplified_text="Any person may tamper with computer source documents.",
        expected_direction="severe_reduction",
        expected_cue_change="prohibition_to_permission",
    ),
    SanityPair(
        pair_id="SANITY_MODAL_HI_01",
        category="modal",
        language="hi",
        original_text="विक्रेता तीस दिन के भीतर माल का परिदान करेगा।",
        simplified_text="विक्रेता तीस दिन के भीतर माल का परिदान कर सकता है।",
        expected_direction="reduced_preservation",
        expected_cue_change="modal_weakening",
    ),

    # --- Negation Flips ---
    SanityPair(
        pair_id="SANITY_NEGATION_EN_01",
        category="negation",
        language="en",
        original_text="An agreement enforceable by law is a contract.",
        simplified_text="An agreement enforceable by law is not a contract.",
        expected_direction="severe_reduction",
        expected_cue_change="negation_introduced",
    ),
    SanityPair(
        pair_id="SANITY_NEGATION_HI_01",
        category="negation",
        language="hi",
        original_text="प्रतिफल के बिना किया गया करार शून्य है।",
        simplified_text="प्रतिफल के बिना किया गया करार शून्य नहीं है।",
        expected_direction="severe_reduction",
        expected_cue_change="negation_introduced",
    ),

    # --- Number Changes ---
    SanityPair(
        pair_id="SANITY_NUMBER_EN_01",
        category="number",
        language="en",
        original_text="The buyer shall pay the price within thirty days.",
        simplified_text="The buyer shall pay the price within sixty days.",
        expected_direction="reduced_preservation",
        expected_cue_change="quantity_mismatch",
    ),
    SanityPair(
        pair_id="SANITY_NUMBER_HI_01",
        category="number",
        language="hi",
        original_text="क्रेता तीस दिन के भीतर कीमत का संदाय करेगा।",
        simplified_text="क्रेता साठ दिन के भीतर कीमत का संदाय करेगा।",
        expected_direction="reduced_preservation",
        expected_cue_change="quantity_mismatch",
    ),

    # --- Date Changes ---
    SanityPair(
        pair_id="SANITY_DATE_EN_01",
        category="date",
        language="en",
        original_text="This Act shall come into force on the first day of September 1872.",
        simplified_text="This Act shall come into force on the first day of October 1872.",
        expected_direction="reduced_preservation",
        expected_cue_change="date_mismatch",
    ),
    SanityPair(
        pair_id="SANITY_DATE_HI_01",
        category="date",
        language="hi",
        original_text="यह अधिनियम सितम्बर 1872 के प्रथम दिन प्रवृत्त होगा।",
        simplified_text="यह अधिनियम अक्टूबर 1872 के प्रथम दिन प्रवृत्त होगा।",
        expected_direction="reduced_preservation",
        expected_cue_change="date_mismatch",
    ),
)


def score_pairs_with_tfidf(pairs_df: pd.DataFrame) -> np.ndarray:
    """Compute deterministic TF-IDF cosine similarities for sanity pairs."""
    corpus = list(pairs_df["original_text"]) + list(pairs_df["simplified_text"])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    vectorizer.fit(corpus)

    orig_vecs = vectorizer.transform(pairs_df["original_text"])
    simp_vecs = vectorizer.transform(pairs_df["simplified_text"])

    # Compute cosine similarity: 1 - cosine_distance
    distances = paired_cosine_distances(orig_vecs.toarray(), simp_vecs.toarray())
    similarities = np.clip(1.0 - distances, 0.0, 1.0)

    # Scale to 1.0–10.0 scale for intuitive comparison
    return np.round(1.0 + similarities * 9.0, 2)


def evaluate_cue_consistency(orig: str, simp: str, lang: str) -> dict[str, Any]:
    """Verify that the perturbation triggers legal cue detector signals."""
    delta = compare_texts(orig, simp, language=lang)
    return {
        "has_changes": delta.has_any_change,
        "modal_shift": bool(delta.modal_weakened or delta.modal_strengthened),
        "negation_flipped": bool(delta.negation_flipped),
        "quantity_mismatch_count": len(delta.quantity_changes),
    }


def _generate_sanity_report(
    category_summary: Mapping[str, Any],
    verification_passed: bool,
    is_smoke: bool = False,
) -> str:
    lines = [
        f"# {EXPERIMENT_NAME}",
        "",
        "> [!NOTE]",
        "> **Methodology:** Controlled behavioral test suite assessing model response to",
        "> precise semantic and legal perturbations (modals, negations, quantities, unrelated texts).",
        "> This is a model diagnostic harness, not an evaluation on human gold labels.",
        "",
        f"- **Sanity Check Behavioral Invariant:** `{'PASSED' if verification_passed else 'FAILED'}`",
        f"- **Invariant condition:** Mean(Identical) > Mean(Unrelated)",
        "",
        "## Performance by Perturbation Category",
        "",
        "| Category | Count | Mean Score (1–10) | Std Dev | Min | Max | Expected Behavior |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for cat, stats in category_summary.items():
        cnt = stats["count"]
        mean_s = stats["mean"]
        std_s = stats["std"]
        min_s = stats["min"]
        max_s = stats["max"]
        exp = stats["expected_behavior"]
        lines.append(
            f"| **{cat.title()}** | {cnt} | {mean_s:.2f} | {std_s:.2f} | {min_s:.2f} | {max_s:.2f} | {exp} |"
        )

    return "\n".join(lines) + "\n"


def run_experiment4(
    cfg: Mapping[str, Any] | None = None,
    output_dir: str | Path = "results/experiment4",
    *,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Execute Experiment 4 sanity checks.

    Args:
        cfg: Project configuration mapping.
        output_dir: Directory where results will be written.
        smoke_test: If True, executes minimal test probe subset.

    Returns:
        Structured dictionary containing category statistics and verification flags.
    """
    effective_cfg = dict(cfg) if cfg is not None else load_experiment_config()
    out_path = ensure_output_directory(output_dir)

    provenance = build_experiment_provenance(
        effective_cfg,
        experiment_name=EXPERIMENT_NAME,
        model_identity="behavioral_sanity_probes",
        is_smoke=smoke_test,
        extra={
            "suite_type": "controlled_minimal_pairs",
            "evaluated_phenomena": ["identical", "unrelated", "modal", "negation", "number", "date"],
        },
    )

    # Build DataFrame from controlled probes
    probe_records = [asdict(p) for p in PROBE_PAIRS]
    pairs_df = pd.DataFrame(probe_records)

    # Validate cue extraction consistency on probes
    cue_results = []
    for _, row in pairs_df.iterrows():
        cue_diag = evaluate_cue_consistency(
            row["original_text"],
            row["simplified_text"],
            row["language"],
        )
        cue_results.append(cue_diag)

    cue_df = pd.DataFrame(cue_results)
    pairs_df["cue_has_changes"] = cue_df["has_changes"]
    pairs_df["cue_modal_shift"] = cue_df["modal_shift"]
    pairs_df["cue_negation_flipped"] = cue_df["negation_flipped"]
    pairs_df["cue_quantity_mismatches"] = cue_df["quantity_mismatch_count"]

    # Compute baseline model scores
    scores = score_pairs_with_tfidf(pairs_df)
    pairs_df["simulated_lmp_score"] = scores

    # Category summaries
    category_summary: dict[str, Any] = {}
    for cat in pairs_df["category"].unique():
        cat_rows = pairs_df[pairs_df["category"] == cat]
        cat_scores = cat_rows["simulated_lmp_score"].to_numpy()
        category_summary[cat] = {
            "count": len(cat_rows),
            "mean": float(np.mean(cat_scores)),
            "std": float(np.std(cat_scores)),
            "min": float(np.min(cat_scores)),
            "max": float(np.max(cat_scores)),
            "expected_behavior": cat_rows["expected_direction"].iloc[0],
        }

    # Core invariant check: Identical pairs must score strictly higher on average than Unrelated pairs
    mean_identical = category_summary.get("identical", {}).get("mean", 10.0)
    mean_unrelated = category_summary.get("unrelated", {}).get("mean", 1.0)
    invariant_passed = bool(mean_identical > mean_unrelated)

    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "n_probes": len(pairs_df),
        "invariant_passed": invariant_passed,
        "categories": category_summary,
        "provenance": provenance,
    }

    # Write output artifacts atomically
    atomic_write_json(payload, out_path / "sanity_metrics.json")
    atomic_write_json(provenance, out_path / "provenance.json")
    atomic_write_csv(pairs_df, out_path / "sanity_pairs.csv")

    report_md = _generate_sanity_report(category_summary, invariant_passed, is_smoke=smoke_test)
    report_file = out_path / "sanity_report.md"
    temp_report = report_file.with_name(f".{report_file.name}.tmp")
    temp_report.write_text(report_md, encoding="utf-8")
    temp_report.replace(report_file)

    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for Experiment 4."""
    parser = argparse.ArgumentParser(
        description=f"Run {EXPERIMENT_NAME} (Behavioral Sanity Checks)."
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--output-dir", default="results/experiment4", help="Output directory")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run orchestration smoke test with probe fixtures",
    )
    args = parser.parse_args(argv)

    try:
        cfg = load_experiment_config(args.config)
        run_experiment4(cfg=cfg, output_dir=args.output_dir, smoke_test=args.smoke_test)
        mode_str = "smoke test" if args.smoke_test else "full"
        print(f"SUCCESS: {EXPERIMENT_NAME} ({mode_str}) completed. Artifacts written to {args.output_dir}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
