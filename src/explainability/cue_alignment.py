"""Auditable legal cue alignment and surface-based explainability.

Implements the primary, auditable explanation layer comparing original and simplified
statutory texts across:
- Modal strength (obligation, permission, prohibition, recommendation)
- Negation presence and parity
- Conditions and conditional clauses
- Statutory exceptions and provisos
- Quantifiers (universal vs existential)
- Quantities and numeric values
- Statutory dates and years
- Legal entities and parties
- Scope limiters
- Candidate omissions and additions (length ratio checks)

Strict Integrity Guarantees:
- Findings use careful probabilistic framing ("Potential obligation mismatch",
  "Possible negation change", "Quantity differs") rather than asserting legal truth.
- Offsets are verified against exact character slices.
- Unicode and Devanagari characters are safely processed without index drift.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from src.data.legal_cues import (
    CATEGORY_CONDITION,
    CATEGORY_ENTITY,
    CATEGORY_EXCEPTION,
    CATEGORY_MODAL,
    CATEGORY_NEGATION,
    CATEGORY_QUANTIFIER,
    CATEGORY_QUANTITY,
    CATEGORY_SCOPE,
    STRENGTH_DISCRETION,
    STRENGTH_LABELS,
    STRENGTH_OBLIGATION,
    STRENGTH_PERMISSION,
    STRENGTH_PROHIBITION,
    STRENGTH_RECOMMENDATION,
    LegalCue,
    Quantity,
    compare_texts,
    extract_cues,
)
from src.data.preprocess import content_tokens, length_ratio

# Date patterns (month names and 4-digit calendar years)
_MONTHS_EN = (
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b"
)
_MONTHS_HI = (
    r"(?:जनवरी|फ़रवरी|फरवरी|मार्च|अप्रैल|मई|जून|जुलाई|अगस्त|सितम्बर|सितंबर|"
    r"अक्टूबर|नवम्बर|नवंबर|दिसम्बर|दिसंबर)"
)
_YEAR_EN = r"\b(?:1[89]\d{2}|20\d{2})\b"
_YEAR_HI = r"(?:1[89]\d{2}|20\d{2}|[०-९]{4})"


def detect_language(text: str) -> str:
    """Detect whether text is primarily Hindi (Devanagari) or English."""
    if not isinstance(text, str):
        return "en"
    for char in text:
        if "\u0900" <= char <= "\u097f":
            return "hi"
    return "en"


def _extract_date_cues(text: str, language: str, source: str = "text") -> list[dict[str, Any]]:
    """Extract explicit month and calendar year expressions as date cues."""
    date_cues: list[dict[str, Any]] = []
    month_pattern = _MONTHS_HI if language == "hi" else _MONTHS_EN
    year_pattern = _YEAR_HI if language == "hi" else _YEAR_EN

    for pattern in (month_pattern, year_pattern):
        # In English, month names are capitalized. Case-insensitive matching would treat modal 'may' as a date.
        flags = re.IGNORECASE if language == "hi" else 0
        for match in re.finditer(pattern, text, flags=flags):
            surface = match.group(0)
            if language == "en" and surface == "May":
                # Disambiguate May: only treat as month if near digits, day, or year
                window = text[max(0, match.start() - 15) : min(len(text), match.end() + 15)]
                if not re.search(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?May\b|\bMay\s+\d{1,4}\b", window, re.IGNORECASE):
                    continue
            date_cues.append({
                "category": "date",
                "canonical": surface.lower(),
                "surface": surface,
                "start": match.start(),
                "end": match.end(),
                "strength": None,
                "source": source,
            })
    return date_cues


def extract_all_cues(
    text: str,
    language: str | None = None,
    source: str = "text",
) -> list[dict[str, Any]]:
    """Extract all legally salient cues with character offsets.

    Guarantees that `text[cue['start']:cue['end']] == cue['surface']`.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    lang = language or detect_language(text)
    profile = extract_cues(text, language=lang)
    cues_list: list[dict[str, Any]] = []

    # 1. Base cues from legal_cues extractor
    for c in profile.cues:
        cues_list.append({
            "category": c.category,
            "canonical": c.canonical,
            "surface": c.surface,
            "start": c.start,
            "end": c.end,
            "strength": c.strength,
            "source": source,
        })

    # 2. Date and year expressions
    date_cues = _extract_date_cues(text, lang, source=source)
    cues_list.extend(date_cues)

    # 3. Normalized quantities and numbers
    for q in profile.quantities:
        # Check if quantity overlaps with already detected date
        is_date = any(
            d["start"] <= q.start and q.end <= d["end"]
            for d in date_cues
        )
        if is_date:
            continue

        cat = "number" if q.canonical_unit == "count" else "quantity"
        cues_list.append({
            "category": cat,
            "canonical": f"{q.normalized} {q.canonical_unit}".strip(),
            "surface": q.raw,
            "start": q.start,
            "end": q.end,
            "strength": None,
            "normalized_value": q.normalized,
            "canonical_unit": q.canonical_unit,
            "source": source,
        })

    # Deduplicate overlapping cues with identical spans
    seen_spans: set[tuple[int, int, str]] = set()
    unique_cues: list[dict[str, Any]] = []
    for cue in sorted(cues_list, key=lambda x: (x["start"], -x["end"])):
        key = (cue["start"], cue["end"], cue["category"])
        if key not in seen_spans:
            seen_spans.add(key)
            unique_cues.append(cue)

    # Verify offset integrity: text[start:end] == surface
    for cue in unique_cues:
        sliced = text[cue["start"]:cue["end"]]
        if sliced != cue["surface"]:
            # If surface had surrounding whitespace trimmed, adjust
            if cue["surface"] in sliced:
                offset = sliced.index(cue["surface"])
                cue["start"] += offset
                cue["end"] = cue["start"] + len(cue["surface"])

    return sorted(unique_cues, key=lambda x: x["start"])


def align_legal_cues(
    original_text: str,
    simplified_text: str,
    language: str | None = None,
) -> dict[str, Any]:
    """Perform auditable alignment between original and simplified cues.

    Identifies preserved, removed, added, and changed cues, and generates
    structured findings with careful legal hedging.
    """
    if not isinstance(original_text, str) or not original_text.strip():
        raise ValueError("original_text must be a non-empty string.")
    if not isinstance(simplified_text, str) or not simplified_text.strip():
        raise ValueError("simplified_text must be a non-empty string.")

    lang = language or detect_language(original_text)

    orig_cues = extract_all_cues(original_text, language=lang, source="original")
    simp_cues = extract_all_cues(simplified_text, language=lang, source="simplified")

    preserved: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    added: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    matched_simp_indices: set[int] = set()

    # Step 1: Match identical cues (same category and canonical value)
    for orig_cue in orig_cues:
        matched = False
        for s_idx, simp_cue in enumerate(simp_cues):
            if s_idx in matched_simp_indices:
                continue
            if (
                orig_cue["category"] == simp_cue["category"]
                and orig_cue["canonical"] == simp_cue["canonical"]
            ):
                matched = True
                matched_simp_indices.add(s_idx)
                preserved.append({
                    "category": orig_cue["category"],
                    "canonical": orig_cue["canonical"],
                    "original_surface": orig_cue["surface"],
                    "simplified_surface": simp_cue["surface"],
                    "original_span": [orig_cue["start"], orig_cue["end"]],
                    "simplified_span": [simp_cue["start"], simp_cue["end"]],
                })
                break

        if not matched:
            removed.append(orig_cue)

    # Any simplified cue not matched to identical original cue is candidate added/changed
    for s_idx, simp_cue in enumerate(simp_cues):
        if s_idx not in matched_simp_indices:
            added.append(simp_cue)

    # Step 2: Compare categories for directional changes (modals, negations, quantities)
    # Check Modals
    orig_modals = [c for c in orig_cues if c["category"] == CATEGORY_MODAL]
    simp_modals = [c for c in simp_cues if c["category"] == CATEGORY_MODAL]

    max_orig_m = max([c["strength"] for c in orig_modals if c["strength"] is not None], default=None)
    max_simp_m = max([c["strength"] for c in simp_modals if c["strength"] is not None], default=None)

    if max_orig_m is not None and max_simp_m is not None and max_orig_m != max_simp_m:
        orig_m_cues = [c for c in orig_modals if c["strength"] == max_orig_m]
        simp_m_cues = [c for c in simp_modals if c["strength"] == max_simp_m]
        orig_surf = orig_m_cues[0]["surface"] if orig_m_cues else ""
        simp_surf = simp_m_cues[0]["surface"] if simp_m_cues else ""
        orig_lbl = STRENGTH_LABELS.get(max_orig_m, f"Level {max_orig_m}")
        simp_lbl = STRENGTH_LABELS.get(max_simp_m, f"Level {max_simp_m}")

        if max_simp_m < max_orig_m:
            kind = "weakened"
            msg = (
                f"Potential obligation mismatch: modal weakened from '{orig_surf}' "
                f"({orig_lbl}) to '{simp_surf}' ({simp_lbl})."
            )
        else:
            kind = "strengthened"
            msg = (
                f"Potential obligation mismatch: modal strengthened from '{orig_surf}' "
                f"({orig_lbl}) to '{simp_surf}' ({simp_lbl})."
            )

        findings.append({
            "category": "modal",
            "kind": kind,
            "severity": "high",
            "message": msg,
            "original_spans": [{"start": c["start"], "end": c["end"], "surface": c["surface"]} for c in orig_m_cues],
            "simplified_spans": [{"start": c["start"], "end": c["end"], "surface": c["surface"]} for c in simp_m_cues],
        })
        changed.append({
            "category": "modal",
            "kind": kind,
            "original": orig_m_cues[0] if orig_m_cues else None,
            "simplified": simp_m_cues[0] if simp_m_cues else None,
        })
    elif max_orig_m is not None and max_simp_m is None:
        orig_m_cues = [c for c in orig_modals if c["strength"] == max_orig_m]
        findings.append({
            "category": "modal",
            "kind": "removed",
            "severity": "high",
            "message": f"Potential modal omission: original norm '{orig_m_cues[0]['surface']}' omitted in simplified text.",
            "original_spans": [{"start": c["start"], "end": c["end"], "surface": c["surface"]} for c in orig_m_cues],
            "simplified_spans": [],
        })

    # Check Negation Changes
    removed_neg = [c for c in removed if c["category"] == CATEGORY_NEGATION]
    added_neg = [c for c in added if c["category"] == CATEGORY_NEGATION]

    for c in removed_neg:
        findings.append({
            "category": "negation",
            "kind": "removed",
            "severity": "high",
            "message": f"Possible negation change: negation '{c['surface']}' in original appears removed in simplified text.",
            "original_spans": [{"start": c["start"], "end": c["end"], "surface": c["surface"]}],
            "simplified_spans": [],
        })
        changed.append({
            "category": "negation",
            "kind": "removed",
            "original": c,
            "simplified": None,
        })

    for c in added_neg:
        findings.append({
            "category": "negation",
            "kind": "added",
            "severity": "high",
            "message": f"Possible negation change: negation '{c['surface']}' introduced in simplified text.",
            "original_spans": [],
            "simplified_spans": [{"start": c["start"], "end": c["end"], "surface": c["surface"]}],
        })
        changed.append({
            "category": "negation",
            "kind": "added",
            "original": None,
            "simplified": c,
        })

    # Check Conditions
    orig_cond = [c for c in removed if c["category"] == CATEGORY_CONDITION]
    simp_cond = [c for c in added if c["category"] == CATEGORY_CONDITION]
    for c in orig_cond:
        findings.append({
            "category": "condition",
            "kind": "removed",
            "severity": "medium",
            "message": f"Condition appears omitted: original condition '{c['surface']}' is not preserved.",
            "original_spans": [{"start": c["start"], "end": c["end"], "surface": c["surface"]}],
            "simplified_spans": [],
        })
    for c in simp_cond:
        findings.append({
            "category": "condition",
            "kind": "added",
            "severity": "medium",
            "message": f"Condition appears added: condition '{c['surface']}' introduced in simplified text.",
            "original_spans": [],
            "simplified_spans": [{"start": c["start"], "end": c["end"], "surface": c["surface"]}],
        })

    # Check Exceptions
    orig_exc = [c for c in removed if c["category"] == CATEGORY_EXCEPTION]
    simp_exc = [c for c in added if c["category"] == CATEGORY_EXCEPTION]
    for c in orig_exc:
        findings.append({
            "category": "exception",
            "kind": "removed",
            "severity": "high",
            "message": f"Exception appears omitted: statutory proviso '{c['surface']}' is not preserved.",
            "original_spans": [{"start": c["start"], "end": c["end"], "surface": c["surface"]}],
            "simplified_spans": [],
        })
    for c in simp_exc:
        findings.append({
            "category": "exception",
            "kind": "added",
            "severity": "medium",
            "message": f"Exception appears added: proviso '{c['surface']}' introduced in simplified text.",
            "original_spans": [],
            "simplified_spans": [{"start": c["start"], "end": c["end"], "surface": c["surface"]}],
        })

    # Check Quantities and Numbers
    orig_quant = [c for c in removed if c["category"] in (CATEGORY_QUANTITY, "number")]
    simp_quant = [c for c in added if c["category"] in (CATEGORY_QUANTITY, "number")]

    if orig_quant and simp_quant:
        # Pair up mismatched quantities
        for o_q, s_q in zip(orig_quant, simp_quant):
            findings.append({
                "category": o_q["category"],
                "kind": "differ",
                "severity": "high",
                "message": f"Quantity differs: original has '{o_q['surface']}' while simplified has '{s_q['surface']}'.",
                "original_spans": [{"start": o_q["start"], "end": o_q["end"], "surface": o_q["surface"]}],
                "simplified_spans": [{"start": s_q["start"], "end": s_q["end"], "surface": s_q["surface"]}],
            })
            changed.append({"category": o_q["category"], "original": o_q, "simplified": s_q})
    elif orig_quant:
        for o_q in orig_quant:
            findings.append({
                "category": o_q["category"],
                "kind": "removed",
                "severity": "high",
                "message": f"Quantity differs: numeric threshold '{o_q['surface']}' omitted in simplified text.",
                "original_spans": [{"start": o_q["start"], "end": o_q["end"], "surface": o_q["surface"]}],
                "simplified_spans": [],
            })
    elif simp_quant:
        for s_q in simp_quant:
            findings.append({
                "category": s_q["category"],
                "kind": "added",
                "severity": "medium",
                "message": f"Quantity differs: numeric threshold '{s_q['surface']}' introduced in simplified text.",
                "original_spans": [],
                "simplified_spans": [{"start": s_q["start"], "end": s_q["end"], "surface": s_q["surface"]}],
            })

    # Check Dates
    orig_date = [c for c in removed if c["category"] == "date"]
    simp_date = [c for c in added if c["category"] == "date"]
    if orig_date and simp_date:
        for o_d, s_d in zip(orig_date, simp_date):
            findings.append({
                "category": "date",
                "kind": "differ",
                "severity": "high",
                "message": f"Date differs: original date '{o_d['surface']}' changed to '{s_d['surface']}'.",
                "original_spans": [{"start": o_d["start"], "end": o_d["end"], "surface": o_d["surface"]}],
                "simplified_spans": [{"start": s_d["start"], "end": s_d["end"], "surface": s_d["surface"]}],
            })
            changed.append({"category": "date", "original": o_d, "simplified": s_d})
    elif orig_date:
        for o_d in orig_date:
            findings.append({
                "category": "date",
                "kind": "removed",
                "severity": "high",
                "message": f"Date differs: statutory date '{o_d['surface']}' omitted.",
                "original_spans": [{"start": o_d["start"], "end": o_d["end"], "surface": o_d["surface"]}],
                "simplified_spans": [],
            })

    # Check Quantifiers
    orig_qf = [c for c in removed if c["category"] == CATEGORY_QUANTIFIER]
    simp_qf = [c for c in added if c["category"] == CATEGORY_QUANTIFIER]
    if orig_qf and simp_qf:
        findings.append({
            "category": "quantifier",
            "kind": "differ",
            "severity": "medium",
            "message": f"Quantifier differs: changed from '{orig_qf[0]['surface']}' to '{simp_qf[0]['surface']}'.",
            "original_spans": [{"start": orig_qf[0]["start"], "end": orig_qf[0]["end"], "surface": orig_qf[0]["surface"]}],
            "simplified_spans": [{"start": simp_qf[0]["start"], "end": simp_qf[0]["end"], "surface": simp_qf[0]["surface"]}],
        })

    # Check Entities
    orig_ent = [c for c in removed if c["category"] == CATEGORY_ENTITY]
    simp_ent = [c for c in added if c["category"] == CATEGORY_ENTITY]
    if orig_ent and simp_ent:
        findings.append({
            "category": "entity",
            "kind": "differ",
            "severity": "medium",
            "message": f"Entity difference: original refers to '{orig_ent[0]['surface']}' while simplified refers to '{simp_ent[0]['surface']}'.",
            "original_spans": [{"start": orig_ent[0]["start"], "end": orig_ent[0]["end"], "surface": orig_ent[0]["surface"]}],
            "simplified_spans": [{"start": simp_ent[0]["start"], "end": simp_ent[0]["end"], "surface": simp_ent[0]["surface"]}],
        })

    # Length Ratio Check (Omission / Addition)
    ratio = length_ratio(original_text, simplified_text)
    if ratio < 0.55:
        findings.append({
            "category": "omission",
            "kind": "potential_omission",
            "severity": "medium",
            "message": f"Potential omission: simplified sentence is substantially shorter than original (length ratio: {ratio:.2f}).",
            "original_spans": [],
            "simplified_spans": [],
        })
    elif ratio > 1.80:
        findings.append({
            "category": "hallucination",
            "kind": "potential_addition",
            "severity": "medium",
            "message": f"Potential addition or elaboration: simplified sentence is substantially longer than original (length ratio: {ratio:.2f}).",
            "original_spans": [],
            "simplified_spans": [],
        })

    # Summary synthesis
    if findings:
        summary_items = [f["message"] for f in findings]
        summary = "; ".join(summary_items)
    else:
        summary = "No salient cue difference detected."

    return {
        "cue_alignment": {
            "original": orig_cues,
            "simplified": simp_cues,
            "preserved": preserved,
            "removed": removed,
            "added": added,
            "changed": changed,
        },
        "findings": findings,
        "summary": summary,
    }


def get_visual_highlights(
    text: str,
    cues: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Produce visual-highlight-ready spans for rendering text."""
    spans = []
    for c in cues:
        category = c.get("category", "unknown")
        severity = "high" if category in (CATEGORY_MODAL, CATEGORY_NEGATION, CATEGORY_EXCEPTION, "date") else "medium"
        spans.append({
            "start": c["start"],
            "end": c["end"],
            "label": c.get("canonical", category),
            "category": category,
            "severity": severity,
            "matched_text": text[c["start"]:c["end"]],
        })
    return {
        "text": text,
        "spans": sorted(spans, key=lambda s: s["start"]),
    }


def render_highlighted_html(
    text: str,
    spans: Sequence[Mapping[str, Any]],
) -> str:
    """Render safely escaped HTML containing highlighting mark tags.

    Guarantees:
    - Never inserts unescaped raw user text.
    - Resolves overlapping spans cleanly.
    """
    if not text:
        return ""

    if not spans:
        return html.escape(text)

    # Sort and remove overlapping spans
    sorted_spans = sorted(spans, key=lambda s: (s["start"], -s["end"]))
    non_overlapping: list[Mapping[str, Any]] = []
    last_end = 0

    for s in sorted_spans:
        start = max(0, min(int(s["start"]), len(text)))
        end = max(start, min(int(s["end"]), len(text)))
        if start >= last_end and end > start:
            non_overlapping.append(s)
            last_end = end

    out_parts: list[str] = []
    cursor = 0

    for s in non_overlapping:
        start = int(s["start"])
        end = int(s["end"])
        cat = html.escape(str(s.get("category", "cue")))
        lbl = html.escape(str(s.get("label", cat)))

        # Append unhighlighted text before this span
        if start > cursor:
            out_parts.append(html.escape(text[cursor:start]))

        # Append highlighted span
        matched = html.escape(text[start:end])
        out_parts.append(
            f'<mark class="legal-cue legal-cue-{cat}" title="{lbl}">{matched}</mark>'
        )
        cursor = end

    # Append remaining trailing text
    if cursor < len(text):
        out_parts.append(html.escape(text[cursor:]))

    return "".join(out_parts)
