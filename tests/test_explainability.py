"""Comprehensive tests for the legal cue alignment and explainability layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.explainability.cue_alignment import (
    align_legal_cues,
    extract_all_cues,
    get_visual_highlights,
    render_highlighted_html,
)
from src.explainability.explanations import explain_pair, main


def test_identical_english_text() -> None:
    """Identical English sentences must produce zero change findings and state no difference."""
    text = "The seller shall deliver the goods in accordance with the contract."
    result = explain_pair(text, text, pair_id="ID_EN_01")

    assert result["pair_id"] == "ID_EN_01"
    assert result["language"] == "en"
    assert result["summary"] == "No salient cue difference detected."

    # Findings must be empty list or explicit no-change
    change_findings = [f for f in result["findings"] if f.get("kind") != "preserved"]
    assert len(change_findings) == 0

    # Cues should be in preserved alignment
    preserved = result["cue_alignment"]["preserved"]
    assert len(preserved) > 0
    assert any(p["category"] == "modal" and p["canonical"] == "shall" for p in preserved)


def test_identical_hindi_text() -> None:
    """Identical Hindi sentences must produce valid Devanagari alignment with no differences."""
    text = "विधि द्वारा प्रवर्तनीय करार संविदा है।"
    result = explain_pair(text, text, pair_id="ID_HI_01")

    assert result["language"] == "hi"
    assert result["summary"] == "No salient cue difference detected."

    change_findings = [f for f in result["findings"] if f.get("kind") != "preserved"]
    assert len(change_findings) == 0


def test_modal_obligation_to_permission_weakened() -> None:
    """Shall -> May modal shift must be reported as a weakened obligation."""
    orig = "The tenant shall pay the rent on the first day of each month."
    simp = "The tenant may pay the rent on the first day of each month."
    result = explain_pair(orig, simp)

    modal_findings = [f for f in result["findings"] if f["category"] == "modal"]
    assert len(modal_findings) >= 1
    finding = modal_findings[0]
    assert finding["kind"] == "weakened"
    assert finding["severity"] == "high"
    assert "Potential obligation mismatch" in finding["message"]
    assert finding["original_spans"][0]["surface"] == "shall"
    assert finding["simplified_spans"][0]["surface"] == "may"


def test_negation_removal() -> None:
    """Dropping a negation must be detected as a removed negation."""
    orig = "An agreement without consideration is not valid."
    simp = "An agreement without consideration is valid."
    result = explain_pair(orig, simp)

    neg_findings = [f for f in result["findings"] if f["category"] == "negation"]
    assert len(neg_findings) >= 1
    assert neg_findings[0]["kind"] == "removed"
    assert "Possible negation change" in neg_findings[0]["message"]


def test_condition_removal() -> None:
    """Omitting an 'if' condition must be flagged as a removed condition."""
    orig = "If notice is given in writing, the contract shall remain in force."
    simp = "The contract shall remain in force."
    result = explain_pair(orig, simp)

    cond_findings = [f for f in result["findings"] if f["category"] == "condition"]
    assert len(cond_findings) >= 1
    assert cond_findings[0]["kind"] == "removed"
    assert "Condition appears omitted" in cond_findings[0]["message"]


def test_exception_removal() -> None:
    """Omitting an 'except' clause must be reported as an omitted exception."""
    orig = "The goods shall be delivered immediately, except where prevented by law."
    simp = "The goods shall be delivered immediately."
    result = explain_pair(orig, simp)

    exc_findings = [f for f in result["findings"] if f["category"] == "exception"]
    assert len(exc_findings) >= 1
    assert exc_findings[0]["kind"] == "removed"
    assert "Exception appears omitted" in exc_findings[0]["message"]


def test_quantity_difference() -> None:
    """Modifying 30 days to 60 days must be reported as a quantity difference."""
    orig = "Notice shall be delivered within thirty days."
    simp = "Notice shall be delivered within sixty days."
    result = explain_pair(orig, simp)

    quant_findings = [f for f in result["findings"] if f["category"] == "quantity"]
    assert len(quant_findings) >= 1
    assert quant_findings[0]["kind"] == "differ"
    assert "Quantity differs" in quant_findings[0]["message"]


def test_date_difference() -> None:
    """Modifying calendar dates must be identified as a date difference."""
    orig = "This Act came into force on the first day of September 1872."
    simp = "This Act came into force on the first day of October 1872."
    result = explain_pair(orig, simp)

    date_findings = [f for f in result["findings"] if f["category"] == "date"]
    assert len(date_findings) >= 1
    assert date_findings[0]["kind"] == "differ"
    assert "Date differs" in date_findings[0]["message"]


def test_offset_correctness_and_hindi_unicode() -> None:
    """Character offsets must slice exact matched substrings in both English and Hindi."""
    en_text = "The seller shall deliver within 30 days unless exempted."
    en_cues = extract_all_cues(en_text, language="en")
    assert len(en_cues) > 0
    for c in en_cues:
        assert en_text[c["start"]:c["end"]] == c["surface"]

    hi_text = "यदि क्रेता तीस दिन के भीतर संदाय नहीं करता है तो संविदा शून्य होगी।"
    hi_cues = extract_all_cues(hi_text, language="hi")
    assert len(hi_cues) > 0
    for c in hi_cues:
        assert hi_text[c["start"]:c["end"]] == c["surface"]


def test_structured_output_contract() -> None:
    """Structured output must be completely JSON serializable and contain all required contract keys."""
    orig = "The buyer must inspect the goods within seven days."
    simp = "The buyer may inspect the goods within fourteen days."
    result = explain_pair(orig, simp, pair_id="CONTRACT_TEST_01", model_prediction=6.5, model_name="judgebert")

    # JSON serialization test
    json_str = json.dumps(result, ensure_ascii=False)
    parsed = json.loads(json_str)

    # Contract checks
    assert parsed["pair_id"] == "CONTRACT_TEST_01"
    assert "cue_alignment" in parsed
    assert "findings" in parsed
    assert "summary" in parsed
    assert "highlights" in parsed
    assert "model_prediction" in parsed
    assert parsed["model_prediction"]["score"] == 6.5
    assert parsed["research_valid"] is False
    assert parsed["human_gold_used"] is False
    assert "disclaimer" in parsed
    assert "This is an auditable cue comparison, not legal advice." in parsed["disclaimer"]


def test_html_rendering_escapes_xss() -> None:
    """HTML rendering must safely escape user inputs."""
    malicious_text = '<script>alert("xss")</script> shall pay & receive.'
    cues = extract_all_cues(malicious_text, language="en")
    highlights = get_visual_highlights(malicious_text, cues)
    rendered = render_highlighted_html(malicious_text, highlights["spans"])

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&amp;" in rendered


def test_invalid_text_inputs_raise_clear_error() -> None:
    """Empty or non-string inputs must raise ValueError."""
    with pytest.raises(ValueError, match="original_text must be a non-empty string"):
        explain_pair("", "Simplified text")

    with pytest.raises(ValueError, match="simplified_text must be a non-empty string"):
        explain_pair("Original text", "   ")


def test_cli_single_pair(tmp_path: Path) -> None:
    """CLI single pair execution must write valid JSON."""
    out_file = tmp_path / "test_out.json"
    exit_code = main([
        "--original", "The party shall provide notice.",
        "--simplified", "The party may provide notice.",
        "--output", str(out_file),
    ])
    assert exit_code == 0
    assert out_file.is_file()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["summary"] != ""
    assert data["human_gold_used"] is False


def test_entity_difference() -> None:
    """Changing named legal entities (e.g. buyer/seller vs tenant/landlord) must be identified."""
    orig = "The buyer shall notify the seller."
    simp = "The tenant shall notify the landlord."
    result = explain_pair(orig, simp)

    entity_findings = [f for f in result["findings"] if f["category"] == "entity"]
    assert len(entity_findings) >= 1
    assert "Entity difference" in entity_findings[0]["message"]


def test_quantifier_difference() -> None:
    """Changing universal to existential quantifier must be identified."""
    orig = "All parties shall execute the agreement."
    simp = "Some parties shall execute the agreement."
    result = explain_pair(orig, simp)

    qf_findings = [f for f in result["findings"] if f["category"] == "quantifier"]
    assert len(qf_findings) >= 1
    assert "Quantifier differs" in qf_findings[0]["message"]


def test_added_and_removed_text_omission_hallucination() -> None:
    """Extreme length disparity triggers omission or hallucination warnings."""
    long_orig = "The seller shall deliver the goods in accordance with the specifications, quality requirements, packaging norms, transit instructions, and delivery schedule specified in the schedule."
    short_simp = "The seller delivers."
    omission_result = explain_pair(long_orig, short_simp)
    omission_findings = [f for f in omission_result["findings"] if f["category"] == "omission"]
    assert len(omission_findings) >= 1
    assert "Potential omission" in omission_findings[0]["message"]

    short_orig = "The seller delivers."
    long_simp = "The seller shall deliver the goods in accordance with the specifications, quality requirements, packaging norms, transit instructions, and delivery schedule specified in the schedule."
    hallucination_result = explain_pair(short_orig, long_simp)
    hallucination_findings = [f for f in hallucination_result["findings"] if f["category"] == "hallucination"]
    assert len(hallucination_findings) >= 1
    assert "Potential addition" in hallucination_findings[0]["message"]


def test_cli_batch_file(tmp_path: Path) -> None:
    """CLI batch execution on CSV file must process all rows and write valid JSON."""
    csv_file = tmp_path / "input_pairs.csv"
    csv_file.write_text(
        "pair_id,original_text,simplified_text\n"
        "P01,The seller shall deliver.,The seller may deliver.\n"
        "P02,Notice within 30 days.,Notice within 60 days.\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "batch_out"
    exit_code = main([
        "--input", str(csv_file),
        "--output-dir", str(out_dir),
    ])
    assert exit_code == 0
    batch_json = out_dir / "batch_explanations.json"
    assert batch_json.is_file()
    data = json.loads(batch_json.read_text(encoding="utf-8"))
    assert data["count"] == 2
    assert len(data["explanations"]) == 2
    assert data["explanations"][0]["pair_id"] == "P01"
    assert data["explanations"][1]["pair_id"] == "P02"


def test_safety_dataset_immutability() -> None:
    """Explainability execution must never touch or mutate the master dataset."""
    import hashlib
    from src.explainability import explain_pair

    master_path = Path("data/processed/lmp_pairs.csv")
    if master_path.is_file():
        digest_before = hashlib.sha256(master_path.read_bytes()).hexdigest()
        explanation = explain_pair(
            "The borrower shall repay the loan within 30 days.",
            "The borrower may repay the loan within 60 days.",
        )
        assert "cue_alignment" in explanation
        digest_after = hashlib.sha256(master_path.read_bytes()).hexdigest()
        assert digest_before == digest_after

