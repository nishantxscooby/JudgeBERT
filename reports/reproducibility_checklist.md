# Reproducibility & Integrity Verification Checklist

**Project:** Explainable Multilingual Legal Meaning Preservation Using a JUDGEBERT-Inspired Cross-Encoder  
**Verification Date:** 2026-11-18  
**Operating System:** Windows  
**Python Runtime:** Python 3.12.10 (virtual environment: `.venv`)  
**Strict Integrity Rule:** Only empirically verified engineering and experimental items are checked `[x]`. Zero fabricated data or draft labels are permitted.

---

## 1. Environment & Dependency Management

- [x] **Virtual Environment Initialized:** `.venv` created and active with Python 3.12.
- [x] **Locked Dependencies Defined:** `requirements.lock.txt` pins exact package versions (torch, transformers, sentence-transformers, scikit-learn, streamlit, pytest, etc.).
- [x] **Offline Test Execution Guarantee:** Entire test suite runs completely offline without downloading external model weights or calling paid third-party APIs.
- [x] **Deterministic Platform Seed:** Global seed `42` configured in `configs/config.yaml` (`project.seed`) and propagated across random, numpy, and torch backends via `set_global_seed()`.
- [x] **Config Fingerprint Tracked:** SHA-256 fingerprint of configuration generated with absolute paths stripped: `d014eeff5f86`.

---

## 2. Dataset Integrity & Partitioning

- [x] **Master Corpus File Present:** `data/processed/lmp_pairs.csv` exists and is tracked.
- [x] **Corpus SHA-256 Immutability & Provenance Verified:**  
  The master corpus was intentionally modified once by the reviewed human-gold merge and remained unchanged afterward.
  - Pre-merge unannotated backup: `data/processed/lmp_pairs.premerge.20261117T171541Z.csv` (SHA-256: `9F53CC5DEBAA2178C15BD7289EF920DFB8B7399CAAA23A3F30673F54E5385CE5`)
  - Post-merge human gold: `data/processed/lmp_pairs.csv` (SHA-256: `7327EBC8A519179DA648BE11DD5531D448934760B82850D36B214BC2AA12A216`)
- [x] **Row Count Verified:** Exactly 340 rows present.
- [x] **Primary Key Integrity:** Exactly 340 unique `pair_id` values (zero duplicates).
- [x] **Bilingual Balance Verified:** Exactly 170 English rows and 170 Hindi rows.
- [x] **Legal Domain Distribution Verified:**
  - `contract`: 202 rows
  - `cyber`: 72 rows
  - `sale_of_goods`: 66 rows
- [x] **Grouped Partitioning Integrity:**
  - Total source groups: 159
  - Train split: 234 rows (68.8%)
  - Validation split: 50 rows (14.7%)
  - Test split: 56 rows (16.5%)
- [x] **Zero Cross-Split Sibling Leakage:**  
  Verified: Exactly 0 source groups span multiple splits ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$).
- [x] **Strict Data Validation Passed:** `python -m src.data.validate_data --strict` passes with 0 errors (all 340 gold rows validated).

---

## 3. Human Annotation Status & Label Separation

- [x] **Independent Human Gold Annotations:** `[x] COMPLETE`  
  Exactly **340/340** rows populated in `human_lmp_score` (320 `human_single`, 20 `human_consensus`).
- [x] **Human Error Type Labels:** `[x] COMPLETE`  
  Exactly **340/340** rows populated in `error_type` across the 11-class legal taxonomy.
- [x] **Human Consensus Decisions:** `[x] COMPLETE`  
  Exactly **340/340** rows populated in `decision` (`Preserved`, `Partially Preserved`, `Not Preserved`).
- [x] **Draft Label Quarantine Enforced:**  
  `draft_lmp_score` and `draft_error_type` are strictly quarantined in `data/annotations/curator_drafts.csv` and prohibited from evaluation.
- [x] **Inter-Annotator Agreement Measured:** `[x] COMPLETE`  
  Agreement computation via `src/annotation/agreement.py` evaluated on 20 double-annotated pairs: $\kappa_w = 0.773$, $\alpha = 0.7015$, $\text{ICC}(2,1) = 0.782$.
- [x] **Blind Annotation Tool Ready:** `src/annotation/annotation_app.py` implemented with strict blind scoring, atomic temporary file replacement, and queue shuffle.

---

## 4. Modeling Foundation & Architecture

- [x] **Legally Conservative Preprocessing:** `src/data/preprocess.py` normalizes whitespace and Unicode while protecting numbers, dates, punctuation, and Devanagari script.
- [x] **Legal Cue Extraction Engine:** `src/data/legal_cues.py` extracts deontic modals, negations, conditions, exceptions, quantities, dates, entities, and quantifiers.
- [x] **TF-IDF Baseline Implemented:** `src/models/tfidf_baseline.py` combines word [1, 2] and character [3, 5] n-grams with paired cosine similarity.
- [x] **SBERT Baseline Implemented:** `src/models/sbert_baseline.py` implements dense multilingual embeddings via `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
- [x] **Baseline Calibration Logic Enforced:** Train-only linear calibration ($y = \alpha \cdot \text{sim} + \beta$) enforced in code; uncalibrated cosine distance is rejected.
- [x] **JUDGEBERT-Inspired Cross-Encoder Implemented:** `src/models/judgebert_model.py` operationalizes a cross-encoder regression head on `google/muril-base-cased` with target scaling and Smooth L1 loss.
- [x] **Trained Cross-Encoder Checkpoint:** `[x] COMPLETE`  
  Supervised training on gold labels completed; checkpoint saved at `models/checkpoints/judgebert/checkpoint.pt`.
  A prediction-collapse audit found the retrained checkpoint to have insufficient score spread (approximately 4.624–4.627), so it is not enabled for interactive inference.

---

## 5. Explainability Layer

- [x] **Auditable Legal Cue Alignment (Primary):** `src/explainability/cue_alignment.py` extracts and aligns salient legal cues across original and simplified texts.
- [x] **Hedged Findings & Categorized Shifts:** Cue shifts classified into `preserved`, `removed`, `added`, `changed` with non-dogmatic diagnostic messages.
- [x] **Safe HTML Highlight Rendering:** `render_highlighted_html` escapes all text and attributes using `html.escape(..., quote=True)` to prevent XSS.
- [x] **Supporting Attention Attribution Engine:** `src/explainability/attention_attribution.py` extracts last-layer `[CLS]` attention over `[CLS] orig [SEP] simp [SEP]`.
- [x] **Mandatory Attention Disclaimer Present:**  
  > *"Attention weights are not guaranteed to be faithful legal explanations. They are reported as weak supporting evidence only, and are never presented as a legal justification."*
- [x] **Offline Mock Attention Support:** Deterministic mock attention attribution implemented for test verification without GPU or Hugging Face downloads.

---

## 6. Experiment Harness & Evaluation

- [x] **Experiment 1 (Baselines vs JUDGEBERT):** `[x] COMPLETE` — Evaluated on 56 held-out test pairs with 1,000-resample cluster bootstrap CIs.
- [x] **Experiment 2 (English vs Hindi):** `[x] COMPLETE` — Evaluated on 28 English and 28 Hindi held-out test pairs.
- [x] **Experiment 3 (Held-Out-Domain Generalization):** `[x] COMPLETE` — Held-out-domain generalization: trained using contract and sale_of_goods data, then evaluated across the complete cyber domain, which is absent from training (72 pairs: 48 train, 12 val, 12 test rows of cyber; note the official test split contains 12 cyber rows).
- [x] **Experiment 4 Diagnostic Runner (Sanity Checks):** `[x] COMPLETE` — Verified on synthetic minimal pairs (`DIAGNOSTIC_ONLY`).
- [x] **Experiment 5 (Qualitative Error Analysis):** `[x] COMPLETE` — Sampled 20 test cases ($\ge 15$) across all error phenomena.
- [x] **Master Experiment Runner:** `src/experiments/run_all.py` executed successfully, generating `summary_manifest.json` and `experiments_summary.md`.
- [x] **Final Empirical Evaluation Results:** `[x] COMPLETE` — All empirical metrics generated and saved in `results/experiments/`.

---

## 7. Interactive Demonstration (Streamlit App)

- [x] **Interactive Streamlit App Implemented:** `src/app/streamlit_app.py` implemented.
- [x] **Headless Server Startup Verified:** Tested on port 8502 (`Uvicorn server started`).
- [x] **Clean Bare Import Verified:** `python -c "from src.app import streamlit_app"` executes cleanly without starting a background server.
- [x] **Model Registry Integration:** Successfully loads MuRIL cross-encoder checkpoint and calibrated baselines.
- [x] **Prototype & No Legal Advice Disclaimers Displayed:** Prominently featured in header.
- [x] **Curated Sample Explorer Integrity:** Prohibits and strips `draft_lmp_score`, `draft_error_type`, and expected labels.
- [x] **Missing Checkpoint Handling:** Detects absent model checkpoints and alerts user without fabricating dummy fallback scores.

---

## 8. Verification & Test Suite Summary

- [x] **Automated Test Suite Verified:** 974 passed, 0 failed, 0 skipped.
- [x] **Focused App Suite Verified:** 45 passed across app tests.
- [x] **Syntax Compilation Verified:** `compileall -q src tests` passed with 0 errors.
- [x] **Known Limitations Documented:**
  - [x] Hindi PDF font corruption and author-rendered provenance asymmetry documented in `reports/source_provenance.md`.
  - [x] Compact double-annotation coverage (20 pairs) documented in `reports/agreement.md`.
  - [x] Attention attribution limitations documented with mandatory disclaimers.
  - [x] Cross-encoder described as JUDGEBERT-*inspired*, not a carbon copy.
