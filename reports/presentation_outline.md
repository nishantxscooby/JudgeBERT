# Presentation Deck Outline — Explainable Multilingual Legal Meaning Preservation Using JUDGEBERT

**Course:** CSET 346 — Natural Language Processing  
**Project Title:** Explainable Multilingual Legal Meaning Preservation Using a JUDGEBERT-Inspired Cross-Encoder  
**Format:** 15-Slide Presentation Deck Specification  

---

## Slide 1: Title & Overview

- **Slide Title:** Explainable Multilingual Legal Meaning Preservation Using a JUDGEBERT-Inspired Cross-Encoder
- **Subtitle:** Assessing Statutory Reformulations in English and Hindi Across Indian Legal Domains
- **Presenter / Team:** CSET 346 Semester Project Team
- **Suggested Visual:** Project header banner featuring Indian statutory iconography, bilingual sentence pair comparison (English & Devanagari), and a clear system badge: `Complete · Human Gold Verified & Evaluated`.
- **Main Bullet Points:**
  - Automated evaluation of Legal Meaning Preservation (LMP) between statutory provisions and simplified versions.
  - Bilingual coverage: English and Hindi across three authentic Indian legal domains.
  - Three distinct modeling paradigms: TF-IDF baseline, Multilingual SBERT, and MuRIL Cross-Encoder.
  - Transparent, two-tiered explainability: Auditable legal cue alignment + supporting token attention.
  - Strict academic integrity: Full pipeline implemented and verified with 974 tests; human gold annotation, supervised training, and Experiments 1–5 complete.
- **Speaker Notes:**
  > "Good morning, everyone. Today we present our work on Explainable Multilingual Legal Meaning Preservation. When legal documents or statutes are simplified for citizens, a critical question arises: does the simplified text preserve the exact legal meaning of the original law? We have built an end-to-end, reproducible NLP pipeline and explainability system inspired by JUDGEBERT to evaluate this in English and Hindi. The entire pipeline—from human gold annotation and inter-annotator agreement to supervised cross-encoder training, held-out evaluation across Experiments 1–5, and two-tiered explainability—is fully implemented, empirically evaluated, and verified by 974 passing tests."
- **Claims That Must NOT Be Made:**
  - Do NOT claim that this is an exact carbon-copy reproduction of JUDGEBERT.
  - Do NOT claim the software is ready for commercial or clinical legal practice.

---

## Slide 2: Motivation — Why Legal Similarity Is Different

- **Slide Title:** The Problem: Why Standard Semantic Similarity Fails in Law
- **Suggested Visual:** Contrast diagram comparing standard STS vs. Legal Meaning Preservation:
  - *Case A (Paraphrase):* "The car is red" vs "The automobile is crimson" $\rightarrow$ High semantic similarity, identical meaning.
  - *Case B (Legal Shift):* "The tenant shall pay within 30 days" vs "The tenant may pay within 60 days" $\rightarrow$ High lexical/embedding similarity ($>0.92$), completely divergent legal consequence.
- **Main Bullet Points:**
  - Plain-language legal reform is vital for public access to justice, but reformulations risk semantic drift.
  - Standard STS models (e.g., Cosine on embeddings) optimize for topical overlap, not legal enforceability.
  - Legally load-bearing tokens (deontic modals, negation, dates, statutory deadlines) carry disproportionate weight.
  - A single word change ("shall" $\rightarrow$ "may") alters mandatory obligations into discretionary permissions.
- **Speaker Notes:**
  > "Standard NLP models view sentences through semantic closeness. But in law, two sentences can share 95% of their vocabulary and still mean opposite things. For instance, shifting 'shall' to 'may' or changing '30 days' to '60 days' completely changes the legal rights of the parties involved. That is why general semantic similarity tools are insufficient, and why a dedicated Legal Meaning Preservation approach is necessary."
- **Claims That Must NOT Be Made:**
  - Do NOT claim that standard sentence transformers are useless in all legal tasks; emphasize their limitation on deontic preservation specifically.

---

## Slide 3: Research Question & Scope

- **Slide Title:** Research Question & Project Scope
- **Suggested Visual:** Core question callout box with language flags (English / Hindi) and three domain emblems: Contract Act, IT Act, Sale of Goods Act.
- **Main Bullet Points:**
  - **Core Research Question:** *How effectively can a JUDGEBERT-based model assess legal meaning preservation in a new language and legal domain, and does it outperform generic semantic-similarity approaches?*
  - **Bilingual Scope:** English and Hindi (Devanagari script).
  - **Statutory Domains:**
    - Indian Contract Act, 1872 (Civil/Commercial — Primary domain)
    - Information Technology Act, 2000 (Cyber Law — Held-out domain)
    - Sale of Goods Act, 1930 (Commercial Transactions — Third domain)
  - **Output Scale:** Continuous 1–10 LMP score mapped to a 3-way decision (Preserved, Partially Preserved, Not Preserved).
- **Speaker Notes:**
  > "Our research question evaluates whether a cross-encoder model inspired by JUDGEBERT can generalize across languages and novel legal domains compared to lexical and bi-encoder baselines. We focus on Indian statutory frameworks, evaluating contract law, sale of goods, and cyber law."
- **Claims That Must NOT Be Made:**
  - Do NOT claim answers to the research question have been definitively settled prior to human gold scoring.

---

## Slide 4: The Legal Meaning Preservation (LMP) Task

- **Slide Title:** Defining the 1–10 Legal Meaning Preservation Scale
- **Suggested Visual:** Horizontal graduated scale diagram from 1 to 10 with the three decision bands and anchor definitions:
  - Score 1–3: `Not Preserved` (Contradiction, fatal omission, or unauthorized scope expansion)
  - Score 4–7: `Partially Preserved` (Core intent retained, but modal force or minor condition weakened)
  - Score 8–10: `Preserved` (Complete legal equivalence; rights and obligations identical)
- **Main Bullet Points:**
  - Modeled as a regression task on a 1.0 to 10.0 continuous scale.
  - Operationalized three-way legal decision:
    - **Preserved ($\ge 8.0$):** Valid legal substitute; no alteration of rights/liabilities.
    - **Partially Preserved ($4.0 \le \text{score} < 8.0$):** Minor drift or qualification shift requiring legal review.
    - **Not Preserved ($< 4.0$):** Meaning distorted, obligation broken, or liability flipped.
  - Accompanied by an 11-category single-label error taxonomy (Omission, Hallucination, Modal shift, Negation, Condition, Exception, Scope, Date, Number, Entity, Other).
- **Speaker Notes:**
  > "LMP is scored from 1 to 10. A score of 8 to 10 means the text is legally equivalent and can serve as a substitute. A score of 4 to 7 means partial preservation where core rights are maintained but nuances are altered. Scores below 4 indicate critical legal failure. In addition, our error taxonomy categorizes exactly what went wrong."
- **Claims That Must NOT Be Made:**
  - Do NOT claim that this tool provides legal advice or automated legal certification.

---

## Slide 5: The Corpus — Composition & Balance

- **Slide Title:** Dataset Composition: 340 Statutory Pairs
- **Suggested Visual:** Split and domain breakdown tables:
  - 340 Total Pairs (170 English, 170 Hindi)
  - 202 Contract / 72 Cyber / 66 Sale of Goods
  - 234 Train / 50 Validation / 56 Test
- **Main Bullet Points:**
  - Exactly balanced across languages: 170 English pairs and 170 Hindi pairs.
  - Authored from authentic statutory provisions across 159 unique `source_group`s.
  - Balanced representation of all 11 error categories and full 1–10 score spectrum in the drafted corpus.
  - The master corpus was intentionally modified once by the reviewed human-gold merge and remained unchanged afterward.
  - Pre-merge backup: `data/processed/lmp_pairs.premerge.20261117T171541Z.csv` (SHA-256 `9F53CC5DEBAA2178C15BD7289EF920DFB8B7399CAAA23A3F30673F54E5385CE5`); post-merge gold: `data/processed/lmp_pairs.csv` (SHA-256 `7327EBC8A519179DA648BE11DD5531D448934760B82850D36B214BC2AA12A216`).
- **Speaker Notes:**
  > "The corpus consists of 340 authentic pairs. It is exactly 50% English and 50% Hindi, covering 202 contract provisions, 72 cyber law provisions, and 66 sale of goods provisions. Every pair originates from an identified statutory section."
- **Claims That Must NOT Be Made:**
  - Do NOT claim the corpus is large; acknowledge it is a focused, high-quality statutory corpus.

---

## Slide 6: Data Provenance & The Hindi Typographic Asymmetry

- **Slide Title:** Critical Provenance Finding: Official Hindi PDF Corruption
- **Suggested Visual:** Side-by-side comparison snippet from `reports/source_provenance.md`:
  - Correct Hindi: भारतीय संविदा **अधिनियम**, 1872
  - PDF Extracted: भारतीय संविदा **अविवियम**, 1872
  - Character mapping failure: `ध` maps to `ग` in one word and `ि` in an adjacent word on the same page.
- **Main Bullet Points:**
  - English provisions were cleanly extracted from official Government of India PDFs.
  - All official Hindi PDFs obtained contain broken legacy non-Unicode font mappings.
  - Non-invertible distortion: cannot be reversed via a character lookup table.
  - Solution: All 170 Hindi rows were manually typed by the author from official printed gazettes (`verbatim_quote: "false"`).
  - **Research Consequence:** English and Hindi have different provenance categories. Cross-lingual comparisons capture both a language effect and a provenance effect.
- **Speaker Notes:**
  > "A crucial finding documented in our provenance report: official government Hindi PDFs use legacy font mappings that decode into gibberish. The word 'Act' (अधिनियम) does not appear correctly anywhere in the extracted PDFs. Because the mapping is non-invertible, we typed the Hindi provisions by hand from printed gazette texts. This creates an unavoidable provenance asymmetry between English direct quotes and Hindi author-transcriptions."
- **Claims That Must NOT Be Made:**
  - Do NOT claim that the Hindi sub-corpus is AI-generated; it is author-rendered from authentic gazette printings.
  - Do NOT hide the provenance asymmetry; present it as a key research finding.

---

## Slide 7: Leakage-Resistant Data Pipeline

- **Slide Title:** Strict Anti-Leakage Pipeline & Grouped Splitting
- **Suggested Visual:** Grouped splitting flowchart showing `source_group` partitioning:
  - Source Section $\rightarrow$ `source_group` $\rightarrow$ Atomic allocation to Train, Val, or Test.
  - Cross-split overlap: **0 groups**.
- **Main Bullet Points:**
  - Problem: Sibling simplifications derived from the same source sentence in train and test would cause severe data leakage.
  - Solution: Partitioning strictly grouped by `source_group` (159 total groups).
  - 0 source groups span multiple splits.
  - Grouped stratification preserves language and domain distribution across splits.
  - Legally conservative preprocessing: retains numbers, dates, punctuation, and Devanagari script integrity without destructive tokenization.
- **Speaker Notes:**
  > "To guarantee evaluation validity, our splitting algorithm groups pairs by `source_group`. If multiple simplifications exist for Section 73 of the Contract Act, all of them are confined to either train, validation, or test. Not a single source group crosses split boundaries, eliminating sibling leakage."
- **Claims That Must NOT Be Made:**
  - Do NOT claim that individual pairs were shuffled randomly; emphasize grouped allocation.

---

## Slide 8: Baseline Models & Train-Only Calibration

- **Slide Title:** Baselines: TF-IDF & Multilingual Sentence-BERT
- **Suggested Visual:** Dual calibration regression plot illustration: Raw Cosine Distance $[-1.0, 1.0] \rightarrow$ Linear Calibration $\rightarrow$ Scaled Score $[1.0, 10.0]$.
- **Main Bullet Points:**
  - **Baseline 1 (TF-IDF):** Word n-grams [1, 2] + Character n-grams [3, 5] (`char_wb`). Captures both lexical overlap and morphological inflection in Hindi.
  - **Baseline 2 (SBERT):** Multilingual MPNet (`paraphrase-multilingual-mpnet-base-v2`) dense bi-encoder.
  - **Strict Calibration Requirement:** Raw cosine similarity is NOT an LMP score.
  - Linear regression calibration ($y = \alpha \cdot \text{sim} + \beta$) is fitted **strictly on the train split**.
  - Validation/test prediction without training calibration is blocked by design.
- **Speaker Notes:**
  > "We established two baselines: word/character TF-IDF and multilingual SBERT. A critical methodological discipline here is calibration. Cosine similarity ranges from -1 to 1; reporting cosine similarity as an LMP score is scientifically invalid. Both baselines enforce a linear calibration fitted strictly on train data."
- **Claims That Must NOT Be Made:**
  - Do NOT report uncalibrated cosine similarity as an LMP score.

---

## Slide 9: JUDGEBERT-Inspired Cross-Encoder Architecture

- **Slide Title:** Main Model: MuRIL Multilingual Cross-Encoder
- **Suggested Visual:** Architecture block diagram:
  - Input: `[CLS] Statutory Text [SEP] Simplified Text [SEP]`
  - Backbone: MuRIL Base Cased (16 Indic languages + English)
  - Head: Dropout ($p=0.2$) $\rightarrow$ Linear Regression Layer
  - Loss: Smooth L1 Loss ($\beta=0.5$) with Min-Max Target Scaling
- **Main Bullet Points:**
  - Full cross-attention between statutory provision and simplified candidate across all layers.
  - Backbone: `google/muril-base-cased` pre-trained on Indian legal and general corpora.
  - Target scaling: Normalizes 1–10 scores to $[0, 1]$ via $(y - 1)/9$ to stabilize gradients.
  - Smooth L1 (Huber) loss provides resilience against scoring variance.
  - Early stopping on validation RMSE with encoder freezing during Epoch 1.
- **Speaker Notes:**
  > "Our primary neural model is a cross-encoder built on Google's MuRIL backbone. Unlike bi-encoders that process sentences separately, the cross-encoder allows every token of the simplified text to cross-attend to every token of the statutory text. We use target scaling, Smooth L1 loss, and early stopping on validation RMSE."
- **Claims That Must NOT Be Made:**
  - Do NOT claim this is an exact clone of the French JUDGEBERT model.

---

## Slide 10: Two-Tiered Explainability Layer

- **Slide Title:** Explainability: Auditable Legal Cue Alignment
- **Suggested Visual:** Screen capture of the highlighted HTML output:
  - Original: "The contractor <mark class='modal'>shall</mark> deliver within <mark class='quantity'>30 days</mark>."
  - Simplified: "The contractor <mark class='modal'>may</mark> deliver within <mark class='quantity'>60 days</mark>."
  - Finding: *"Potential obligation mismatch: modal weakened from 'shall' to 'may'."*
  - Finding: *"Quantity differs: original has '30 days' while simplified has '60 days'."*
- **Main Bullet Points:**
  - **Tier 1 (Primary & Auditable):** Rule-based legal cue extractor tracking 9 categories:
    - Modals (Deontic strength 1–3), Negations, Conditions, Exceptions, Quantities, Dates, Entities, Quantifiers, Length Ratios.
    - Classified into `preserved`, `removed`, `added`, `changed`.
    - Generates hedged, non-dogmatic explanations.
  - **Tier 2 (Supporting Evidence):** Token-level attention attribution with mandatory disclaimer:
    > *"Attention weights are not guaranteed to be faithful legal explanations."*
  - Safe HTML highlight rendering with strict `html.escape`.
- **Speaker Notes:**
  > "Neural predictions require auditability. We implemented a two-tiered explainability layer. The primary explanation is rule-based cue alignment across nine legal categories. It clearly flags modal weakening, negation removal, or statutory deadline shifts. Supporting this is token-level attention attribution, but we explicitly disclaim attention as a legal justification."
- **Claims That Must NOT Be Made:**
  - Do NOT claim that attention weights prove legal causality.
  - Do NOT claim rule-based cue detection replaces legal review.

---

## Slide 11: Experiments 1–5 Empirical Research Findings

- **Slide Title:** Empirical Research Findings (Experiments 1–5)
- **Suggested Visual:** Experiment matrix table showing designs, splits, and completed statuses:
  - Exp 1: Baselines vs JUDGEBERT $\rightarrow$ ✅ **Evaluated on Human Gold** ($n=56$ test pairs)
  - Exp 2: English vs Hindi Breakdown $\rightarrow$ ✅ **Evaluated on Human Gold** ($n=28$ en, $n=28$ hi)
  - Exp 3: Held-Out-Domain Generalization $\rightarrow$ ✅ **Evaluated on Human Gold** ($n=72$ cyber pairs)
  - Exp 4: Behavioral Sanity Checks $\rightarrow$ ✅ **Verified (Diagnostic Probes Passed)**
  - Exp 5: Qualitative Error Analysis $\rightarrow$ ✅ **Sampled on Human Gold** (20 cases $\ge 15$)
- **Main Bullet Points:**
  - All experiments executed on genuine human gold ground truth.
  - Held-out-domain generalization (Exp 3): trained using contract and sale_of_goods data, then evaluated across the complete cyber domain, which is absent from training (72 pairs: 48 train, 12 val, 12 test rows under the standard grouped split; the official test split contains 12 cyber rows).
  - Experiment 4 verified on synthetic behavioral fixtures (probes confirmed expected sensitivities: identical pairs +0.8755 higher similarity; deontic modal, negation, and quantity flips flagged).
  - Experiment 5 stratified 20 qualitative test cases across error phenomena, observing that major model divergences occur on modal shifts and condition omissions.
  - Atomic provenance metadata stamped on all generated outputs in `results/experiments/`.
- **Speaker Notes:**
  > "Our experiment harness covers five core investigations: overall model comparison, cross-lingual breakdown, held-out domain generalization on cyber law, diagnostic sanity checks, and qualitative error analysis. All empirical experiments were executed on genuine human gold labels. For Experiment 3, the model was trained strictly on contract and sale of goods provisions and evaluated across the complete held-out cyber domain of 72 pairs, testing generalization to distinct statutory regimes."
- **Claims That Must NOT Be Made:**
  - Do NOT present Experiment 4 diagnostic numbers as research performance metrics.
  - Do NOT claim that 72 cyber provisions are all from the official test split; clarify that 12 are in the official test split, while all 72 were held out from training.

---

## Slide 12: Streamlit Demonstration Application

- **Slide Title:** Interactive Demonstration UI (Phase 8)
- **Suggested Visual:** UI screenshot of `streamlit_app.py` showing:
  - Informational banner: `Interactive Legal Meaning Analyzer`
  - Bilingual sentence input text areas
  - Primary legal cue alignment with color-coded tokens
  - Active model selector: MuRIL Cross-Encoder, Calibrated TF-IDF, Calibrated SBERT
- **Main Bullet Points:**
  - Interactive demonstration tool implemented in `src/app/streamlit_app.py`.
  - Verified headless startup and tested with `streamlit.testing.v1.AppTest`.
  - Bilingual input validation rejecting whitespace or excessively long text while preserving Devanagari Unicode.
  - Curated corpus explorer displaying authentic statutory pairs without leaking draft scores.
  - Loads trained MuRIL cross-encoder checkpoint and calibrated baselines.
  - Graceful degradation: cue alignment runs seamlessly even if model weights are unloaded.
  - Zero fabrication: never returns a dummy score when a checkpoint is missing.
- **Speaker Notes:**
  > "In Phase 8, we delivered the interactive Streamlit demo. The UI demonstrates bilingual text input, curated statutory sample exploration, and safe HTML cue highlighting. It connects to the trained MuRIL checkpoint and calibrated baselines, presenting both predictions and auditable cue differences."
- **Claims That Must NOT Be Made:**
  - Do NOT claim the UI provides legal advice or automated legal certification.

---

## Slide 13: Academic Integrity & Master Corpus Provenance

- **Slide Title:** Data Integrity & Master Corpus Provenance
- **Suggested Visual:** Three-tier integrity enforcement pyramid:
  - Top: `assert_gold()` verification in all evaluation code
  - Middle: `schema.is_gold()` Boolean gating
  - Base: Author draft quarantine (`draft_lmp_score` $\neq$ `human_lmp_score`)
- **Main Bullet Points:**
  - **The Integrity Rule:** Machine drafts or author suggestions must NEVER be treated as human labels.
  - The master corpus was intentionally modified once by the reviewed human-gold merge and remained unchanged afterward.
  - Pre-merge backup: `data/processed/lmp_pairs.premerge.20261117T171541Z.csv` (SHA-256 `9F53CC5DEBAA2178C15BD7289EF920DFB8B7399CAAA23A3F30673F54E5385CE5`).
  - Post-merge live corpus: `data/processed/lmp_pairs.csv` (SHA-256 `7327EBC8A519179DA648BE11DD5531D448934760B82850D36B214BC2AA12A216`).
  - Exactly 340 human gold rows populated (320 `human_single`, 20 `human_consensus`).
  - Inter-annotator agreement: measured on 20 overlapping pairs ($\kappa_w = 0.773, \alpha = 0.7015, \text{ICC} = 0.782$).
  - Zero cross-split leakage verified (0 source groups span multiple splits).
  - **Repository Status:**
    - Test Suite: **974 passed** (0 failures, 0 skips).
    - Data Validation: PASS (0 errors).
- **Speaker Notes:**
  > "Integrity has been the governing discipline of this project. The master corpus was intentionally modified once by the reviewed human-gold merge and remained unchanged afterward. Pre-merge backups are cryptographically preserved. All 340 rows contain genuine human gold labels, with substantial agreement measured across the double-annotated subset. Our test suite passes all 974 tests with zero failures."
- **Claims That Must NOT Be Made:**
  - Do NOT claim that the master corpus was never modified; state accurately that it was intentionally modified once by the reviewed gold merge and remained unchanged afterward.

---

## Slide 14: Empirical Evaluation Results on Human Gold Test Split

- **Slide Title:** Empirical Evaluation Results on Human Gold Test Split
- **Suggested Visual:** Results table comparing models on 56 held-out test pairs:
  - TF-IDF Baseline (Train-calibrated): Pearson 0.491 [0.170, 0.703], Spearman 0.411 [0.036, 0.693], RMSE 2.261 [1.794, 2.675], Decision Acc 0.446, Macro-F1 0.429
  - SBERT Baseline (Train-calibrated): Pearson 0.285 [-0.016, 0.540], Spearman 0.302 [0.040, 0.548], RMSE 2.426 [2.012, 2.766], Decision Acc 0.375, Macro-F1 0.384
  - JUDGEBERT / MuRIL Cross-Encoder: Pearson 0.468 [0.245, 0.655], Spearman 0.409 [0.158, 0.612], RMSE 2.728 [2.404, 3.018], Decision Acc **0.607**, Macro-F1 **0.528**
- **Main Bullet Points:**
  - Evaluated on locked 15% test split (56 pairs, 0 cross-split source group leakage).
  - Linear-calibrated lexical overlap (TF-IDF) achieved lower observed RMSE (2.261 vs 2.728).
  - JUDGEBERT cross-encoder showed higher observed accuracy (0.607 vs 0.446) and achieved the highest observed Macro-F1 (0.528 vs 0.429) on three-way legal decision categorization.
  - Cross-lingual evaluation (Exp 2): JUDGEBERT achieved higher correlation on English ($r=0.580$) than Hindi ($r=0.382$), reflecting both cross-lingual representation differences and the author-rendered provenance asymmetry.
  - 1,000 cluster-bootstrap resamples grouped by `source_group` provide robust empirical confidence intervals.
- **Speaker Notes:**
  > "Evaluating on the 56 held-out test pairs reveals complementary model behaviors. TF-IDF with train-only calibration achieved lower observed RMSE, benefiting from lexical overlap penalties. However, on the 3-way legal decision task, the JUDGEBERT cross-encoder showed higher observed accuracy at 60.7% and achieved the highest observed Macro-F1 of 0.528, correctly identifying critical meaning alterations where surface lexical similarity is high."
- **Claims That Must NOT Be Made:**
  - Do NOT claim statistical superiority or definitive victory; use precise observational language ("highest observed Macro-F1", "lower observed RMSE").

---

## Slide 15: Conclusion & Engineering Deliverables

- **Slide Title:** Conclusion: Complete Verified System
- **Suggested Visual:** Project completion checklist with verified checkmarks:
  - ✅ Data Pipeline & Grouped Stratification (159 groups, 0 leakage)
  - ✅ Preprocessing & Legal Cue Extraction Engine (9 cue categories)
  - ✅ Human Gold Annotation & Agreement ($\kappa = 0.773, \alpha = 0.7015$)
  - ✅ Calibrated Baselines & MuRIL Cross-Encoder Checkpoint
  - ✅ Empirical Evaluation (Experiments 1–5 Completed on Human Gold)
  - ✅ Two-Tiered Explainability & Streamlit Demonstration App
  - ✅ 974 Passing Verification Tests
- **Main Bullet Points:**
  - Complete, reproducible engineering foundation established for Indian multilingual LMP.
  - Supervised training and baseline comparisons completed using genuine human gold labels.
  - Verified by 974 automated tests with zero failures.
  - Transparent provenance and limitations documented (Hindi author-rendering, 20-pair multi-annotator overlap, heuristic attention disclaimer).
  - Open, reproducible artifacts: configurations, checkpoints, prediction manifests, and interactive UI.
- **Speaker Notes:**
  > "In conclusion, this project delivers a complete, leak-free, mathematically calibrated, and auditable multilingual NLP system for legal meaning preservation. We have integrated human gold annotations, trained and evaluated baselines alongside a MuRIL cross-encoder, implemented a two-tiered explainability engine, and verified everything with 974 automated tests. Thank you, and we welcome your questions."
- **Claims That Must NOT Be Made:**
  - Do NOT claim commercial readiness; emphasize research prototype nature and the need for licensed legal counsel in actual statutory reformulations.
