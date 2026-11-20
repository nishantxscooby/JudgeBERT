# Comprehensive Viva Voce Defense Question Bank

**Course:** CSET 346 — Natural Language Processing  
**Project:** Explainable Multilingual Legal Meaning Preservation Using a JUDGEBERT-Inspired Cross-Encoder  
**Scope:** 55 Technical and Methodological Questions & Answers for Defense Examination  

---

## 1. Project Motivation & Theoretical Foundation

### Q1. What is the Legal Meaning Preservation (LMP) task?
**Answer:** Legal Meaning Preservation (LMP) is the task of determining whether a simplified, reformulated, or translated legal sentence preserves the exact legal rights, liabilities, obligations, and enforceability of the original authoritative statutory text, evaluated on a continuous 1 to 10 scale.

### Q2. Why is legal similarity fundamentally different from standard semantic textual similarity (STS)?
**Answer:** Standard STS evaluates topical, conceptual, and distributional semantic closeness. In law, two sentences can share 95% of their vocabulary and topics but convey completely opposite legal consequences. For example, changing *"The tenant shall pay"* to *"The tenant may pay"* maintains extremely high semantic embedding similarity while flipping a mandatory legal duty into a discretionary privilege.

### Q3. Why did you choose English and Hindi?
**Answer:** India is a multilingual legal jurisdiction where statutory law is promulgated in English and Hindi, and where citizens frequently interact with Hindi translations or vernacular plain-language summaries of English acts. Evaluating LMP across English and Hindi addresses a real-world statutory communication challenge in the Indian legal system.

### Q4. Why focus specifically on Indian statutory law domains?
**Answer:** Indian statutes offer structured, authoritative, public-domain text with numbered sections and verified legal force. We selected the Indian Contract Act (1872), the Information Technology Act (2000), and the Sale of Goods Act (1930) because they represent well-defined civil, commercial, and cyber law regimes with authentic provisions.

---

## 2. Dataset Design & Corpus Integrity

### Q5. How was the 340-pair corpus constructed?
**Answer:** The corpus was authored from authentic statutory sections across three Acts. Authentic provisions were extracted or transcribed, and deliberate legal reformulations were written by the corpus author to cover diverse legal phenomena (preserving meaning, weakening obligations, flipping conditions, altering numbers/dates) mapped to an 11-category error taxonomy.

### Q6. Why are there exactly 340 pairs in the corpus?
**Answer:** 340 pairs (170 English, 170 Hindi) represents a balanced, carefully curated diagnostic corpus where every sentence pair is manually traced to a specific statutory section. Due to the high cost of authentic legal formulation and the requirement to balance 11 error phenomena and 3 legal domains, 340 high-quality pairs were authored rather than scraping noisy, unverified web text.

### Q7. What are the three legal domains and their pair distributions?
**Answer:**
- Contract Law (*Indian Contract Act, 1872*): 202 pairs (59.4%)
- Cyber Law (*Information Technology Act, 2000*): 72 pairs (21.2%)
- Sale of Goods (*Sale of Goods Act, 1930*): 66 pairs (19.4%)

### Q8. What is `source_group` and why is it the foundation of your anti-leakage mechanism?
**Answer:** `source_group` is an identifier grouping all sibling reformulations that derive from the same underlying statutory provision. If multiple simplifications exist for Section 73 of the Contract Act, assigning them independently to train and test would allow the model to memorize the source text. Grouping splits by `source_group` guarantees that an entire statutory provision and all its variants exist in exactly one split.

### Q9. How do you verify that there is zero cross-split leakage?
**Answer:** In `split_diagnostics.md` and automated tests (`test_split_data.py`), we assert that the intersection of `source_group` sets across train, validation, and test is strictly empty ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$). Across all 159 source groups, 0 groups cross splits.

### Q10. Why are identical source sentences appearing in multiple rows not considered leakage?
**Answer:** Identical source sentences are paired with *different* candidate simplifications to test minimal pair discrimination. As long as all pairs sharing that source sentence are kept within the *same* split (e.g. all in train or all in test), no leakage occurs across split boundaries.

---

## 3. Annotation Workflow & Data Integrity Rules

### Q11. Why are `draft_lmp_score` and `draft_error_type` not treated as ground truth?
**Answer:** Draft scores were created by the corpus author during sentence formulation as suggestions. Treating the author's own drafts as ground truth would introduce severe circular confirmation bias. Ground truth requires independent human annotators evaluating pairs blind.

### Q12. Why can AI-generated annotations (e.g. from Ollama/LLMs) not be presented as human ground truth?
**Answer:** Presenting AI-generated labels as human labels is scientific fraud. LLMs have known blind spots in legal reasoning, hallucinate deontic boundaries, and exhibit prompt sensitivities. In our architecture, AI labels are quarantined in pilot evaluation files and never written to gold columns.

### Q13. What is the difference between `human_lmp_score`, `error_type`, and `decision`?
**Answer:**
- `human_lmp_score`: The continuous 1–10 numerical score assigned by independent human annotators.
- `error_type`: A single-label categorical error chosen from an 11-category legal taxonomy (e.g., Omission, Condition, Modal shift).
- `decision`: A derived 3-way categorical judgment (`Preserved`, `Partially Preserved`, `Not Preserved`) mapped from the numerical score via calibrated thresholds.

### Q14. What is the blind-mode guarantee in your annotation app?
**Answer:** In `src/annotation/annotation_app.py`, annotators are never shown draft scores or other annotators' scores—neither before nor after saving a score. This prevents anchoring bias.

### Q15. How is inter-annotator agreement measured in your system?
**Answer:** Using three complementary metrics in `src/annotation/agreement.py`:
- Cohen's Quadratic Weighted Kappa ($\kappa_w$) for ordinal agreement with distance penalty.
- Krippendorff's Alpha ($\alpha$) on ordinal metric for handling missing annotations.
- Two-way random intraclass correlation ICC(2,1).
Across the 20 overlapping pairs annotated by both independent annotators (`TM01` and `TM02`), the measured agreement is:
- $\kappa_w = 0.773$ (substantial agreement)
- $\alpha = 0.7015$ (reliable ordinal agreement)
- $\text{ICC}(2,1) = 0.782$ (good inter-rater reliability)
Pairs with a difference $\ge 2$ points were subsequently adjudicated via consensus.

### Q16. What happens when two annotators disagree?
**Answer:** If the absolute difference between two annotators is $\ge 2$ points, the pair is flagged for adjudication in `annotation_app.py`. A third person (who must be distinct from both original annotators) reviews the pair and records an authoritative consensus score.

---

## 4. Models & Methodological Architecture

### Q17. Why implement a TF-IDF baseline for legal meaning preservation?
**Answer:** TF-IDF establishes an empirical lower bound for lexical overlap. By using both word n-grams (1–2) and character n-grams (3–5), it captures surface token sharing and Devanagari morphological variations, demonstrating how much predictive power stems from simple vocabulary overlap.

### Q18. Why implement Sentence-BERT as a second baseline?
**Answer:** Multilingual Sentence-BERT (`paraphrase-multilingual-mpnet-base-v2`) represents modern dense bi-encoder sentence representations. It tests whether standard semantic embedding similarity (STS) is sufficient to assess legal meaning preservation.

### Q19. Why choose MuRIL as the backbone for your JUDGEBERT-inspired model?
**Answer:** MuRIL (`google/muril-base-cased`) was pre-trained by Google specifically on 16 Indian languages and English using both monolingual text and translated/transliterated pairs. It significantly outperforms standard mBERT and XLM-RoBERTa on Indian language syntax and Devanagari representations.

### Q20. What is a cross-encoder and how does it compare to a bi-encoder for LMP?
**Answer:** A bi-encoder encodes sentences $A$ and $B$ independently into vector embeddings $u, v$ and computes cosine similarity, meaning tokens in $A$ never interact with tokens in $B$. A cross-encoder feeds $A$ and $B$ simultaneously into the transformer with cross-attention across all layers. This allows every token of the simplified text to attend directly to every token of the statutory text, enabling detection of subtle modal shifts or numeric discrepancies. While bi-encoders allow pre-computed indexing, cross-encoders allow token-level cross-attention between the paired sentences, which is particularly relevant for fine-grained statutory shifts.

### Q21. How are sentence pairs encoded into the cross-encoder?
**Answer:** As a single sequence: `[CLS] original_text [SEP] simplified_text [SEP]`. Token type IDs distinguish the two segments, and position embeddings are maintained up to a maximum length of 192 tokens.

### Q22. Why treat LMP as a regression problem rather than a 3-way classification problem?
**Answer:** Legal meaning preservation exists on a continuous spectrum of severity. A regression formulation preserves the natural ordinality and distance between scores (a score of 7 is much closer to 8 than to 2), whereas standard cross-entropy classification treats all misclassifications as equidistant.

### Q23. What is target scaling and why is it used?
**Answer:** Target scaling normalizes the 1.0–10.0 score range to $[0.0, 1.0]$ via $\tilde{y} = \frac{y - 1}{9}$. Constraining targets to $[0, 1]$ aligns with standard sigmoid/linear initialization, prevents exploding gradients, and stabilizes training on compact datasets. During inference, predictions are scaled back via $\hat{y} = 9 \cdot \tilde{y} + 1$ and clipped to $[1.0, 10.0]$.

### Q24. Why use Smooth L1 (Huber) loss instead of standard Mean Squared Error (MSE)?
**Answer:** MSE penalizes large errors quadratically, making model weights hypersensitive to occasional extreme annotator variances. Smooth L1 behaves quadratically for small errors ($|y - \hat{y}| < 0.5$) and linearly for large errors, providing robustness against outlier labels.

### Q25. What is your early stopping and checkpointing strategy?
**Answer:** Training runs for up to 30 epochs with early stopping triggered if validation RMSE fails to improve by at least 0.001 over 5 consecutive evaluation epochs. Only the checkpoint with the lowest validation RMSE is saved (`save_best_only: true`).

### Q26. Why do you freeze encoder weights during Epoch 1?
**Answer:** Fine-tuning an entire transformer with a randomly initialized regression head on a small dataset can cause catastrophic forgetting and gradient shock. Freezing the encoder during the first epoch allows the regression head weights to stabilize before full end-to-end backpropagation begins.

---

## 5. Evaluation Protocol & Baseline Calibration

### Q27. Why must baseline score calibration use training data only?
**Answer:** Raw cosine similarity ranges from $-1.0$ to $1.0$, which cannot be evaluated against a 1–10 scale without linear calibration ($y = \alpha \cdot \text{sim} + \beta$). If calibration were fitted on validation or test data, test labels would leak into the baseline parameterization, invalidating the evaluation.

### Q28. What does `assert_gold()` protect against?
**Answer:** `src/data/load_data.py:assert_gold()` programmatically inspects the dataset before any experiment or supervised training runs. If the dataset contains non-gold statuses or null human scores, it raises a `ValueError: TODO_HUMAN_ANNOTATION`. This prevents accidental execution on draft or synthetic data.

### Q29. What is Experiment 3 (Held-Out-Domain Generalization)?
**Answer:** Held-out-domain generalization: trained using contract and sale_of_goods data, then evaluated across the complete cyber domain, which is absent from training. The complete cyber domain comprises 72 pairs (48 train, 12 validation, and 12 test rows under the standard grouped split; the official corpus test split specifically contains 12 cyber rows). Cyber law contains distinct technical terminology and statutory mechanisms not present in 19th-century commercial acts.

### Q30. Why is `cyber` selected as the held-out domain rather than `contract`?
**Answer:** `contract` represents the largest domain (202 pairs). Holding out `contract` would leave insufficient training data. `cyber` (72 pairs: 48 train, 12 val, 12 test; official test split contains 12 cyber rows) provides a complete held-out statutory domain to evaluate generalization when trained strictly on contract and sale_of_goods provisions.

### Q31. What metrics are computed to evaluate continuous LMP prediction?
**Answer:**
- **Pearson correlation ($r$):** Linear correlation between predicted and true scores.
- **Spearman rank correlation ($\rho$):** Monotonic ranking accuracy (essential for ranking candidate simplifications).
- **RMSE:** Root Mean Squared Error (penalizes large mispredictions).
- **MAE:** Mean Absolute Error (average point error on the 1–10 scale).
- **$R^2$:** Variance explained by the model.

### Q32. What metrics are computed for the 3-way legal decision?
**Answer:**
- **Decision Accuracy:** Exact agreement with the 3-way gold decision.
- **Macro-Averaged F1:** Harmonic mean of precision and recall averaged equally across `Preserved`, `Partially Preserved`, and `Not Preserved`. Macro-F1 prevents the majority class from masking poor performance on critical minority classes.

### Q33. Why do you use cluster-bootstrap confidence intervals?
**Answer:** The test set contains 56 pairs. Point estimates on small samples have high sampling variance. We perform 1,000 cluster-bootstrap iterations grouped by `source_group` to compute empirical 95% confidence intervals, providing uncertainty bounds on observed model performance and guarding against artifacts of test set selection.

### Q34. What is Experiment 4 and why can it run without human gold labels?
**Answer:** Experiment 4 is a behavioral diagnostic sanity check running on controlled synthetic minimal pairs (identical, unrelated, modal swap, negation flip). It tests whether model mechanics respond logically to known perturbations (e.g. verifying that identical pairs score higher than unrelated pairs). Because it tests algorithmic sensitivity rather than empirical prediction accuracy against human truth, it is labeled `DIAGNOSTIC_ONLY`.

---

## 6. Explainability Architecture

### Q35. What is the primary explanatory signal in your system?
**Answer:** The primary signal is rule-based legal cue alignment implemented in `src/explainability/cue_alignment.py`. It deterministically compares original and simplified provisions across 9 salient legal categories and reports exact character-level deltas.

### Q36. What are the 9 categories of legal cues extracted?
**Answer:**
1. Modal strength & deontic force (Obligation, Permission, Prohibition, Recommendation)
2. Negations and polarity markers
3. Condition markers (`if`, `unless`, `provided that`)
4. Exception clauses (`except`, `save that`, `other than`)
5. Quantities and statutory durations (`30 days`, numeric values)
6. Calendar dates and statutory years
7. Legal entities, courts, sections, and structural actors
8. Quantifiers (Universal: `all`, `every`; Existential: `any`, `some`; Negative: `none`)
9. Length and compression ratios

### Q37. Why are deontic modals so critical in statutory evaluation?
**Answer:** Deontic modals define the legal character of a rule. Shifting "shall" (strength 3, mandatory duty) to "may" (strength 1, discretionary option) relieves a party of statutory liability. Our cue alignment system explicitly categorizes this as a `weakened` modal shift with a high-severity finding.

### Q38. How does the system disambiguate the English word "may"?
**Answer:** In English, "may" can be a deontic modal verb ("party may terminate") or a calendar month ("31st May 2020"). Our extractor checks capitalization and proximity to day numbers and year tokens to ensure calendar months are classified under `Date` rather than `Modal`.

### Q39. Why is token-level attention attribution considered supporting evidence only?
**Answer:** Deep self-attention weights reflect representation routing in multi-head attention layers, not faithful legal reasoning or counterfactual causality. Multiple NLP interpretability papers (e.g., Jain & Wallace, Wiegreffe & Pinter) have shown that attention does not necessarily correlate with feature importance.

### Q40. What is the exact mandatory attention disclaimer in your application?
**Answer:**
> *"Attention weights are not guaranteed to be faithful legal explanations. They are reported as weak supporting evidence only, and are never presented as a legal justification."*

### Q41. How does your HTML highlight renderer prevent cross-site scripting (XSS)?
**Answer:** In `cue_alignment.py:render_highlighted_html`, all plain text segments, matched surface tokens, and title attributes are strictly sanitized using Python's `html.escape(..., quote=True)` before being wrapped into `<mark>` spans.

---

## 7. Interactive Demonstration (Streamlit App)

### Q42. What is the purpose of the Streamlit application?
**Answer:** The Streamlit app (`src/app/streamlit_app.py`) provides an interactive demonstration and diagnostic inspection interface. Users can input sentence pairs in English or Hindi, inspect curated statutory provisions, view machine predictions when available, and explore auditable legal cue alignments with visual highlights.

### Q43. What happens in the UI if no trained model checkpoint exists?
**Answer:** In `model_registry.py`, the system detects missing checkpoints in `models/checkpoints/` and displays an explicit, actionable error message:
> *"JUDGEBERT checkpoint unavailable. Train a valid checkpoint after human annotation is available."*
It refuses to fabricate dummy fallback numbers. The primary cue alignment continue to function independently.

### Q44. How does the UI prevent leakage of draft labels when browsing curated samples?
**Answer:** `load_curated_samples()` in `src/app/ui_helpers.py` projects strictly onto non-evaluative columns (`pair_id`, `language`, `legal_domain`, `original_text`, `simplified_text`, `source_section`). It actively strips `draft_lmp_score`, `draft_error_type`, and all potential label fields, ensuring annotators or users cannot see expected scores.

### Q45. Why is the "Human Annotation Pending" badge displayed on every screen?
**Answer:** To guarantee transparency. Any user interacting with the prototype must immediately understand that the displayed predictions are model outputs or diagnostics, and that no human-validated performance claims have been certified.

### Q46. How are tests run on the Streamlit app without launching a browser or server?
**Answer:** We use Streamlit's official testing framework `streamlit.testing.v1.AppTest`. `AppTest.from_file()` executes the app script in an in-memory simulated runtime, allowing assertions on rendered text, buttons, selectboxes, and error states directly in pytest.

---

## 8. Limitations, Provenance & Scientific Ethics

### Q47. What is the official Hindi PDF font corruption issue?
**Answer:** All official Government of India Hindi statute PDFs obtained use legacy non-Unicode font encodings that map glyphs to incorrect Unicode points. For example, the word अधिनियम ("Act") decodes as अविवियम or अगिगनयम. Furthermore, character mappings are non-invertible (e.g., `ध` maps to two different characters on the same page).

### Q48. How was the Hindi corpus authored in response to this corruption?
**Answer:** To ensure authentic statutory phrasing, the corpus author manually typed the 170 Hindi statutory sentences directly from official printed gazettes into Unicode Devanagari. Every Hindi row carries the explicit provenance metadata `verbatim_quote: "false"`.

### Q49. What is the scientific consequence of this Hindi provenance difference?
**Answer:** English rows represent directly extracted text, whereas Hindi rows represent author-transcribed text. Therefore, cross-lingual performance comparisons in Experiment 2 capture both a language effect and an authoring/provenance effect. This is documented as a known limitation rather than concealed.

### Q50. Can this software be used by a law firm or court today to certify legal simplifications?
**Answer:** No. It is an academic research prototype. Section 1 of the app and reports explicitly state that it does not provide legal advice. Legal meaning preservation must always be verified by qualified legal counsel.

### Q51. What stops you from simply reporting the draft scores as human results?
**Answer:** Scientific ethics and automated code gates. Reporting author drafts as human ground truth would violate the core research principle of independent verification. Furthermore, `assert_gold()` in `src/data/load_data.py` actively halts any script that attempts to evaluate on non-gold data.

### Q52. How was the testing strategy structured across phases?
**Answer:** The automated testing suite (974 tests) was built test-driven to verify the entire engineering and methodological infrastructure: schema validation, grouped splitting, anti-leakage constraints, tokenization, cue extraction, linear calibration logic, provenance logging, model architectures, experiment runners, explainability formatters, and UI components. Pre-annotation assertion tests validate pre-gold blocking behavior using isolated test fixtures; post-merge, 974 automated tests continuously verify all functionality, data structures, and pipeline components with 0 failures.

### Q53. What was the exact workflow executed for human annotation and final evaluation?
**Answer:**
1. Independent annotators scored pairs blind, recording CSV files in `data/annotations/annotators/`.
2. Inter-annotator agreement was measured on the 20 overlapping pairs (`python -m src.annotation.agreement --report`), achieving $\kappa_w = 0.773, \alpha = 0.7015, \text{ICC} = 0.782$.
3. Discrepant pairs ($\ge 2$ point spread) were adjudicated to consensus.
4. The master corpus was intentionally modified once by the reviewed human-gold merge and remained unchanged afterward. Pre-merge backup: `data/processed/lmp_pairs.premerge.20261117T171541Z.csv` (SHA-256 `9F53CC5DEBAA2178C15BD7289EF920DFB8B7399CAAA23A3F30673F54E5385CE5`); post-merge live gold: `data/processed/lmp_pairs.csv` (SHA-256 `7327EBC8A519179DA648BE11DD5531D448934760B82850D36B214BC2AA12A216`).
5. Gold columns were populated via `python -m src.annotation.merge_annotations --apply --backup --report` (320 `human_single`, 20 `human_consensus`).
6. The MuRIL cross-encoder was trained on gold train rows (`python -m src.models.train_judgebert`), saving best checkpoint at `models/checkpoints/judgebert/checkpoint.pt`.
7. Baselines (TF-IDF, SBERT) and Experiments 1–5 were executed on human gold (`python -m src.experiments.run_all`), producing publication tables and summary manifests in `results/experiments/`.

### Q54. What are the key limitations of your system?
**Answer:**
1. Compact double-annotation coverage: 20 pairs double-annotated for agreement, while 320 pairs are single-annotated.
2. Compact overall corpus size (340 pairs) due to authentic manual legal authoring and domain balancing.
3. Hindi vs English provenance asymmetry (author-transcribed Unicode Devanagari due to official Hindi PDF font corruption vs direct PDF extraction for English).
4. Attention attribution cannot be treated as faithful legal justification (supported by mandatory disclaimers).
5. "JUDGEBERT-inspired" cross-encoder architecture rather than an exact replica of Grabar et al.'s French insurance models.

### Q55. What is the single most important achievement of this project?
**Answer:** Delivering a complete, leak-free, mathematically calibrated, and auditable multilingual NLP engineering system that enforces strict scientific integrity by refusing to report unvalidated or fabricated findings.
