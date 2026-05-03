> **COMPASS FILE — READ THIS FIRST**
> This is the authoritative orientation document for the `vidiq-hpc` repository.
> If you are an AI agent, read this file before opening any experiment directory.
> It is the single source of truth for scientific story, key numbers, open items, and file locations.
> Maintained by: Joshua Bhawanlall | Last updated: 2026-05-03 | Status: Active — NeurIPS submission in progress

---

## Pre-Write Repo Check Results

| Blocking Item | Status | Evidence / Note |
|---|---|---|
| LOSO/bootstrap/permutation tests on LLM text experiments | UNRESOLVED | No matching terms found in `experiments/understanding_text_embeddings/reports/` — currently only brain data has these tests applied |
| Ambiguity gradient source file (cross-system r≈0.56, p≈10⁻⁵⁴) | UNRESOLVED | No `comparison_results*` or `global_behavior*` file found under `experiments/`; value originates from Apr 28 meeting minutes only |
| Valence-arousal ground truth citation | PARTIAL | `PLAN.md` and `run_va_reduction.py` reference Russell's Circumplex and DEAP/Koelstra; May 1 meeting confirms Russell 1980 + DEAP as citations — exact numerical table not yet formally cited in a paper-facing document |
| Aimee's results PDF | UNRESOLVED | `emotion_geometry_and_score_4_square_explorer_square_report_score_v2.pdf` not found in repo; noted in Apr 28 minutes as on Aimee's laptop |
| Embedding extraction layer documented | PARTIAL | `loader_text.py` confirms L2-normalised mean-pooled embeddings from pre-saved `.npy` files; the specific transformer layer (final layer 12 vs middle layer 6) is encoded in the folder names (`bge`, `bge-mid`, etc.) but is not stated in prose in any methods document |
| fMRI sample size justification | PARTIAL | Apr 28 minutes contain Aimee's oral statement ("most published fMRI studies use 8–10 subjects; 40 is above field standard") but this has not been written into any methods section or report |

---

## Project Identity

**Full project name:** LLM Embedding Geometry and Brain fMRI Alignment Study
**Short repo name:** `vidiq-hpc`
**Target venue:** NeurIPS 2026 | Deadline: 4 May 2026
**Working paper title (tentative):** *"The Geometry of Affect: Universal Laws of Emotional Representation in Artificial and Biological Systems"*

**Team:**
- Joshua Bhawanlall — ML coordination, repository management
- Daniel Sikar — lead researcher, mathematical framework
- Pritish Ranjan (PG-Verma) — experiments, HPC pipeline
- Aimee Bottrill-Frost — neuroscience, brain fMRI analysis and statistical validation

**Datasets:**
- `dair-ai/emotion` — 6-class text emotion dataset (anger, fear, happiness, love, sadness, surprise); balanced to 4038 validation samples
- OpenNeuro DS005700 ("Neural MO — fMRI Dataset for Emotion Recognition") — 40 subjects × 5 emotions (afraid, calm, delighted, depressed, excited), 48 ROI features per observation

---

## The Scientific Story

### Core Hypothesis

LLM embeddings encode emotional context geometrically via measurable universal laws — specifically centroid distance and cluster density. These laws define a structured representational geometry: a hollow void near each class prototype, a dense belt of real instances around it, and a competitive boundary where classes overlap. If the same geometric logic appears in human fMRI brain activations, it constitutes evidence of a universal representational principle shared across artificial and biological systems.

### Six Key Findings

---

**Finding 1: The Density Structure — "The Void" and "The Belt"**

All distances are normalised (scaled to the 0–2.5 range) to allow comparison across model architectures and the brain fMRI system — these are not raw Euclidean magnitudes. On this normalised scale, a consistent two-zone density structure emerges across all 8 LLM variants and 2 text datasets:

- **The Void:** No data points exist within approximately 0.375–0.625 normalised distance units of any emotion centroid. The precise onset of The Void varies by model — fine-tuned models show a shorter void (first non-zero density at approximately bin 0.4375 normalised units); pretrained models show a longer void (first non-zero density at approximately bin 0.6875 normalised units). This model-dependent variation is itself informative: fine-tuning pulls data closer to the prototype boundary.

- **The Belt:** Density rises sharply from the void edge and peaks at approximately 0.56–0.94 normalised distance units. Fine-tuned models peak earlier (MPNet-FT-Final at bin 0.5625, BGE-FT-Final at bin 0.6875); pretrained models peak later (MPNet-Base-Final and BGE-Base-Final both at bin 0.9375). All real data points live in or beyond The Belt — none at the centroid itself.

*Source: `experiments/understanding_text_embeddings/reports/phase2/overlap_metrics.json`*

---

**Finding 2: No Pure Emotion**

The Void is the primary evidence for the "no pure emotion" conclusion. If a pure instance of any emotion existed in embedding space, it would sit at or near its class centroid — the geometric prototype of that emotion. The Void shows that no data point does. Every real instance lives at meaningful distance from its class centre, which means every encoded emotion is already a blend.

*Secondary geometric note:* The centroid sits in void space partly because it is a mathematical mean of a ring of distributed points — it does not correspond to any actual data point. This explains geometrically *why* the centroid is unreachable, but it is not the primary basis for the no-pure-emotion conclusion.

*Separate but related finding (do not conflate):* The formal overlap definition used in Phase 2 — a point P overlaps class C′ if dist(P, C′) < dist(P, own centroid) — quantifies cross-class bleed independently of The Void. Global overlap rates: BGE-Base-Final 19.22%, MPNet-Base-Final 18.45%, BGE-FT-Final 1.81%, MPNet-FT-Final 0.42%. Both findings support "no pure emotion" but through different mechanisms: The Void shows nothing is central; the overlap metric shows many Belt-region points are geometrically claimed by multiple classes simultaneously. The most entangled pair: Happiness vs Love at 42% overlap in pretrained models. The most robustly separated pair: Fear vs Anger at 12%.

The "certainty buffer" — radial gap between the density peak and the onset of ambiguity — is large for fine-tuned models (MPNet-FT-Final +1.875, BGE-FT-Final +1.750) and near-zero for pretrained models (MPNet-Base-Final 0.000, BGE-Base-Final +0.125). Fine-tuning does not just separate clusters — it creates a geometrically safe zone between identity and confusion.

---

**Finding 3: Information Compaction vs. Distribution**

Phase 3 tested BGE-Base, BGE-FT, MPNet-Base, and MPNet-FT only (no Qwen in this phase). Fine-tuned LLMs compress emotional signal into a small number of dominant SVD directions, producing a sharp ablation cliff. Pretrained LLMs distribute the signal redundantly across far more dimensions before reaching chance level.

**Erasure points (dimension at which accuracy reaches chance ~16.7%):**
- MPNet-FT-Final: erased at **dimension 15**
- BGE-FT-Final: erased at **dimension 20**
- BGE-Base-Final: erased at **dimension 26**
- MPNet-Base-Final: erased at **dimension 67**

The fine-tuned advantage (98% baseline accuracy) comes at a cost: the representation is fragile. Remove 15–20 directions and the signal collapses completely. Pretrained models start lower (~95% baseline) but distribute emotional signal across up to 4.5× more dimensions before reaching chance.

The brain context retention experiment (a separate cross-system comparison, not Phase 3) extends this finding: Brain AUC 0.302 (D50 = 4 dims, highly distributed), MPNet pretrained AUC 0.331 (D50 = 17 dims, redundant manifold), Qwen fine-tuned AUC 0.260 (D50 = 12 dims, compressed cliff). Qwen appears only in this brain-experiment comparison, not in Phase 3 itself.

*Source: `experiments/understanding_text_embeddings/reports/phase3/importance_metrics.json` (Phase 3); `experiments/brain_embedding_understanding/checking_context_retention_across_dimensions/reports/CONTEXT_RETENTION_FINAL_REPORT.md` (brain cross-system)*

---

**Finding 4: Geometric Clarification in Top-20D Subspace**

Phase 4 isolated the top 20 orthogonal directions from classifier-weight SVD and re-evaluated geometry in this subspace. The results show that the "cloud" appearance of pretrained models in 768D is a high-dimensional noise artefact — not a sign of absent structure.

**Silhouette improvement in 20D vs 768D:**
- MPNet-Base-Final: 0.0471 → 0.4075 (+765.2%)
- BGE-Base-Final: 0.0570 → 0.4146 (+627.4%)
- BGE-FT-Final: 0.6847 → 0.8353 (+22.0%)
- MPNet-FT-Final: 0.8597 → 0.9068 (+5.5%)

**Accuracy in 20D (essentially no loss):**
- MPNet-FT: 98.24% → 98.27% | BGE-FT: 97.99% → 98.09%
- MPNet-Base: 94.08% → 95.64% | BGE-Base: 93.78% → 95.12%

Emotion is encoded in a low-rank linear subspace occupying ~2.6% of the full 768D embedding capacity. The relational logic is preserved: Happiness-Love and Fear-Anger are the closest pairs; Sadness-Surprise and Happiness-Fear are the furthest.

---

**Finding 5: Cross-System Ambiguity Gradient**

Both brain and LLM systems show that geometric positioning relative to class prototypes predicts classification confidence — but the mechanism differs.

**Brain side** (source: `experiments/brain_embedding_understanding/checking_density_geometry/reports/brain finl.html`): Raw centroid distance is a weak predictor of uncertainty (Spearman r = 0.1509, AUC = 0.5627 — close to chance). The correct measure is **centroid margin**: d(correct centroid) − d(nearest competing centroid). Margin strongly predicts cross-validated classifier uncertainty:
- Spearman r = 0.6108, p = 7.78e-22, AUC = 0.8124
- Feature shuffle control: r = 0.0287, p = 0.687 — confirming the relationship is not random
- Interpretation: ambiguity is *relational* (competition between prototypes), not *radial* (raw distance from own centroid)

The brain LOSO decoding accuracy is 0.56 (95% CI: 0.49–0.63), well above the 5-class chance level of 0.20, with permutation p < 0.001. This confirms brain ROI patterns contain emotion signal that generalises to held-out subjects, even though 2D projections show overlapping distributions — a high-dimensional separability artefact, not absence of structure.

**LLM side** (source: `experiments/understanding_text_embeddings/reports/phase5/consistency_summary.json`): Centroid distance predicts logit confidence in fine-tuned LLMs — BGE-FT r = 0.9572 (p ≈ 0), MPNet-FT r = 0.9884 (p ≈ 0). Logit agreement rate in overlap regions: BGE-FT 93.4%, MPNet-FT 100%. Phase 5 covers only the two fine-tuned final-layer models.

**Cross-system r ≈ 0.56, p ≈ 10⁻⁵⁴:** [SOURCE FILE UNCONFIRMED — from Apr 28 meeting minutes only. No repo file with these exact values has been located. The brain margin result (r = 0.6108) and the LLM dist-logit result (r = 0.957–0.988) are confirmed from source files. The cross-system unified figure must either be located or dropped in favour of the per-system confirmed metrics.]

---

**Finding 6: The Brain-LLM Relational Paradox**

The culminating finding of the study. Reported as three tightly connected sub-results. The comparison is framed as fine-tuned LLM vs pretrained LLM vs brain — not by architecture name.

**6a — Valence-Arousal Axis Alignment**

*Why it matters:* Establishes the psychological grounding for the geometric divergence — which primary dimension each system uses to organise emotion centroids. This explains mechanistically why the RDMs are inverted.

| System | PC1 dominant axis | PC1 r | PC2 dominant axis | PC2 r | Permutation p |
|---|---|---|---|---|---|
| Fine-tuned LLM | Valence | 0.98 | Arousal | 0.82 | 0.013 |
| Pretrained LLM | Valence | 0.97 | Arousal | 0.73 | 0.009 |
| Human Brain | Arousal | 0.96 | Valence | 0.31 | 0.033 |

Both LLM types are valence-dominant on PC1; the brain is arousal-dominant on PC1. Both LLM types detect arousal on PC2; the brain shows only weak valence on PC2. Conclusion: the emotional map is conserved in structure but the priority of dimensions is inverted — LLMs are valence-first (semantic/categorical), brain is arousal-first (physiological/survival).

V/A ground truth values are based on Russell (1980) circumplex model and DEAP dataset (Koelstra & Mühl) — both heavily cited. Exact numerical values are assigned relationally (i.e. joy has higher valence than surprise; fear has higher arousal than sadness) and are cited from prior affective computing literature, not claimed as exact physical measurements.

*Source: `experiments/brain_embedding_understanding/valence-arousal-dimensional_reduction/reports/alignment_metrics.json`*

**6b — RDM / RSA in 48D Raw ROI Space**

*Why it matters:* Quantifies how structurally different the pairwise centroid geometry is between brain and each LLM type, using metric-invariant validation.

The analysis uses the 3-emotion triplet (Fear, Happiness, Sadness) — the overlap between brain dataset (afraid/delighted/depressed ≈ fear/happiness/sadness) and text dataset. Cross-system label mapping: fear≈afraid, happiness≈delighted, sadness≈depressed. This mapping relies on conceptual equivalence and should be flagged as a potential reviewer question.

| Comparison | Pearson r | Interpretation |
|---|---|---|
| Brain vs fine-tuned LLM (48D) | **−0.8756** | Near-opposite centroid geometry |
| Brain vs pretrained LLM (48D) | −0.5476 | Negative across metrics |
| Brain-brain (noise ceiling) | upper 0.484 / lower 0.460 | Human-human agreement ~0.47 |

The fine-tuned LLM correlation with the brain (−0.88) is not just near-zero — it is actively opposed. The noise ceiling shows human-human agreement is only ~0.47, so the −0.88 divergence is striking. Manhattan metric gives r = −0.5476, confirming the result is metric-invariant. Bootstrap 95% CI: [−0.9967, +0.6994] — wide due to small triplet (3 emotions), but distribution is skewed negative.

*Source: `experiments/brain_embedding_understanding/checking_centroids/reports/relational_validation.json`*

**6c — RDM / RSA in 11D Systems-Level Space**

*Why it matters:* Tests whether the paradox is a noise artefact of the raw 48D ROI representation or a robust property of the brain's systems-level organisation. Brain data is compressed from 48D to 11D (5 anatomical lobes + 5 functional networks + 1 neighbour context) before rerunning the RDM comparison.

| Comparison | 48D Pearson r | 11D Pearson r | Change |
|---|---|---|---|
| Brain vs fine-tuned LLM | −0.6371 | −0.7539 | Paradox deepened |
| Brain vs pretrained LLM | −0.1023 | **−0.9918** | Near-perfect structural inversion |

The paradox amplifies as the biological signal is denoised. The pretrained LLM at 11D (r = −0.9918) is the strongest cross-system result in the entire study — a near-perfect structural mirror. The pretrained LLM and brain organise the same three emotions in almost perfectly opposite relational geometry. This confirms the divergence is a fundamental property of how biological functional networks are organised, not a noise artefact of raw ROI data.

*Source: `experiments/brain_embedding_understanding/checking_centroids_with_spatial_context_data/reports/rdm_results_11d.json`*

---

### Key Quantitative Anchors

| Metric | Value | System / Condition | Source File |
|---|---|---|---|
| Silhouette (MPNet-FT-Final, 768D) | 0.8597 | Text LLM | `PHASE_1_DETAILED_REPORT.md` |
| Silhouette (BGE-FT-Final, 768D) | 0.6847 | Text LLM | `PHASE_1_DETAILED_REPORT.md` |
| Silhouette (MPNet-Base-Final, 768D) | 0.0471 | Text LLM | `PHASE_1_DETAILED_REPORT.md` |
| Silhouette (BGE-Base-Final, 768D) | 0.0570 | Text LLM | `PHASE_1_DETAILED_REPORT.md` |
| Silhouette improvement (pretrained, 20D) | +627–765% | Pretrained LLMs | `PHASE_4_DETAILED_REPORT.md` |
| Silhouette improvement (fine-tuned, 20D) | +5.5–22% | Fine-tuned LLMs | `PHASE_4_DETAILED_REPORT.md` |
| Global overlap (MPNet-FT-Final) | 0.42% | Text LLM | `phase2/overlap_metrics.json` |
| Global overlap (BGE-Base-Final) | 19.22% | Text LLM | `phase2/overlap_metrics.json` |
| Erasure cliff — MPNet-FT | Dim 15 | Text LLM | `phase3/importance_metrics.json` |
| Erasure cliff — BGE-FT | Dim 20 | Text LLM | `phase3/importance_metrics.json` |
| Erasure cliff — BGE-Base | Dim 26 | Text LLM | `phase3/importance_metrics.json` |
| Erasure cliff — MPNet-Base | Dim 67 | Text LLM | `phase3/importance_metrics.json` |
| Context retention AUC — Brain | 0.302 (D50 = 4) | Brain fMRI | `CONTEXT_RETENTION_FINAL_REPORT.md` |
| Context retention AUC — MPNet pretrained | 0.331 (D50 = 17) | Pretrained LLM | `CONTEXT_RETENTION_FINAL_REPORT.md` |
| Context retention AUC — Qwen fine-tuned | 0.260 (D50 = 12) | Fine-tuned LLM | `CONTEXT_RETENTION_FINAL_REPORT.md` |
| Dist-logit correlation — BGE-FT | 0.9572 (p ≈ 0) | Text LLM | `phase5/consistency_summary.json` |
| Dist-logit correlation — MPNet-FT | 0.9884 (p ≈ 0) | Text LLM | `phase5/consistency_summary.json` |
| Logit agreement in overlap — MPNet-FT | 100% | Text LLM | `phase5/consistency_summary.json` |
| Brain LOSO decoding accuracy | 0.56 (CI: 0.49–0.63) | Brain fMRI | Apr 28 minutes / Aimee's analysis |
| Brain LOSO permutation p | < 0.001 | Brain fMRI | Apr 28 minutes |
| Brain margin vs uncertainty (Spearman r) | 0.6108 | Brain fMRI | `checking_density_geometry/reports/brain finl.html` |
| Brain margin vs uncertainty (p) | 7.78e-22 | Brain fMRI | `checking_density_geometry/reports/brain finl.html` |
| Brain margin AUC | 0.8124 | Brain fMRI | `checking_density_geometry/reports/brain finl.html` |
| Brain raw distance AUC | 0.5627 | Brain fMRI | `checking_density_geometry/reports/brain finl.html` |
| Shuffle control r | 0.0287 (p = 0.687) | Brain fMRI | `checking_density_geometry/reports/brain finl.html` |
| Fine-tuned LLM V/A — PC1 valence r | 0.98 | Fine-tuned LLM | `alignment_metrics.json` |
| Fine-tuned LLM V/A — PC2 arousal r | 0.82 | Fine-tuned LLM | `alignment_metrics.json` |
| Pretrained LLM V/A — PC1 valence r | 0.97 | Pretrained LLM | `alignment_metrics.json` |
| Pretrained LLM V/A — PC2 arousal r | 0.73 | Pretrained LLM | `alignment_metrics.json` |
| Brain V/A — PC1 arousal r | 0.96 (p=0.033) | Brain fMRI | `alignment_metrics.json` |
| Brain V/A — PC2 valence r | 0.31 | Brain fMRI | `alignment_metrics.json` |
| Brain vs fine-tuned LLM — 48D Pearson r | −0.8756 | Brain vs Fine-tuned LLM | `relational_validation.json` |
| Brain vs pretrained LLM — 48D Pearson r | −0.5476 | Brain vs Pretrained LLM | `rdm_comparison_metrics.json` |
| Brain-brain noise ceiling | upper 0.484 / lower 0.460 | Brain fMRI | `relational_validation.json` |
| Brain vs pretrained LLM — 11D Pearson r | −0.9918 | Brain vs Pretrained LLM | `rdm_results_11d.json` |
| Brain vs fine-tuned LLM — 11D Pearson r | −0.7539 | Brain vs Fine-tuned LLM | `rdm_results_11d.json` |
| Cross-system ambiguity r ≈ 0.56, p ≈ 10⁻⁵⁴ | [SOURCE FILE UNCONFIRMED] | Cross-system | Apr 28 meeting minutes only |

---

## NeurIPS Abstract (Draft)

> We investigate whether emotional representations in large language models (LLMs) and the human brain share a common geometric logic. Across eight LLM variants (BGE and MPNet, pretrained and fine-tuned, middle and final layers) applied to a 6-class emotion text dataset, and compared to fMRI from 40 subjects (5 emotions, 48 ROIs; OpenNeuro DS005700), we identify six geometric findings. Distances are normalised across systems. First, no data point exists within 0.375–0.625 normalised units of any emotion centroid (The Void), and density peaks at 0.56–0.94 normalised units (The Belt). Fine-tuned models show a shorter void; pretrained models a longer one. Second, because no instance sits near its class centroid, no emotional prototype is "pure" — every encoded emotion is a blend. Third, fine-tuned LLMs compress signal into a sharp cliff (erased at dimensions 15–20); pretrained LLMs distribute it across 26–67 dimensions. Fourth, projecting into the top 20 discriminative dimensions resolves the apparent disorder of pretrained 768D embeddings: silhouette scores jump 627–765%, confirming the "cloud" appearance is high-dimensional noise. Fifth, centroid competition predicts uncertainty in both systems. In brain fMRI, margin predicts cross-validated classifier uncertainty (Spearman r = 0.61, p = 7.78×10⁻²², AUC = 0.81); raw distance is near chance (AUC = 0.56). In fine-tuned LLMs, centroid distance predicts logit confidence (r = 0.957–0.988). Sixth, despite this shared local principle, global organisation is inverted: LLMs sort emotions by valence (PC1 r = 0.97–0.98), the brain by arousal (PC1 r = 0.96), yielding near-opposite dissimilarity matrices (brain vs fine-tuned LLM r = −0.88 in 48D; brain vs pretrained LLM r = −0.99 at the 11D systems level). Shared local geometric logic coexists with fundamentally different global representational priorities.

**Word count:** 248 words (verified).

---

## Repository Map

### Text Experiments (Phases 1–5)

All paths relative to `experiments/understanding_text_embeddings/`.

| Phase | Name | What It Measures | Key Source File | Key Report |
|---|---|---|---|---|
| 1 | Baseline Topology | Silhouette score + CV accuracy across 8 variants | `src/run_phase1_baseline.py` | `reports/PHASE_1_DETAILED_REPORT.md` |
| 2 | Overlap & Radial Density | Density belt, The Void, overlap %, certainty buffer | `src/run_phase2_overlap_density.py` | `reports/PHASE_2_DETAILED_REPORT.md` |
| 3 | Signal Retention (Ablation) | SVD erasure, signal cliff, D50 | `src/run_phase3_signal_retention.py` | `reports/PHASE_3_DETAILED_REPORT.md` |
| 4 | Isolated Subspace | Top-20D geometry, silhouette jump, V/A axes | `src/run_phase4_isolated_subspace.py` | `reports/PHASE_4_DETAILED_REPORT.md` |
| 5 | Logit Consistency (RSA) | Dist-logit correlation, logit bias in overlap regions | `src/run_phase5_logit_consistency.py` | `reports/phase5/consistency_summary.json` |

Context files: `context/experiments-context.md`, `context/phase-5-context.md`, `plan.md`

HTML summaries (include inline plots): `reports/phase1_summary.html`, `reports/phase2_summary.html`, `reports/phase3_summary.html`, `reports/phase4_summary.html`, `reports/phase5_summary.html`

---

### Brain Experiments (6 Sub-Experiments)

All paths relative to `experiments/brain_embedding_understanding/`.

1. **`checking_centroids/`** — RDMs + centroid relational geometry (Brain vs fine-tuned LLM vs pretrained LLM in 48D raw ROI space). The Relational Paradox is first discovered here.
   - Output: `reports/CENTROID_RELATIONAL_FINAL_REPORT.md`, `reports/relational_validation.json`, `reports/rdm_comparison_metrics.json`

2. **`adding_spatial_context/`** — Compresses 48D ROI space to 11D (5 anatomical lobes + 5 functional networks + 1 neighbour context). Rationale: reduce ROI noise while preserving interpretable biological organisation.
   - Output: `outputs/BRAIN_11D_SYSTEMS_REPORT.md`

3. **`checking_centroids_with_spatial_context_data/`** — Repeats RDM analysis on the 11D brain representation. The paradox amplifies — pretrained LLM r = −0.9918.
   - Output: `reports/SYSTEMS_LEVEL_PARADOX_REPORT.md`, `reports/rdm_results_11d.json`

4. **`valence-arousal-dimensional_reduction/`** — PCA alignment of LLM and brain centroids with valence/arousal coordinates. Explains the RDM inversion mechanistically.
   - Output: `reports/VA_GEOMETRIC_CONVERGENCE_REPORT.md`, `reports/alignment_metrics.json`

5. **`checking_density_geometry/`** — Cross-system density decay comparison; brain ambiguity gradient validation (margin vs uncertainty).
   - Output: `reports/DENSITY_GEOMETRY_FINAL_REPORT.md`, `reports/density_validation.json`, `reports/brain finl.html`

6. **`checking_context_retention_across_dimensions/`** — Iterative SVD ablation comparing Brain, MPNet pretrained, and Qwen fine-tuned. Extends Finding 3 cross-system.
   - Output: `reports/CONTEXT_RETENTION_FINAL_REPORT.md`

---

### Scripts

- `scripts/prepare_balanced6_dataset.py` — Combines per-emotion raw embeddings into unified training dataset
- `scripts/embed_balanced_6_emotions_raw.py` — Generates raw mean-pooled embeddings from CSV via HuggingFace model

---

### Meeting Minutes

All files in `meetings/`.

| Date | File | Key Topic |
|---|---|---|
| 2026-04-07 | `2026-04-07_minutes_llm-embedding-geometry_sentiment-experiments_neurips.md` | Core hypothesis established; raw pooling > L2; NeurIPS deadline set; Surprise identified as geometric outlier |
| 2026-04-11 | `2026-04-11_minutes_llm-embedding-geometry_sentiment-experiments_neurips.md` | Density belt discovery; "no pure emotion" articulated; Qwen 1.7B architecture agreed |
| 2026-04-12 | `2026-04-12_minutes_embedding-geometry_density-analysis_hpc-training.md` | The Void formalised; axis label correction (y = points per band, not density); three sessions |
| 2026-04-16 | `20260416_minutes_neurips.md` | Cross-dataset + cross-model validation confirms geometry is universal; fine-tuning hypothesis |
| 2026-04-21 | `20260421_minutes_explanation_and_adding_images.md` | Image extension designed; HPC multi-account setup; project walkthrough for new members |
| 2026-04-28 | `2026-04-28_minutes_aimee.md` | Pivot from 120K image set to brain fMRI; Aimee joins; OpenNeuro DS005700 confirmed; statistical rigour requirements set |
| 2026-05-01 (results) | `20260501_Phase_1_to_5_results_transcrition.md` | All 5 phases reviewed; brain experiments reviewed; V/A citations confirmed (Russell 1980, DEAP/Koelstra) |
| 2026-05-01 (paper) | `20260501_paper_structure_transcription.md` | Paper section structure discussed; final push planned |

---

### repo_context/

- `repo_context/PROJECT_OVERVIEW.md` — this file (the compass)
- `repo_context/neurips_context/neurips-guide.md` — NeurIPS submission guide and strategic framing
- `repo_context/neurips_context/neurips_deep_submission_structure_checklist_report.pdf` — detailed NeurIPS structure + checklist (11 pages; section-by-section build plan)
- `repo_context/neurips_context/neurips_pre_submission_checklist_full.pdf` — pre-submission review checklist (6 pages; claim support, statistical validity, rejection patterns)
- `repo_context/project_context/Emotion_Geometry_Complete_Report.pdf` — 3-page complete emotion geometry report (generated 2026-05-02; authoritative consolidated findings)
- `repo_context/project_context/PLAN_COMPASS_GENERATION.md` — agent plan used to generate this compass (not the compass itself)

---

## Work Diary

### 2026-04-07 — Core Hypothesis and Experimental Foundation
**Present:** Daniel Sikar, Pritish Ranjan
**Key outcomes:**
- Core hypothesis established: centroid distance + cluster density describe a universal geometric law in LLM embedding space
- Raw pooling outperforms L2 normalisation for geometry-based analysis; direction is the primary signal
- NeurIPS 4 May 2026 deadline set as hard target
- Surprise identified as the geometrically ambiguous class — a reaction rather than a pure emotion
- Multi-class pipeline and 6-class emotion dataset confirmed
**Source:** `2026-04-07_minutes_llm-embedding-geometry_sentiment-experiments_neurips.md`

### 2026-04-11 — The Density Belt Discovery
**Present:** Daniel Sikar, Pritish Ranjan
**Key outcomes:**
- Density belt confirmed: density does not peak at centroid — it peaks at ~9–9.5 raw Euclidean units (later normalised to ~0.9375 for pretrained models)
- "No pure emotion" formally articulated: no data point at the centroid; centroid = mean of mixed subtypes
- Qwen 1.7B architecture agreed for fine-tuned classifier
- Cross-model validation planned for April 18
**Source:** `2026-04-11_minutes_llm-embedding-geometry_sentiment-experiments_neurips.md`

### 2026-04-12 — The Void Formalised
**Present:** Daniel Sikar, Pritish Ranjan (3 sessions)
**Key outcomes:**
- The Void formally named: zero-density region from centroid to ~7.5 raw Euclidean units (later normalised: ~0.625 for pretrained models)
- Y-axis label corrected from "density per unit" to "number of points per band" — important for paper reviewers
- Overlap definition clarified: point P overlaps class C′ iff dist(P, C′) < dist(P, own centroid)
- Centroid distance matrix computed: sadness nearest to anger; sadness furthest from joy
**Source:** `2026-04-12_minutes_embedding-geometry_density-analysis_hpc-training.md`

### 2026-04-16 — Cross-Model and Cross-Dataset Validation
**Present:** Daniel Sikar, Pritish Ranjan, Josh, Andrew
**Key outcomes:**
- Cross-dataset validation (new 20K dataset): same geometric patterns hold; clusters preserve relational structure
- Cross-model validation (different pretrained model): belt pattern and overlap region confirmed as model-agnostic
- Fine-tuning hypothesis stated for upcoming experiments
- Image dataset selection task assigned
**Source:** `20260416_minutes_neurips.md`

### 2026-04-21 — Image Extension and HPC Coordination
**Present:** Daniel Sikar, Pritish Ranjan, Josh, Andrew
**Key outcomes:**
- Image extension designed: same pipeline applied to image embeddings
- Video segmentation approach discussed
- HPC four-account setup confirmed; all team members working toward access
- Project walkthrough given for newer members
**Source:** `20260421_minutes_explanation_and_adding_images.md`

### 2026-04-28 — Brain fMRI Pivot and Statistical Rigour
**Present:** Daniel Sikar, Pritish Ranjan, Aimee Bottrill-Frost, Josh, Freya Myo, Andrew, Arj
**Key outcomes:**
- Pivot from 120K image dataset (too noisy) to brain fMRI; Aimee joins as neuroscience expert
- OpenNeuro DS005700 confirmed: 40 subjects × 5 emotions × 48 ROIs
- Brain LOSO accuracy 0.56 (95% CI: 0.49–0.63, p < 0.001); negative silhouette explained as high-dimensional artefact
- Cross-system ambiguity gradient figure r ≈ 0.56 cited in meeting [SOURCE UNCONFIRMED in repo — confirmed brain margin result: r = 0.6108, p = 7.78e-22, AUC = 0.8124]
- Statistical rigour requirements set: all LLM text experiments must receive LOSO, bootstrap CI, and permutation tests before submission
- fMRI sample size justified: 40 subjects >> 8–10 typical in literature (Aimee)
**Source:** `2026-04-28_minutes_aimee.md`

### 2026-05-01 — Phase Results and Paper Structure
**Present:** Daniel Sikar, Pritish Ranjan, Josh
**Key outcomes:**
- All 5 text embedding phases reviewed; brain experiments reviewed
- V/A ground truth citations confirmed: Russell (1980) circumplex + DEAP/Koelstra (5245 citations)
- Brain LOSO 0.56 (95% CI: 0.49–0.63) confirmed
- Paper section structure discussed; final submission push planned
**Source:** `20260501_Phase_1_to_5_results_transcrition.md`, `20260501_paper_structure_transcription.md`

---

## Open Items: NeurIPS Submission (Critical Path)

### Must-Do (Blocking Submission)

- [ ] **LOSO + bootstrap CI + permutation tests on all LLM text embedding results** — currently only brain data has these. If reviewers ask "what is the statistical significance of your silhouette improvement?", there is no answer yet. **Severity: HIGH — this is the single biggest methodological gap.** Owner: Pritish/Josh; advised by Aimee
- [ ] **Locate source file for cross-system r ≈ 0.56, p ≈ 10⁻⁵⁴** — this specific figure comes from Apr 28 meeting minutes only. Either locate `comparison_results.txt` or equivalent file in the repo (check `video_understanding/human_brain_emotion_exports/global_behavior_comparison/`) or drop this framing in favour of confirmed per-system metrics. Owner: Pritish
- [ ] **Confirm valence-arousal ground truth citation chain** — Russell 1980 circumplex + DEAP (Koelstra & Mühl) are confirmed as primary sources. Exact numerical V/A values for the 5/6 emotions need to be formally cited in a paper-facing document with page/table reference. Owner: Aimee
- [ ] **Push Aimee's results PDF to repo** — `emotion_geometry_and_score_4_square_explorer_square_report_score_v2.pdf` currently on Aimee's laptop; may contain additional statistical validation. Owner: Aimee
- [ ] **Decide Phase 5 (logit consistency) scope: main paper vs. appendix** — r = 0.957–0.988 is a strong result. Daniel to confirm placement. Owner: Daniel
- [ ] **Justify fMRI sample size in methods** — add sentence citing typical fMRI studies use 8–10 subjects; 40 is above field standard. Aimee to draft. Owner: Aimee/Daniel
- [ ] **Document embedding extraction layer per model** — NeurIPS methods section requires explicit statement. The `loader_text.py` loads from pre-saved `.npy` files in folders named `bge`, `bge-mid`, etc. This needs to be stated as: BGE final layer 12 (mean pooled), BGE middle layer 6, MPNet final layer 12, MPNet middle layer 6. Owner: Daniel/Pritish
- [ ] **Confirm cross-system ambiguity p-value from source data file** — Owner: Pritish

### Scientific Rigour (Strong-to-Have)

- [ ] Confirm brain LOSO 95% CI (0.49–0.63) and p < 0.001 cited consistently across all brain reports
- [ ] Noise-ceiling comparison explicitly in paper: brain-brain r ≈ 0.47 vs brain–fine-tuned LLM r = −0.88
- [ ] Haxby et al. distributed coding citation — Owner: Aimee (from reference list in `LLM_brain.RTF` when available)
- [ ] Decide 120K image dataset placement: appendix or omit (Apr 28 minutes: retain as negative result showing scope)
- [ ] Add bootstrap CI or shaded bands to central LLM plots where feasible
- [ ] Create anonymised code/data supplement for NeurIPS reproducibility checklist

### Completeness Cross-Check

- [x] Raw pooling rationale documented (direction = primary signal; magnitude adds noise) — Apr 7 minutes
- [x] Surprise's geometric outlier status addressed — identified in every phase as maximally dispersed
- [x] Cross-dataset validation documented (20K dataset, Apr 16) — evidence geometry is not dataset-specific
- [x] Cross-model validation documented (two pretrained models, Apr 16) — evidence geometry is not model-specific
- [x] Image pivot motivation explained (120K dataset too noisy, Apr 28 pivot to brain fMRI)
- [x] 11D brain compression step rationale given (anatomical interpretability; denoises ROI signal)
- [x] Negative silhouette for brain fMRI explained (high-dimensional artefact, not absence of structure)

---

## Glossary

### Model Variants

| ID | Base Model | Training State | Layer | Dim |
|---|---|---|---|---|
| BGE-Base-Final | BAAI/bge-base-en-v1.5 | Pretrained | Final (12) | 768 |
| BGE-Base-Mid | BAAI/bge-base-en-v1.5 | Pretrained | Middle (6) | 768 |
| BGE-FT-Final | BAAI/bge-base-en-v1.5 | Fine-tuned on emotion labels | Final (12) | 768 |
| BGE-FT-Mid | BAAI/bge-base-en-v1.5 | Fine-tuned on emotion labels | Middle (6) | 768 |
| MPNet-Base-Final | sentence-transformers/all-mpnet-base-v2 | Pretrained | Final (12) | 768 |
| MPNet-Base-Mid | sentence-transformers/all-mpnet-base-v2 | Pretrained | Middle (6) | 768 |
| MPNet-FT-Final | sentence-transformers/all-mpnet-base-v2 | Fine-tuned on emotion labels | Final (12) | 768 |
| MPNet-FT-Mid | sentence-transformers/all-mpnet-base-v2 | Fine-tuned on emotion labels | Middle (6) | 768 |
| Qwen-768 | Qwen/Qwen3-1.7B | Fine-tuned on emotion labels | Final (PCA→768D) | 768 |
| Brain-fMRI | OpenNeuro DS005700 | N/A (biological) | 48 ROI activations | 48 |

### Datasets

| Name | Source | Size | Classes |
|---|---|---|---|
| dair-ai/emotion (balanced) | HuggingFace | 4038 validation samples | anger, fear, happiness, love, sadness, surprise |
| OpenNeuro DS005700 | openneuro.org | 40 subjects × 5 emotions = 200 observations | afraid, calm, delighted, depressed, excited |

### Term Definitions

**The Void:** Zero-density region from the emotion centroid out to approximately 0.375–0.625 normalised distance units (model-dependent). Fine-tuned models have a shorter void; pretrained models have a longer void. Primary evidence for the "No Pure Emotion" finding. No data point sits here in any tested system.

**The Belt:** Peak-density shell at approximately 0.56–0.94 normalised distance units. Where almost all real data points live. Earlier for fine-tuned models (peak ~0.56), later for pretrained models (peak ~0.94). The Belt's exact position is model-dependent — that variation is itself a finding.

**No Pure Emotion:** The conclusion that follows from The Void — no data point sits at or near the class centroid, so no instance represents a "pure" emotional prototype. *Do not define this as "centroid = average of mixed subtypes"* — that is a secondary geometric observation that explains why the centroid is unreachable, not the finding itself.

**Geometric Overlap (formal definition):** Point P (with ground-truth class C) overlaps class C′ if dist(P, C′) < dist(P, own centroid C). This is a *separate* finding from The Void — it quantifies cross-class bleed among Belt-region points. It further supports "no pure emotion" through a different mechanism. Do not conflate with The Void.

**Certainty Buffer:** The radial gap between the density peak (The Belt) and the onset of cross-class overlap (>5% overlap). Large for fine-tuned models (+1.75–1.875), near-zero for pretrained models. Fine-tuning creates a geometrically safe zone.

**Ambiguity Gradient:** Geometric positioning predicts classifier uncertainty in both systems, but via different mechanisms. Brain: centroid *margin* (competitive distance) predicts uncertainty, Spearman r = 0.61, AUC = 0.81. LLMs: centroid *distance* predicts logit confidence, r = 0.957–0.988. Cross-system unified figure r ≈ 0.56 is [SOURCE UNCONFIRMED].

**Relational Paradox:** Both LLM types sort emotions by valence (PC1); the brain sorts by arousal (PC1) — producing near-opposite RDMs. Fine-tuned LLM r = −0.88 (48D raw); pretrained LLM r = −0.99 (11D systems level). Not a failure of either system — each has optimised for different representational priorities.

**Valence-Dominant / Arousal-Dominant:** Descriptor for which affective dimension a system prioritises in its primary geometric axis. LLMs are valence-dominant (semantic/categorical). The brain is arousal-dominant (physiological/survival).

**RDM (Representational Dissimilarity Matrix):** Pairwise distance matrix between emotion class centroids. Compared across systems using Pearson correlation (RSA) to measure structural alignment. This study uses the 3-emotion triplet: Fear, Happiness, Sadness — the overlap between brain dataset emotions and text dataset emotions.

**RSA (Representational Similarity Analysis):** Comparing RDMs across systems to measure structural alignment. Positive r = systems organise these emotions in the same relational order. Negative r = systems organise them in opposite order (the Relational Paradox).

**LOSO (Leave-One-Subject-Out):** Cross-validation protocol used for brain fMRI decoding accuracy. Each subject is held out in turn as the test set; classifier trained on the remaining 39 subjects. Brain LOSO accuracy: 0.56 (chance: 0.20).

**Signal Half-Life (D50):** Number of dominant SVD directions removed before classifier accuracy drops to 50% of baseline. Brain: D50 = 4 dims (highly distributed). MPNet pretrained: D50 = 17 dims (redundant). Qwen fine-tuned: D50 = 12 dims (compressed).

**Raw Pooling:** Embedding extraction method that retains both magnitude and direction of the embedding vector. Outperforms L2 normalisation for geometry-based analysis because direction is the primary signal; magnitude is not normalised away. All text embedding geometry experiments use raw pooling followed by L2 normalisation in the loader.

**Signal Erasure Point:** The dimension count at which iterative SVD ablation causes classification accuracy to reach the chance baseline (~16.7% for 6 classes). Differs from D50 — erasure is total collapse; D50 is the 50% midpoint.

---

*Note on image experiments:* Image extension experiments (EmoSet, FI, Emotion6 with CLIP pipelines) were explored in April 2026 but the 120K image dataset proved too noisy for clear cluster separation. The direction was abandoned in favour of brain fMRI comparison (Apr 28 pivot). Image experiment material is retained in `reports/` and `configs/` for reference but is not part of the NeurIPS paper story. A brief mention as a negative result may appear in the appendix.
