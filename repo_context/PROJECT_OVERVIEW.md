> **COMPASS FILE — READ THIS FIRST**
> This is the authoritative orientation document for the `vidiq-hpc` repository.
> If you are an AI agent, read this file before opening any experiment directory.
> It is the single source of truth for scientific story, key numbers, open items, and file locations.
> Maintained by: Joshua Bhawanlall | Last updated: 2026-05-03 | Status: Active — NeurIPS submission in progress
>
> **What this file contains:** Project identity · Six key findings with verified numeric anchors · NeurIPS abstract (248 words) · Full repository map with current meeting file naming convention · Work diary (8 sessions) · Open items checklist · Glossary with V/A coordinates · Potential references section
>
> **Known data quality issues flagged in this file:** (1) Brain vs MPNet 48D RDM discrepancy (−0.5476 vs −0.1023) in Finding 6b — needs Pritish clarification. (2) Cross-system ambiguity gradient RESOLVED — confirmed r = 0.9565, p = 3.20e-54 from `experiments/brain_embedding_understanding/global_behavior_comparison/comparison_results.txt` (note: meeting minutes cited r ≈ 0.56 in error — confirmed value is 0.9565). (3) LOSO/bootstrap/permutation for LLM text experiments — DECISION NOT REQUIRED: cross-model replication across 4 architectures is the generalization argument; paper pushes pattern story not metrics benchmark.

---

## Pre-Write Repo Check Results

| Blocking Item | Status | Evidence / Note |
|---|---|---|
| LOSO/bootstrap/permutation tests on LLM text experiments | DECISION: NOT REQUIRED | Cross-model replication across 4 architectures (BGE-Base, BGE-FT, MPNet-Base, MPNet-FT) serves as the generalization evidence for LLM results. The paper's contribution is the geometric pattern story and consistent structure across models — not a metrics benchmark. Statistical rigour at the subject level (LOSO etc.) applies to the brain fMRI data because of subject-level variance; LLM embedding geometry is deterministic given fixed weights. Decision made 2026-05-03. |
| Ambiguity gradient source file | RESOLVED | File pushed by Pritish. Confirmed at `experiments/brain_embedding_understanding/global_behavior_comparison/comparison_results.txt`. Actual values: r = **0.9565**, p = 3.1984e-54. ⚠️ **Correction:** Meeting minutes cited r ≈ 0.56 — this was an error (0.56 is the Brain LOSO accuracy, not this correlation). The paper must cite r = 0.9565. 95% CI: [0.9370, 0.9725]. Permutation p (n=5000): 0.000. |
| Valence-arousal ground truth citation | PARTIAL | `PLAN.md` and `run_va_reduction.py` reference Russell's Circumplex and DEAP/Koelstra; May 1 meeting confirms Russell 1980 + DEAP as citations — exact numerical table not yet formally cited in a paper-facing document |
| Aimee's results PDF | NOT IN SCOPE | File will not be included in the project. Removed 2026-05-03. |
| Embedding extraction layer documented | PARTIAL | `loader_text.py` confirms L2-normalised mean-pooled embeddings from pre-saved `.npy` files; the specific transformer layer (final layer 12 vs middle layer 6) is encoded in the folder names (`bge`, `bge-mid`, etc.) but is not stated in prose in any methods document |
| fMRI sample size justification | PARTIAL | Apr 28 minutes contain Aimee's oral statement ("most published fMRI studies use 8–10 subjects; 40 is above field standard"). Precedent studies confirmed: IBC dataset (12 subjects, Nature Scientific Data), PFM Children (12 subjects, Dosenbach Lab), NSD (8 subjects, ~40 sessions/person). These are high-impact published studies — our 40 subjects exceed all three. Statement now has citable comparison points, but still needs writing into a methods section. |

---

## Project Identity

**Full project name:** LLM Embedding Geometry and Brain fMRI Alignment Study
**Short repo name:** `vidiq-hpc`
**Target venue:** NeurIPS 2026 | Deadline: 4 May 2026
**Working paper title (tentative):** *"Geometric Competition as a Shared Principle of Ambiguity in LLM and Neural Emotion Representations"*
*(Revised 2026-05-03 per Aimee: previous title "The Geometry of Affect: Universal Laws…" was flagged as an overstatement — evidence from a few LLMs and one brain dataset does not justify "universal laws" claims. Title now reflects the actual finding: a geometric competition principle for ambiguity.)*

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

LLM embeddings encode emotional context geometrically via measurable structural principles — specifically centroid distance and cluster density. These principles define a consistent representational geometry: a void near each class prototype that no observed sample occupies, a dense belt of real instances around it, and a competitive boundary where class centroids contest assignment. If the same geometric competition structure appears in human fMRI brain activations, it constitutes evidence of a shared geometric principle for representational ambiguity across artificial and biological systems.

> **Scientific scope note (Aimee, 2026-05-03):** This paper does not claim universal laws. Our evidence covers a few LLM architectures and one brain fMRI dataset. The claim is that a geometric competition principle *is observable* across these systems, not that it holds universally. Philosophical interpretations (e.g. "emotions are inherently mixed") are left to readers.

### Six Key Findings

---

**Finding 1: The Density Structure — "The Void" and "The Belt"**

All distances are normalised (scaled to the 0–2.5 range) to allow comparison across model architectures and the brain fMRI system — these are not raw Euclidean magnitudes. On this normalised scale, a consistent two-zone density structure emerges across all 8 LLM variants and 2 text datasets:

- **The Void:** No data points exist within approximately 0.375–0.625 normalised distance units of any emotion centroid. The precise onset of The Void varies by model — fine-tuned models show a shorter void (first non-zero density at approximately bin 0.4375 normalised units); pretrained models show a longer void (first non-zero density at approximately bin 0.6875 normalised units). This model-dependent variation is itself informative: fine-tuning pulls data closer to the prototype boundary.

- **The Belt:** Density rises sharply from the void edge and peaks at approximately 0.56–0.94 normalised distance units. Fine-tuned models peak earlier (MPNet-FT-Final at bin 0.5625, BGE-FT-Final at bin 0.6875); pretrained models peak later (MPNet-Base-Final and BGE-Base-Final both at bin 0.9375). All real data points live in or beyond The Belt — none at the centroid itself.

*Source: `experiments/understanding_text_embeddings/reports/phase2/overlap_metrics.json`*

---

**Finding 2: Unoccupied Centroids — No Embedding Aligns Exclusively with a Single Class Prototype**

> **Scientific framing (Aimee, 2026-05-03):** The claim "no pure emotion" is not scientifically usable. The precise finding is: *"centroids are not occupied by observed samples"* and *"no embedding aligns exclusively with a single class prototype."* The paper must present this as a geometric observation about the representational system, not a philosophical statement about emotions. The interpretation is left to readers.

We define a representation as "pure" if it is significantly closer to its assigned class centroid than all alternative centroids — i.e. it would sit within the void region near its class prototype. The Void (Finding 1) demonstrates that no observed sample meets this criterion: every embedding is located at meaningful distance from its assigned centroid, placing all observations in the Belt where competing centroids may also exert geometric influence. **No embedding aligns exclusively with a single class prototype in any tested model or the brain fMRI system.**

This is a statement about the geometry of the representational system. The centroid sits in void space partly as a mathematical consequence of being the mean of a distributed ring of points — it does not correspond to any actual sample, which explains mechanistically why the void is always unoccupied. The finding's substance, however, is the competitive distance result: no sample is ever unambiguously claimed by one class under the margin criterion.

*Graduated finding — relative representation quality:* The formal overlap metric — point P overlaps class C′ if dist(P, C′) < dist(P, own centroid) — quantifies how many samples lie closer to a competing centroid than to their own. Overlap rates by model: BGE-Base-Final 19.22%, MPNet-Base-Final 18.45%, BGE-FT-Final 1.81%, MPNet-FT-Final 0.42%. Certain emotion categories exhibit higher centroid separation and lower overlap, reducing ambiguity in their representations: **Happiness and Love** show the highest relative representation quality (greatest centroid separation, lowest cross-class overlap in fine-tuned models). The most geometrically entangled pair: Happiness vs Love at 42% overlap in pretrained models. The most separable pair: Fear vs Anger at 12% overlap. This graduated picture — "while no embedding is strictly unambiguous, Happiness and Love exhibit higher relative purity defined as greater separation from competing class centroids" — is the scientifically appropriate description.

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

**Cross-system ambiguity gradient r = 0.9565, p = 3.1984e-54:** [SOURCE CONFIRMED — `experiments/brain_embedding_understanding/global_behavior_comparison/comparison_results.txt`]. Pearson r = 0.9565, p = 3.1984e-54. Enhanced validation: R² = 0.9148, 95% CI [0.9370, 0.9725], permutation p (n=5000) = 0.0000. ⚠️ **Note on prior discrepancy:** Apr 28 meeting minutes cited r ≈ 0.56, which was an error — 0.56 is the Brain LOSO decoding accuracy, not the cross-system correlation. The correct and confirmed cross-system Pearson r is **0.9565**. Use this value in the paper. Additional finding from the same folder: density distribution similarity KS statistic = 0.57 (p = 2.46e-15), Cohen's d = 1.34 — systems differ in point packing density despite shared ambiguity gradient trend.

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

| Comparison | Pearson r | Source | Interpretation |
|---|---|---|---|
| Brain vs fine-tuned LLM (48D, cosine RDM) | **−0.8756** | `relational_validation.json` | Near-opposite centroid geometry |
| Brain vs fine-tuned LLM (48D, Manhattan sensitivity) | −0.5476 | `relational_validation.json` | Metric-invariant confirmation |
| Brain vs pretrained LLM (48D, cosine RDM) | −0.1023 | `rdm_comparison_metrics.json` | Weak negative alignment |
| Brain vs pretrained LLM (48D, as per `CENTROID_RELATIONAL_FINAL_REPORT.md` table) | **−0.5476** ⚠️ | `checking_centroids/reports/` | **DISCREPANCY NOTE:** The CENTROID_RELATIONAL_FINAL_REPORT.md table lists −0.5476 under "Brain vs MPNet" but this value is identical to the Manhattan sensitivity correlation for brain vs fine-tuned LLM (Qwen). The `rdm_comparison_metrics.json` file shows brain_vs_mpnet cosine r = −0.1023. This discrepancy needs clarification from Pritish before citing in the paper. |
| Brain-brain (noise ceiling) | upper 0.484 / lower 0.460 | `relational_validation.json` | Human-human agreement ~0.47 |

The fine-tuned LLM correlation with the brain (−0.88) is not just near-zero — it is actively opposed. The noise ceiling shows human-human agreement is only ~0.47, so the −0.88 divergence is striking. Bootstrap 95% CI: [−0.9967, +0.6994] — wide due to small triplet (3 emotions), but distribution is skewed negative.

**⚠️ Action required:** Clarify whether the CENTROID_RELATIONAL_FINAL_REPORT.md Brain vs MPNet = −0.5476 was computed using Manhattan distance (matching the metric sensitivity test value) or cosine distance. The `rdm_comparison_metrics.json` cosine value (−0.1023) and the report value (−0.5476) cannot both be correct cosine correlations for the same data.

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
| Brain vs fine-tuned LLM — 48D cosine Pearson r | −0.8756 | Brain vs Fine-tuned LLM | `relational_validation.json` |
| Brain vs fine-tuned LLM — 48D Manhattan r (sensitivity) | −0.5476 | Brain vs Fine-tuned LLM | `relational_validation.json` |
| Brain vs pretrained LLM — 48D cosine Pearson r | −0.1023 | Brain vs Pretrained LLM | `rdm_comparison_metrics.json` |
| Brain vs pretrained LLM — 48D (per CENTROID_RELATIONAL_FINAL_REPORT.md) | −0.5476 ⚠️ DISCREPANCY — see Finding 6b note | Brain vs Pretrained LLM | `checking_centroids/reports/` |
| Brain-brain noise ceiling | upper 0.484 / lower 0.460 | Brain fMRI | `relational_validation.json` |
| Brain vs pretrained LLM — 11D Pearson r | **−0.9918** | Brain vs Pretrained LLM | `rdm_results_11d.json` |
| Brain vs fine-tuned LLM — 11D Pearson r | −0.7539 | Brain vs Fine-tuned LLM | `rdm_results_11d.json` |
| Cross-system ambiguity gradient r | **0.9565** (p = 3.1984e-54; 95% CI [0.9370, 0.9725]; permutation p = 0.000) | Cross-system | `experiments/brain_embedding_understanding/global_behavior_comparison/comparison_results.txt` ✓ CONFIRMED |
| Cross-system density KS statistic | 0.5700 (p = 2.46e-15; Cohen's d = 1.34) | Cross-system | `experiments/brain_embedding_understanding/global_behavior_comparison/enhanced_statistical_results.txt` ✓ CONFIRMED |

---

## NeurIPS Abstract (Draft)

> We investigate whether emotional representations in large language models (LLMs) and the human brain share a common geometric logic. Across eight LLM variants (BGE and MPNet, pretrained and fine-tuned) applied to a 6-class emotion text dataset, and compared to fMRI from 40 subjects (5 emotions, 48 ROIs; OpenNeuro DS005700), we identify six normalised-distance findings. First, no data point exists within 0.375–0.625 normalised units of any emotion centroid (The Void), and density peaks at 0.56–0.94 normalised units (The Belt). Second, centroids are not occupied by observed samples: no embedding aligns exclusively with a single class prototype, placing all observations in the Belt where competing centroids may claim them. Third, fine-tuned LLMs compress signal into a sharp cliff (erased at dimensions 15–20); pretrained LLMs distribute it across 26–67 dimensions. Fourth, projecting into the top 20 discriminative dimensions resolves the apparent disorder of pretrained 768D embeddings: silhouette scores jump 627–765%, confirming the "cloud" appearance is high-dimensional noise. Fifth, in brain fMRI centroid margin predicts cross-validated uncertainty (Spearman r = 0.61, p = 7.78×10⁻²², AUC = 0.81; raw distance: AUC = 0.56); in LLMs, centroid distance predicts logit confidence (r = 0.957–0.988). Sixth, despite this shared local principle, global organisation is inverted: LLMs sort emotions by valence (PC1 r = 0.97–0.98), the brain by arousal (PC1 r = 0.96), yielding near-opposite dissimilarity matrices (brain vs fine-tuned LLM r = −0.88 in 48D; brain vs pretrained LLM r = −0.99 in 11D). Shared local geometry coexists with fundamentally different global representational priorities.

**Word count:** 247 words (verified).

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

2. **`adding_spatial_context/`** — Compresses 48D ROI space to 11D (5 anatomical lobes + 5 functional networks + 1 neighbour context). Rationale: domain-informed feature construction — not generic PCA — where each dimension corresponds to a biologically meaningful unit. The 5 functional network dimensions correspond to established large-scale brain systems: Default Mode Network, Salience Network, Central Executive Network, and related functional systems. The neighbour-context feature captures local spatial interaction between ROIs (inter-regional coordination). Aggregating within anatomically/functionally coherent groups acts as structured denoising, suppressing high-frequency noise while preserving distributed emotional representation patterns.
   - Output: `outputs/BRAIN_11D_SYSTEMS_REPORT.md`
   - Justification document: `repo_context/project_context/Full Scientific Justification (Tailored to Your Method).docx`

3. **`checking_centroids_with_spatial_context_data/`** — Repeats RDM analysis on the 11D brain representation. The paradox amplifies — pretrained LLM r = −0.9918.
   - Output: `reports/SYSTEMS_LEVEL_PARADOX_REPORT.md`, `reports/rdm_results_11d.json`

4. **`valence-arousal-dimensional_reduction/`** — PCA alignment of LLM and brain centroids with valence/arousal coordinates. Explains the RDM inversion mechanistically.
   - Output: `reports/VA_GEOMETRIC_CONVERGENCE_REPORT.md`, `reports/alignment_metrics.json`

5. **`checking_density_geometry/`** — Cross-system density decay comparison; brain ambiguity gradient validation (margin vs uncertainty).
   - Output: `reports/DENSITY_GEOMETRY_FINAL_REPORT.md`, `reports/density_validation.json`, `reports/brain finl.html`

6. **`checking_context_retention_across_dimensions/`** — Iterative SVD ablation comparing Brain, MPNet pretrained, and Qwen fine-tuned. Extends Finding 3 cross-system.
   - Output: `reports/CONTEXT_RETENTION_FINAL_REPORT.md`

7. **`global_behavior_comparison/`** — Cross-system ambiguity gradient comparison (Finding 5 unified figure). Computes Pearson correlation between brain and LLM centroid-margin/distance signals across emotion categories.
   - Output: `comparison_results.txt` (r = 0.9565, p = 3.20e-54), `enhanced_statistical_results.txt` (CI, permutation test, Cohen's d), `Density_Decay_Comparison_Report.pdf`, `global_behavior_comparison.png`

---

### Scripts

- `scripts/prepare_balanced6_dataset.py` — Combines per-emotion raw embeddings into unified training dataset
- `scripts/embed_balanced_6_emotions_raw.py` — Generates raw mean-pooled embeddings from CSV via HuggingFace model

---

### Meeting Minutes

All files in `meetings/`.

| Date | Minutes File | Full Transcript | Key Topic |
|---|---|---|---|
| 2026-04-07 | `2026-04-07_transcript_minutes.md` | `2026-04-07_transcript.md` | Core hypothesis established; raw pooling > L2; NeurIPS deadline set; Surprise identified as geometric outlier; 6% truncation risk flagged |
| 2026-04-11 | `2026-04-11_transcript_minutes.md` | `2026-04-11_transcript.md` | Density belt discovery; "no pure emotion" articulated as initial team phrasing (later refined by Aimee 2026-05-03 → "no embedding aligns exclusively with a single class prototype" — see Finding 2 and Centroid Exclusion glossary entry); Qwen 1.7B agreed; logits > softmax principle; cross-model validation plan |
| 2026-04-12 | `2026-04-12_1_2_3_transcript_minutes.md` | `2026-04-12_1_transcript.md`, `2026-04-12_2_transcript.md`, `2026-04-12_3_transcript.md` | The Void formalised (nothing from centroid to 7.5 raw units); y-axis label correction (number of points per band, not density/volume); overlap definition clarified; HPC text-model training setup recorded |
| 2026-04-16 | `2026-04-16_transcript_minutes.md` | `2026-04-16_transcript.md` | Cross-dataset + cross-model validation confirms geometry is universal; fine-tuning hypothesis; image dataset selection assigned |
| 2026-04-21 | `2026-04-21_transcript_minutes.md` | `2026-04-21_transcript.md` | Image extension designed; video segmentation approach; HPC four-account setup; project walkthrough for newer members |
| 2026-04-28 | `2026-04-28_transcript_minutes.md` | `2026-04-28_transcript.md` | Pivot from 120K image set to brain fMRI; Aimee joins; OpenNeuro DS005700 confirmed; LOSO/bootstrap/permutation requirements set; cross-system comparison file path given (not in repo) |
| 2026-05-01 (session 1) | `2026-05-01_1_transcript_minutes.md` | `2026-05-01_1_transcript.md` | All 5 text phases reviewed; Phase 5 logit-geometry results; brain data walkthrough (NeuroEmo dataset); cross-system geometry direction agreed |
| 2026-05-01 (session 2) | `2026-05-01_2_transcript_minutes.md` | `2026-05-01_2_transcript.md` | Brain context retention + V/A reduction reviewed; Russell 1980 + DEAP citations confirmed; valence/arousal framing as operational labels agreed |

---

### repo_context/

- `repo_context/PROJECT_OVERVIEW.md` — this file (the compass)
- `repo_context/neurips_context/neurips-guide.md` — NeurIPS submission guide and strategic framing
- `repo_context/neurips_context/neurips_deep_submission_structure_checklist_report.pdf` — detailed NeurIPS structure + checklist (11 pages; section-by-section build plan)
- `repo_context/neurips_context/neurips_pre_submission_checklist_full.pdf` — pre-submission review checklist (6 pages; claim support, statistical validity, rejection patterns)
- `repo_context/project_context/Emotion_Geometry_Complete_Report.pdf` — 3-page complete emotion geometry report (generated 2026-05-02; authoritative consolidated findings)
- `repo_context/project_context/Full Scientific Justification (Tailored to Your Method).docx` — scientific justification for the 48D→11D brain representation transformation; includes network citations (Default Mode, Salience, Central Executive) and reviewer-facing framing for the domain-informed feature construction
- `repo_context/project_context/Justification brain data.docx` — fMRI sample size justification with precedent studies: IBC (N=12), PFM (N=12), NSD (N=8) all published in high-impact venues; confirms our N=40 is above field standard
- `repo_context/project_context/PLAN_COMPASS_GENERATION.md` — agent plan used to generate this compass (not the compass itself)

---

## Work Diary

### 2026-04-07 — Core Hypothesis and Experimental Foundation
**Present:** Daniel Sikar, Pritish Ranjan
**Key outcomes:**
- Core hypothesis established: centroid distance + cluster density describe a universal geometric law in LLM embedding space
- Raw pooling outperforms L2 normalisation for geometry-based analysis; direction is the primary signal; magnitude adds noise
- NeurIPS 4 May 2026 deadline set as hard target
- Surprise identified as the geometrically ambiguous class — a reaction rather than a pure emotion
- Multi-class pipeline and 6-class emotion dataset confirmed; binary SST-2 also tested
- Model selection on 10% training data agreed (justified as lighter task than embedding validation); same model carries forward
- 6% truncation rate in multi-class flagged as a risk to address
**Source:** `2026-04-07_transcript_minutes.md`

### 2026-04-11 — The Density Belt Discovery
**Present:** Daniel Sikar, Pritish Ranjan
**Key outcomes:**
- Density belt confirmed: density does not peak at centroid — it peaks at ~9–9.5 raw Euclidean units (later normalised to ~0.9375 for pretrained models)
- "No pure emotion" formally articulated as initial team shorthand: no data point at the centroid; centroid = mean of mixed subtypes. **Note:** This phrasing was later refined (2026-05-03, Aimee Bottrill-Frost) to the scientifically defensible geometric claim: *"no embedding aligns exclusively with a single class prototype."* See Finding 2 and the Centroid Exclusion glossary entry. The paper must not use "no pure emotion" as a standalone claim.
- Logits > softmax principle stated: logits preserve absolute magnitude for geometry analysis; softmax collapses it into relative distribution. All fine-tuned classifier experiments record logits, not softmax probabilities.
- Qwen 1.7B architecture agreed for fine-tuned classifier
- Cross-model validation planned for April 18
- Bengio (2017) label-randomisation idea discussed as future/separate paper, not in NeurIPS scope
**Source:** `2026-04-11_transcript_minutes.md`

### 2026-04-12 — The Void Formalised, Overlap Clarified, and HPC Training Setup
**Present:** Daniel Sikar, Pritish Ranjan
**Key outcomes:**
- The Void formally named: zero-density region from centroid to ~7.5 raw Euclidean units (later normalised: ~0.625 for pretrained models). "There's a void in every emotion" — Pritish
- Y-axis label corrected: "number of data points per band" (NOT "density" or "density per volume"). The volume of radial shells increases with distance in 768D; raw point counts rise then fall; but per-unit-volume density strictly decreases from the centroid outward. Only the number-of-points-per-band plot should be in the paper; the volume-normalised curve is technically correct but shows the opposite trend visually.
- Overlap definition formally clarified: point P overlaps class C′ iff dist(P, C′) < dist(P, own centroid). Not a radius rule — purely a competitive distance comparison.
- HPC planning captured: Qwen 3 1.7B selected as the initial text-model starting point; HPC environment bring-up, repo clone, training run, and post-training geometry rerun agreed as the next execution path.
- Three sessions were later consolidated into `2026-04-12_1_2_3_transcript_minutes.md`, based on `2026-04-12_1_transcript.md`, `2026-04-12_2_transcript.md`, and `2026-04-12_3_transcript.md`.
**Source:** `2026-04-12_1_2_3_transcript_minutes.md`

### 2026-04-16 — Cross-Model and Cross-Dataset Validation
**Present:** Daniel Sikar, Pritish Ranjan, Josh, Andrew
**Key outcomes:**
- Cross-dataset validation (new 20K dataset, ~4–5× larger): same geometric patterns hold; minimum/maximum distances from centroid remain comparable; clusters preserve relational structure
- Cross-model validation (different pretrained model): belt pattern and overlap region confirmed model-agnostic; absolute magnitudes differ (peak at ~9 units vs ~4–5 units) but functional pattern unchanged
- Vector arithmetic analogy: emotional embeddings can be decomposed/composed geometrically (Italy − Rome + France ≈ Paris)
- Fine-tuning hypothesis stated: fine-tuning expected to tighten clusters
- Image dataset selection task assigned to Josh/Daniel
**Source:** `2026-04-16_transcript_minutes.md`

### 2026-04-21 — Image Extension and HPC Coordination
**Present:** Daniel Sikar, Pritish Ranjan, Josh, Andrew
**Key outcomes:**
- Image extension designed: same pipeline applied to image embeddings (same embeddings, different labelling schemes — test whether one embedding space simultaneously supports clustering by multiple attributes)
- Video segmentation approach: coarser segmentation (fixed timestamps / short intervals) preferred over frame-by-frame
- HPC four-account setup confirmed; all team members working toward access
- Embedding extraction point per model must be explicitly stated in methods section (action assigned Apr 21 — still open)
- Qwen HPC run completed during meeting; initial push missing .npy embedding files
**Source:** `2026-04-21_transcript_minutes.md`

### 2026-04-28 — Brain fMRI Pivot and Statistical Rigour
**Present:** Daniel Sikar, Pritish Ranjan, Aimee Bottrill-Frost, Josh, Freya Myo, Andrew, Arj
**Key outcomes:**
- Pivot from 120K image dataset (too noisy for clear cluster separation) to brain fMRI; Aimee Bottrill-Frost joins as neuroscience expert
- OpenNeuro DS005700 confirmed: 40 subjects × 5 emotions × 48 ROIs. Dataset full name: "Neural MO — fMRI Dataset for Emotion Recognition"
- Brain LOSO accuracy 0.56 (95% CI: 0.49–0.63, p < 0.001); negative silhouette explained as high-dimensional artefact, not absence of structure (supported by Haxby et al.)
- Cross-system ambiguity gradient figure r ≈ 0.56, p ≈ 10⁻⁵⁴ cited in meeting [SOURCE FILE PATH KNOWN — `video_understanding/human_brain_emotion_exports/global_behavior_comparison/comparison_results.txt` — but this file is on Pritish's MacBook and has NOT been pushed to the repo]
- Statistical rigour requirements set by Aimee: all LLM text experiments must receive LOSO, bootstrap CI, and permutation tests before submission. "Otherwise reviewers will throw it out."
- fMRI sample size justified: 40 subjects >> 8–10 typical in literature (Aimee's statement, not yet formally cited)
- Loss of spatial brain adjacency in flat 48D ROI vector noted as a limitation / future work
- Aimee's results PDF (`emotion_geometry_and_score_4_square_explorer_square_report_score_v2.pdf`) created 27 April; not yet in repo
**Source:** `2026-04-28_transcript_minutes.md`

### 2026-05-01 (Session 1) — Phase Reviews and Brain Data Walkthrough
**Present:** Daniel Sikar, Pritish Ranjan, Amy (Aimee)
**Key outcomes:**
- All 5 text embedding phases reviewed; brain data walkthrough on NeuroEmo dataset (same dataset as DS005700, alternate name used in May 1 meeting)
- Phase 5 logit-geometry results confirmed: BGE-FT 76 overlaps (1.88%), MPNet-FT 71 overlaps (1.76%); logit-agreement rate very high; approximately linear relation between geometric margin and logit margin
- Certainty buffer description corrected: it is the gap between density peak and overlap peak, not "start of ambiguity"
- Overlap heatmap colour scales need to be comparable across models before use as paper figures
- Phase 3–4 terminology needs simplifying; SVD and "20 directions" (not raw embedding indices) need clear definitions
- Brain data: 40 subjects × 5 emotions × 200 total samples × 48 cortical ROI dimensions (Harvard-Oxford atlas); repeated blocks per subject/emotion averaged to yield stable ROI vectors
**Source:** `2026-05-01_1_transcript_minutes.md`

### 2026-05-01 (Session 2) — Context Retention, V/A Framing, Paper Storyline
**Present:** Daniel Sikar, Pritish Ranjan, Amy (Aimee)
**Key outcomes:**
- Brain context retention: fine-tuned LLM starts highest/decays fastest (compressed); pretrained distributes gradually; brain maintains steady plateau with the most distributed encoding
- V/A correlation confirmed: fine-tuned LLM PC1 valence r ≈ 0.98; brain PC1 arousal dominant; brain PC2 valence weaker
- V/A ground truth table framed as operational/cited labels, NOT measured physical constants. Sources: Russell (1980) circumplex + DEAP/Koelstra. Exact numerical values per emotion confirmed (see Glossary).
- Paper framing agreed: from "interesting one-off" → "candidate universal property of intelligence-related representation"
- Missing HTML/output summary for V/A phase noted as needing consolidation
**Source:** `2026-05-01_2_transcript_minutes.md`

---

## Open Items: NeurIPS Submission (Critical Path)

### Must-Do (Blocking Submission)

- [x] **LOSO + bootstrap CI + permutation tests on LLM text embedding results** — DECISION: NOT REQUIRED. Cross-model replication across 4 architectures is the generalization argument. The paper pushes the geometric pattern story; LOSO-style cross-validation applies to biological subject-level data (brain fMRI) where individual differences are the source of variance. LLM embedding geometry is deterministic. Reviewer challenge ("what is the significance of your silhouette improvement?") is answered by pointing to consistent patterns across all 4 model variants. Decision: 2026-05-03.
- [x] **Push cross-system comparison file to repo** — DONE. File is at `experiments/brain_embedding_understanding/global_behavior_comparison/comparison_results.txt`. Confirmed r = 0.9565 (not 0.56 as cited in meeting — see correction note in Finding 5). Also includes enhanced_statistical_results.txt with CI/permutation test. Owner: Pritish ✓
- [ ] **Clarify Brain vs MPNet 48D discrepancy** — CENTROID_RELATIONAL_FINAL_REPORT.md Table 3.2 shows Brain vs MPNet Pearson r = −0.5476, but `rdm_comparison_metrics.json` shows brain_vs_mpnet cosine = −0.1023. These cannot both be correct cosine correlations. Determine whether the −0.5476 in the report was computed with Manhattan distance (matching the relational_validation.json manhattan_correlation value). Correct the report if wrong. Owner: Pritish
- [ ] **Confirm valence-arousal ground truth citation chain** — Russell 1980 circumplex + DEAP (Koelstra & Mühl) are confirmed as primary sources. Exact numerical V/A values for the 5/6 emotions need to be formally cited in a paper-facing document with page/table reference. Owner: Aimee
- [x] **Aimee's results PDF** — `emotion_geometry_and_score_4_square_explorer_square_report_score_v2.pdf` will not be included in the project. Removed from scope 2026-05-03.
- [ ] **Decide Phase 5 (logit consistency) scope: main paper vs. appendix** — r = 0.957–0.988 is a strong result. Daniel to confirm placement. Owner: Daniel
- [ ] **Justify fMRI sample size in methods** — add sentence citing typical fMRI studies use 8–10 subjects; 40 is above field standard. Specific precedents now available: IBC (12 subjects), PFM Children (12 subjects), NSD (8 subjects with ~40 sessions/person for dense sampling). Strategy: cite these high-impact studies to show 40 is well above field standard. Aimee to draft. Owner: Aimee/Daniel
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

**The Void:** Zero-density region from the emotion centroid out to approximately 0.375–0.625 normalised distance units (model-dependent). Fine-tuned models have a shorter void; pretrained models have a longer void. Primary evidence for centroid exclusion — no observed sample aligns exclusively with a single class prototype. No data point sits here in any tested system.

**The Belt:** Peak-density shell at approximately 0.56–0.94 normalised distance units. Where almost all real data points live. Earlier for fine-tuned models (peak ~0.56), later for pretrained models (peak ~0.94). The Belt's exact position is model-dependent — that variation is itself a finding.

**Centroid Exclusion (Unoccupied Centroids):** We define a representation as *pure* if it is significantly closer to its assigned class centroid than all alternative centroids — i.e., it lies in void space near its own prototype with no competing class nearby. No observed sample meets this criterion in any tested system (see Finding 2). The scientifically appropriate claim is: *"no embedding aligns exclusively with a single class prototype."* All observations reside in the Belt where competing centroids may claim them. Do not use "no pure emotion" as a standalone paper claim — this phrase makes a philosophical statement about emotions rather than a geometric observation about representations.

**Geometric Overlap (formal definition):** Point P (with ground-truth class C) overlaps class C′ if dist(P, C′) < dist(P, own centroid C). This is a *separate* finding from The Void — it quantifies cross-class bleed among Belt-region points. It further supports the centroid exclusion finding through a different mechanism: even within the Belt, competing centroids geometrically claim many points. Do not conflate with The Void.

**Certainty Buffer:** The radial gap between the density peak (The Belt) and the onset of cross-class overlap (>5% overlap). Large for fine-tuned models (+1.75–1.875), near-zero for pretrained models. Fine-tuning creates a geometrically safe zone.

**Ambiguity Gradient:** Geometric positioning predicts classifier uncertainty in both systems, but via different mechanisms. Brain: centroid *margin* (competitive distance) predicts uncertainty, Spearman r = 0.61, AUC = 0.81. LLMs: centroid *distance* predicts logit confidence, r = 0.957–0.988. Cross-system unified Pearson r = **0.9565**, p = 3.1984e-54 (95% CI [0.9370, 0.9725]; permutation p = 0.000) — source confirmed at `experiments/brain_embedding_understanding/global_behavior_comparison/comparison_results.txt`. ⚠️ Meeting minutes cited r ≈ 0.56 in error; that value is the Brain LOSO decoding accuracy.

**Relational Paradox:** Both LLM types sort emotions by valence (PC1); the brain sorts by arousal (PC1) — producing near-opposite RDMs. Fine-tuned LLM r = −0.88 (48D raw); pretrained LLM r = −0.99 (11D systems level). Not a failure of either system — each has optimised for different representational priorities.

**Valence-Dominant / Arousal-Dominant:** Descriptor for which affective dimension a system prioritises in its primary geometric axis. LLMs are valence-dominant (semantic/categorical). The brain is arousal-dominant (physiological/survival).

**RDM (Representational Dissimilarity Matrix):** Pairwise distance matrix between emotion class centroids. Compared across systems using Pearson correlation (RSA) to measure structural alignment. This study uses the 3-emotion triplet: Fear, Happiness, Sadness — the overlap between brain dataset emotions and text dataset emotions.

**RSA (Representational Similarity Analysis):** Comparing RDMs across systems to measure structural alignment. Positive r = systems organise these emotions in the same relational order. Negative r = systems organise them in opposite order (the Relational Paradox).

**LOSO (Leave-One-Subject-Out):** Cross-validation protocol used for brain fMRI decoding accuracy. Each subject is held out in turn as the test set; classifier trained on the remaining 39 subjects. Brain LOSO accuracy: 0.56 (chance: 0.20).

**Signal Half-Life (D50):** Number of dominant SVD directions removed before classifier accuracy drops to 50% of baseline. Brain: D50 = 4 dims (highly distributed). MPNet pretrained: D50 = 17 dims (redundant). Qwen fine-tuned: D50 = 12 dims (compressed).

**Raw Pooling:** Embedding extraction method that retains both magnitude and direction of the embedding vector. Outperforms L2 normalisation for geometry-based analysis because direction is the primary signal; magnitude is not normalised away. All text embedding geometry experiments use raw pooling followed by L2 normalisation in the loader.

**Signal Erasure Point:** The dimension count at which iterative SVD ablation causes classification accuracy to reach the chance baseline (~16.7% for 6 classes). Differs from D50 — erasure is total collapse; D50 is the 50% midpoint.

**Procrustes Disparity:** A metric (0 = perfect alignment, 1 = no alignment) measuring how well a learned 2D map fits the V/A "ideal map" after optimal rotation, scaling, and translation. LLMs: disparity ~0.18 (very close fit). Brain: disparity ~0.80 (complex structure — arousal axis is present but overall 2D shape does not fit the simplified circumplex). Source: `valence-arousal-dimensional_reduction/reports/va_alignment_report.html`.

**Logits (vs. Softmax):** The raw pre-softmax layer outputs from the fine-tuned classifier. All fine-tuned experiments record logits, not softmax probabilities. Logits preserve absolute magnitude; softmax collapses to a relative distribution, preventing inter-class magnitude comparisons. Principle stated Apr 11: "inter-class statements such as 'anger scores higher in absolute terms than love' can only be made from logits."

---

### Valence-Arousal Ground Truth Coordinates

The V/A values used to test axis alignment (source: `run_va_reduction.py`, derived from Russell's Circumplex Model with expert consultation). These are operational/cited labels, not measured physical constants.

| Emotion | Valence | Arousal | Used in Experiments |
|---|---|---|---|
| Joy / Happiness / Delighted | +0.85 | 0.70 | Text + Brain |
| Love | +0.85 | 0.55 | Text only |
| Surprise | +0.10 | 0.85 | Text only |
| Calm | +0.70 | 0.15 | Brain only |
| Excited | +0.75 | 0.90 | Brain only |
| Sadness / Depressed | −0.85 | 0.25 | Text + Brain |
| Anger | −0.75 | 0.80 | Text only |
| Fear / Afraid | −0.80 | 0.85 | Text + Brain |

**Note on VA alignment metric discrepancy (MPNet):** The VA_GEOMETRIC_CONVERGENCE_REPORT.md table shows MPNet PC1 Valence r = 0.96, while `alignment_metrics.json` shows valence_r = 0.9655 (rounds to 0.97). The JSON value (0.97) is the precise computed value; the report rounded down to 0.96. Use 0.97 when citing this figure precisely.

---

*Note on image experiments:* Image extension experiments (EmoSet, FI, Emotion6 with CLIP pipelines) were explored in April 2026 but the 120K image dataset proved too noisy for clear cluster separation. The direction was abandoned in favour of brain fMRI comparison (Apr 28 pivot). Image experiment material is retained in `reports/` and `configs/` for reference but is not part of the NeurIPS paper story. A brief mention as a negative result may appear in the appendix.

---

## Potential References

This section lists references that have been mentioned in meeting minutes, cited in experiment code/reports, or are required to support claims in the paper. Entries marked **[CONFIRMED]** appear explicitly in meeting transcripts or experiment source files. Entries marked **[REQUIRED — confirm citation]** are needed for the paper but the exact bibliographic details have not yet been formally recorded in the repo.

> **Important:** Do NOT hallucinate reference details. Only the entries below are to be cited. If a reference is needed for a claim and is not listed here, flag it as [NEEDS CITATION] in the paper draft. Aimee Bottrill-Frost is compiling the full neuroscience reference list into a `LLM_brain.RTF` bundle — check that file when available.

### Affective Science / Psychological Grounding

**[CONFIRMED]** Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161–1178.
— Foundational source for the valence-arousal coordinate system. Justifies the two-axis (valence, arousal) decomposition of emotional states used throughout the paper.

**[CONFIRMED]** Koelstra, S., Mühl, C., Soleymani, M., Lee, J.-S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., & Patras, I. (2012). DEAP: A database for emotion analysis using physiological signals. *IEEE Transactions on Affective Computing, 3*(1), 18–31.
— Second V/A citation source. DEAP provides operational numerical V/A scores for emotion categories used in affective computing benchmarks. Confirmed as primary source in May 1 meeting; reportedly ~5245 citations.

### Neuroscience / Brain Distributed Coding

**[CONFIRMED — from justification doc]** Li, W., Mai, X., & Liu, C. (2014). The default mode network and social understanding of others: what do brain connectivity studies tell us. *Frontiers in Human Neuroscience, 8*. doi:10.3389/fnhum.2014.00074.
— Supports the Default Mode Network as one of the 5 functional network dimensions in the 11D brain representation. Justifies grouping ROIs into functional systems for the 48D→11D transformation.

**[CONFIRMED — from justification doc]** Seeley, W. W. (2019). The Salience Network: A Neural System for Perceiving and Responding to Homeostatic Demands. *Journal of Neuroscience, 39*(50), 9878–9882. Available at: https://www.jneurosci.org/content/39/50/9878.
— Supports the Salience Network dimension in the 11D representation. The Salience Network is one of the 5 established large-scale functional systems used in the brain dimensionality reduction.

**[CONFIRMED — from justification doc]** Seung Schik, Y. Central Executive Network. ScienceDirect Topics. Available at: https://www.sciencedirect.com/topics/psychology/central-executive-network.
— Supports the Central Executive Network dimension in the 11D representation. Together with Li et al. (2014) and Seeley (2019), these three references anchor the functional network grouping strategy to published neurobiological principles.

**[REQUIRED — confirm citation]** Haxby, J. V. et al. — Referenced in Apr 28 meeting minutes as support for the claim that "emotional states are encoded as distributed patterns rather than in single regions" and that negative silhouette in brain fMRI is not a failure. Aimee Bottrill-Frost has been tasked with providing the exact citation. Most likely reference: Haxby, J.V., Gobbini, M.I., Furey, M.L., Ishai, A., Schouten, J.L., & Pietrini, P. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science, 293*(5539), 2425–2430.
— *Use with caution — exact paper not confirmed in transcript; verify with Aimee before submitting.*

### Datasets

**[CONFIRMED]** Saravia, E., Liu, H. C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). CARER: Contextualized Affect Representations for Emotion Recognition. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing* (pp. 3687–3697). Association for Computational Linguistics.
— `dair-ai/emotion` dataset on HuggingFace (`dair-ai/emotion`). The 6-class balanced text emotion dataset used in all text embedding phases. *[Note: verify exact paper — the HuggingFace dataset card links to this paper, but confirm it is the correct citation before submitting.]*

**[CONFIRMED]** OpenNeuro DS005700 — "Neural MO — fMRI Dataset for Emotion Recognition." Available at: https://openneuro.org/datasets/ds005700. Last updated approximately mid-2025. 40 subjects × 5 emotions × 48 ROI features. Download: `openneuro-py download dataset DS005700 target_directory` (from Apr 28 meeting minutes §7.2 / analysis notebook).

**[CONFIRMED — fMRI N=40 precedent support, from justification doc]** The following high-impact fMRI studies use N < 40, confirming our 40-subject dataset is above field standard:
- Individual Brain Charting (IBC): 12 subjects — a comprehensive functional atlas across dozens of cognitive tasks. Source: Nature Scientific Data / PMC.
- Precision Functional Mapping (PFM) of Children: 12 children — dense sampling (1.5–6 hours/child); justifies small N via within-subject reliability. Source: Dosenbach Lab / PMC.
- Natural Scenes Dataset (NSD): 8 participants — justified by ~40 sessions/person (thousands of trials per brain). Source: NSD Website.
*Use these as comparison points when writing the sample size justification sentence in the methods section (see Open Items).*
Source document: `repo_context/project_context/Justification brain data.docx`

### Pre-trained Models Used

**[CONFIRMED — from codebase]** Xiao, S., et al. (2023). C-Pack: Packaged resources to advance general Chinese embedding. *arXiv:2309.07597*.
— BAAI/bge-base-en-v1.5 model (used for BGE embedding variants). See model card at huggingface.co/BAAI/bge-base-en-v1.5.

**[CONFIRMED — from codebase]** Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020). MPNet: Masked and permuted pre-training for language understanding. *NeurIPS 2020*.
— sentence-transformers/all-mpnet-base-v2 (used for MPNet embedding variants).

**[CONFIRMED — from codebase]** Qwen Team, Alibaba Cloud. (2025). Qwen3 Technical Report. *arXiv:2505.09388*.
— Qwen/Qwen3-1.7B model used for fine-tuned Qwen-768 variant. *[Note: verify exact Qwen3-1.7B citation — the model series is Qwen3 but confirm the specific technical report.]*

### Deep Learning / Transformer Architecture

**[CONFIRMED — mentioned in Apr 11 minutes]** Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*.
— Foundational transformer architecture paper. Mentioned in Apr 11 meeting as context for BLEU metric and evaluation methodology.

### Future Work / Out of Scope for This Paper

**[MENTIONED — future paper, not NeurIPS scope]** Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *ICLR 2017*.
— The Bengio (2017) label-randomisation idea discussed in Apr 11 meeting. Agreed as future/separate paper. Do not cite in this NeurIPS submission.

### To Be Added (Pending Aimee's Reference Bundle)

The following categories need citations from Aimee's `LLM_brain.RTF` reference list:
- Representational Similarity Analysis (RSA) methodology — likely: Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). *Frontiers in Systems Neuroscience* [confirm]
- Brain-AI alignment prior work — confirm list with Aimee
- Prototype theory for emotion categories — confirm with Aimee
- LOSO cross-validation methodology for fMRI — confirm with Aimee
