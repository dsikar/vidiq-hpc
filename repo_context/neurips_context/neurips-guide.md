# NeurIPS Submission Guide

This folder contains submission planning materials for the NeurIPS 2026 paper. Three files are present:

| File | Contents |
|---|---|
| `neurips_deep_submission_structure_checklist_report.pdf` | 11-page deep strategy guide: exact narrative arc, section-by-section build plan, figure strategy, reviewer attack surface, checklist answer plan, and recommended page budget. The central thesis framing in this PDF should be read before writing any section of the paper. |
| `neurips_pre_submission_checklist_full.pdf` | 6-page pre-submission review guide: paper structure, statistics checklist, methods requirements, ethics and human data wording, compute reporting template, and final readiness scoring. Treat the statistics checklist section as a blocking checklist before upload. |
| `neurips-guide.md` | This file — distilled guide to NeurIPS requirements and how this study's findings map to a competitive NeurIPS paper. |

**Deadline:** 4 May 2026

---

## NeurIPS Paper Requirements

### Format

- **Page limit:** 9 pages of content (including figures and tables). References do NOT count toward the page limit. Camera-ready (accepted) papers are allowed 10 content pages.
- **Appendix:** Allowed, unlimited length, but reviewers are not obligated to read it. All results essential for the main claim must be in the 9-page body.
- **Double-blind:** No author names, no institutional affiliations in the submission PDF. No self-identifying citations (cite your own prior work in the third person or as [Anonymous]).
- **Template:** Official NeurIPS 2026 LaTeX style file required. The Overleaf template is available at `overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc`. No margin hacks, font size changes, or style modifications permitted — violations are grounds for desk rejection.
- **Text format:** 10pt Times New Roman, 11pt leading, confined to 5.5 × 9 inch text rectangle.
- **Figures:** Every caption must be self-contained — a reviewer should be able to understand the figure without reading the body text. Figures are not exempt from the page count.
- **Equations:** All numbered. If you only use definitions (not theorems), answer N/A on the theory checklist item.
- **Paper checklist:** Must be included in the submission PDF. Papers missing the checklist will be desk-rejected.
- **OpenReview profiles:** All authors must have complete, updated profiles with correct conflict-of-interest information before the submission deadline. Missing profiles cause desk rejection.
- **Supplementary material:** Anonymised ZIP, within stated size limits, no identifying metadata, local paths, or author names.

### Required Sections

NeurIPS papers do not have a mandated section list, but the following structure is standard and expected:

1. **Abstract** — One paragraph (~200–250 words). Must state: the problem, what prior approaches miss, the new principle or finding, quantitative evidence, and cautious scope. Do not overclaim universality.
2. **Introduction** — Sets up why the problem matters to the ML community broadly. States contributions explicitly (4 bullet points, not prose). Does not bury the key finding.
3. **Related Work** — Organised by logic of the paper, not chronology. Establishes that prior work identifies the phenomenon (geometry, alignment) but leaves your specific mechanism open.
4. **Problem Formulation / Definitions** — Defines all mathematical objects before the experiments. Required for NeurIPS Clarity score: x, y, K, d, centroid, distance, margin, uncertainty, overlap, RDM.
5. **Methods** — Detailed enough to reproduce. Dataset, models, layers, pooling, splits, preprocessing, cross-validation, statistical tests. Do not assume reviewers know your notebooks.
6. **Results** — Ordered to build mechanistic argument. Not a flat list of experiments.
7. **Discussion** — Synthesises findings, explains both convergence and divergence, addresses the main tension in the results.
8. **Limitations** — Dedicated section. Scope of claim, dataset specificity, centroid assumptions, observational (not causal) design.
9. **Conclusion** — One clear takeaway sentence. Must match the abstract claim exactly.
10. **References** — Unlimited length, not counted toward page limit.
11. **Appendix** — Full formulas, statistical tables, all phase plots, glossary for ML/neuro terminology, compute table, ethics/checklist support.

### What NeurIPS Reviewers Prioritise

| Criterion | What It Means | This Paper's Status |
|---|---|---|
| **Novelty / Originality** | New mechanism, new synthesis, new angle on an established problem | **Strength:** Three novel contributions (first systematic geometry-to-decision account across 8 LLM variants; first margin-based ambiguity mechanism; cross-system biological validation) |
| **Significance** | Does the result matter beyond this dataset? Can others use the metric? | **Strength:** Margin as a reusable metric for representational ambiguity; generalisation to brain data strengthens significance |
| **Soundness / Quality** | Are claims supported by rigorous statistics, controls, and proper experimental design? | **Gap / Risk:** LOSO, bootstrap CI, and permutation tests not yet applied to LLM text experiments (brain data has them; LLM experiments currently do not). This is the single highest-risk methodological gap. |
| **Clarity** | Can a reviewer reproduce the work from the paper + appendix? | **Partial gap:** Methods section needs explicit layer and pooling documentation per model. |
| **Reproducibility** | Code, seeds, dataset links, figure-generation scripts | **Unresolved:** Anonymised code/data supplement not yet prepared. |
| **Empirical validation** | Are claims tested against appropriate baselines and controls? | **Strength on brain side** (shuffle control, LOSO, permutation tests). **Gap on LLM side** (visual inspection without formal statistical significance). |

**Accept rate context:** NeurIPS receives ~30,000 submissions annually with an accept rate typically in the 25–28% range. With this volume, papers that do not clearly state a novel mechanism and provide statistical validation are rarely accepted regardless of the quality of their visualisations.

---

## How This Study Maps to a Strong NeurIPS Contribution

### The Novelty Case

This paper makes three novel contributions that are distinct from prior embedding geometry and brain-LLM alignment work:

1. **First systematic geometric characterisation of the void-belt-overlap density structure** in LLM emotional embeddings across 8 model variants (pretrained vs fine-tuned, middle vs final layer) and 2 text datasets. Prior work characterises that embeddings have semantic structure; this work shows the specific radial organisation of that structure and its model-dependence.

2. **First cross-system RSA comparing LLM embedding geometry to human fMRI at the emotion category level with a validated margin-based ambiguity mechanism**. Prior brain-LLM alignment work asks whether representations are similar; this work asks whether the same local geometric principle — uncertainty as centroid competition — is observable in neural representations. The margin vs distance contrast (AUC 0.81 vs 0.56) with shuffle control is the specific mechanistic contribution.

3. **Discovery and quantification of the valence-arousal axis inversion** as a fundamental divergence between artificial and biological representational logic. Both systems share a local geometric ambiguity principle (Finding 5) while their global organisation is near-opposite (Finding 6). This coexistence of local convergence and global divergence is the central interpretive contribution.

### Findings to Paper Section Mapping

| Finding | NeurIPS Paper Section | Key Figure / Table | Current Status |
|---|---|---|---|
| F1: The Void + The Belt | Results I (LLM Geometry) | Radial density curves per model | Done — `phase2/overlap_metrics.json` |
| F2: No Pure Emotion + Overlap | Results I (LLM Geometry) | Overlap % table, pairwise confusion matrix | Done — Phase 2 reports |
| F3: Compaction vs Distribution | Results II (Low-Rank Core) | Ablation accuracy decay curves | Done — `phase3/importance_metrics.json` |
| F4: Geometric Clarification in 20D | Results II (Low-Rank Core) | Silhouette improvement table, 20D scatter | Done — Phase 4 reports |
| F5: Cross-System Ambiguity Gradient | Results III / Brain Validation | Margin vs uncertainty scatter + AUC comparison | Brain side done; LLM side needs formal statistical tests |
| F6a: V/A Axis Alignment | Results IV / Discussion | V/A scatter plot per system | Done — `alignment_metrics.json` |
| F6b: RDM 48D | Results IV / Brain Validation | RDM heatmap, correlation table | Done — `relational_validation.json` |
| F6c: RDM 11D | Results IV / Brain Validation | RDM comparison table 48D vs 11D | Done — `rdm_results_11d.json` |

### Narrative Arc for Reviewers

The strongest framing for this paper is as a mechanism-discovery paper, not a descriptive LLM-brain comparison. The narrative should read: LLM emotion embeddings have a specific geometric structure (Findings 1–4); we propose centroid margin as the mechanism underlying representational ambiguity; we validate it in LLMs (geometry predicts logits, r = 0.957–0.988, Finding 5-LLM); we then ask whether the same local mechanism appears in biological emotion representations (brain margin r = 0.61, AUC = 0.81, Finding 5-Brain); we find that it does, but the global organisation is inverted — LLMs are valence-first, brains are arousal-first (Findings 6a–6c). The conclusion is not that LLMs replicate the brain, but that a shared local geometric principle — ambiguity as competitive proximity to rival prototypes — coexists with fundamentally different global representational priorities. This framing makes Phase 5 (geometry-logit coupling) the mechanistic anchor of the entire paper, and the brain data the generalisation test rather than the main event.

---

## Pre-Submission Checklist

### Mandatory NeurIPS Technical Requirements

- [ ] Paper is exactly ≤ 9 content pages (figures and tables counted)
- [ ] References section is outside the 9-page limit
- [ ] NeurIPS 2026 official LaTeX style file used; no font, margin, or spacing modifications
- [ ] PDF format; no author names or affiliations in submission file
- [ ] No self-identifying citations (own prior work cited as third-person or [Anonymous])
- [ ] All figure captions are self-contained (readable without body text)
- [ ] All equations are numbered
- [ ] NeurIPS paper checklist included in submission PDF (desk rejection if missing)
- [ ] All authors have complete, updated OpenReview profiles with conflict-of-interest information
- [ ] Supplementary ZIP is anonymised — no local paths, usernames, or institutional names
- [ ] Ethics statement: include human data wording ("This study uses only publicly available, de-identified neuroimaging data from the NeuroEmo dataset, OpenNeuro accession ds005700 v1.2.0. No new participants were recruited.")
- [ ] Code/data availability statement: anonymised supplementary ZIP with README and run-from-scratch scripts
- [ ] Compute resources reported: hardware, runtime, LLM fine-tuning details

### Scientific Completeness (Team's Open Items)

- [ ] LOSO + bootstrap CI + permutation tests applied to all LLM text embedding experiments (currently only brain data has these) — **blocking**
- [ ] Locate or drop cross-system ambiguity r ≈ 0.56 figure — must either find `comparison_results.txt` or replace with confirmed per-system metrics — **blocking**
- [ ] V/A ground truth citation (Russell 1980 + DEAP/Koelstra) formally inserted in methods section with exact table reference — **blocking**
- [ ] Phase 5 scope decision (main paper vs appendix) — **blocking**
- [ ] fMRI sample size justification sentence in methods — needed before submission
- [ ] Embedding extraction layer documented per model in methods section — needed before submission
- [ ] Aimee's results PDF pushed to repo — if it contains additional validation, its absence leaves the compass incomplete
- [ ] Haxby et al. distributed coding citation — from Aimee's reference list in `LLM_brain.RTF`
- [ ] Noise ceiling comparison stated explicitly: brain-brain r ≈ 0.47 vs brain-fine-tuned LLM r = −0.88
- [ ] Cross-dataset validation result included (20K dataset, April 16) — evidence geometry is not dataset-specific
- [ ] Decide whether 120K image dataset result appears in appendix or is omitted
- [ ] Lock one margin sign convention consistently across LLM and brain sections

### Figure Readiness Checklist

Minimum figures for the paper. Each must exist at its expected path and be paper-ready (axis labels, captions, resolution):

- [ ] **Silhouette comparison** (all 8 LLM variants, 768D): `reports/phase1/visuals/`
- [ ] **Signal decay curves** (accuracy vs dimensions removed, all 4 Phase 3 models): `reports/phase3/`
- [ ] **Cluster overlap plots** (global overlap %, certainty buffer table): `reports/phase2/`
- [ ] **Brain vs LLM density decay comparison**: `experiments/brain_embedding_understanding/checking_density_geometry/`
- [ ] **RDM heatmaps** (48D and 11D, Brain vs fine-tuned LLM vs pretrained LLM): `checking_centroids/` and `checking_centroids_with_spatial_context_data/`
- [ ] **V/A alignment scatter** (PCA projection + ground truth overlay, all 3 systems): `valence-arousal-dimensional_reduction/`
- [ ] **Context retention comparison** (ablation curves: Brain, MPNet pretrained, Qwen fine-tuned): `checking_context_retention_across_dimensions/`
- [ ] **MDS affect maps** (centroid positions in 2D for each system): from Phase 4 / centroid relational reports
- [ ] **Margin vs uncertainty** (brain: scatter plot margin vs classifier uncertainty + AUC bar): `checking_density_geometry/reports/brain finl.html`
- [ ] **Distance-logit correlation** (Phase 5, both fine-tuned models): `reports/phase5/`

---

## Related Work Pointers

**Embedding Geometry**
Prior work has characterised that contextual LLM embeddings encode semantic structure geometrically — including anisotropy of BERT embeddings (Ethayarajh 2019), directional probing for syntactic and semantic features (Belinkov 2022), and mechanistic interpretability work on linear representations of features (Elhage et al. 2022, "Toy Models of Superposition"). Your paper extends this from static structure to ambiguity prediction and decision alignment.

**RSA Methodology**
Representational Similarity Analysis (Kriegeskorte et al. 2008, NeuroImage) is the methodological anchor for the RDM comparison. The 48D→11D brain compression approach is analogous to brain region aggregation in systems neuroscience RSA. The key distinction from standard RSA: this paper validates a specific local geometric mechanism (margin), not just structural similarity.

**Brain-LLM Alignment**
Prior work asks whether LLM representations predict brain activity (Schrimpf et al. 2021, "The Neural Architecture of Language"; Goldstein et al. 2022, Nature Neuroscience). Your paper asks a different question: whether the same local geometric principle for ambiguity — centroid margin competition — is observable in biological emotion representations. This is a mechanism question, not a predictivity question.

**Valence-Arousal Models**
The two-dimensional circumplex model of affect (Russell 1980, JPSP) is the foundational psychological grounding. The DEAP dataset (Koelstra & Mühl et al. 2012, IEEE T-AFFC, 5245 citations) provides empirically validated V/A scores used as ground truth for the axis alignment test. Note from May 1 meeting: V/A values for the 5/6 emotions are assigned relationally from these sources, not mapped to exact single values — this limitation should be stated in the methods and limitations sections.

**Distributed Brain Coding**
Haxby et al. (2001, Science — the face/object distributed coding paper) established that cognitive categories are encoded as distributed patterns of activity rather than in single regions. This is the biological prior that explains why brain fMRI shows negative silhouette in 2D projection but decodable signal in 48D. Aimee is the contact for exact Haxby citation and additional neuroscience references; the full reference list will be in `LLM_brain.RTF` when available.

**Fine-Tuning and Representation Learning**
The finding that fine-tuning compresses emotional signal into a lower-dimensional subspace (Phase 3) connects to literature on representation collapse and linear probing in fine-tuned models (Kumar et al. 2022 "Fine-Tuning can Distort Pretrained Features"; Clark et al. 2020 "What Does BERT Look At?"). The contrast between fine-tuned compression and pretrained redundancy directly parallels debates about the trade-off between task performance and representational robustness.

---

## What the PDFs in This Folder Contain

**`neurips_deep_submission_structure_checklist_report.pdf`** (11 pages)

This is the primary submission-control document for this specific paper. It is not a generic NeurIPS guide — it is tailored to this study. Contents: (1) executive strategy for how to frame the paper as a mechanism-discovery paper rather than a descriptive comparison; (2) the exact recommended narrative arc (7 steps from "representations are geometric" to "conclude with reusable principle"); (3) recommended paper structure in substantial detail, including exact text recommendations for the abstract sentences, introduction paragraphs, methods subsections, and results sections; (4) exact figure strategy for a 9-page paper (6 figures, each with panel descriptions); (5) NeurIPS checklist answer plan with recommended responses to each checklist item; (6) reviewer attack surface table with 8 common criticisms and their pre-emptive fixes; (7) exact action checklist before submission; (8) recommended page budget (page-by-page allocation); (9) final interpretation hierarchy (what claims are strong vs cautious). Pay particular attention to the "What not to claim" section and the wording guardrails in Section 12 — these directly protect against overclaiming biological equivalence or causal relationships.

**`neurips_pre_submission_checklist_full.pdf`** (6 pages)

A complementary pre-submission guide also tailored to this project. Contains: (1) acceptance-oriented narrative structure (the "distance fails → margin works → biological validation" arc); (2) section-by-section table of what each section must contain and which reviewer criterion it serves; (3) main-text figure checklist with required panels per figure; (4) statistics checklist mapping each required test to its recommended reporting format; (5) methods section requirements with minimum details per subsection; (6) submission checklist mapped to all 16 NeurIPS checklist questions; (7) ethics and human data wording template; (8) compute reporting table (pre-filled for brain analysis; LLM compute fields to be filled by Pritish); (9) reproducibility package checklist (README, requirements.txt, script files); (10) pre-upload desk-rejection checklist; (11) likely reviewer questions with pre-emptive paper answers; (12) final readiness scoring table. The statistics checklist in Section 4 and the reviewer questions in Section 11 are the highest-priority sections for the final week before submission.
