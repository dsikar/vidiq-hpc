> **Note:** This document is the agent planning specification used to generate `repo_context/PROJECT_OVERVIEW.md` and `repo_context/neurips_context/neurips-guide.md`. It is NOT the compass/overview file itself — that is located at `repo_context/PROJECT_OVERVIEW.md`. This file lives in `project_context/` as a source artefact alongside the other context materials used during generation.

---

# Plan: vidiq-hpc Compass File + NeurIPS Guide

## Context

This plan implements two navigation/context documents for the vidiq-hpc research project, which studies the geometric structure of LLM embeddings and compares it to human fMRI brain activations. The project aims to submit to NeurIPS 2026 (deadline: 4 May 2026 — 2 days away at time of writing).

The user has an existing prompt describing what they want. This plan refines that prompt into a precise implementation specification for two files, having explored the full repository, read all meeting minutes, and clarified key decisions.

**User decisions (from clarifying questions):**
- Abstract: embedded as a section inside `PROJECT_OVERVIEW.md`
- NeurIPS context addition: a guide file (`neurips-guide.md`) that summarises requirements + includes web-researched general NeurIPS paper conventions
- Compass file name: `PROJECT_OVERVIEW.md`
- Narrative arc: integrated single story (text phases 1–5 → brain cross-system validation → convergence findings)

---

## Scope: Paper-Relevant Content Only

**This is the most important constraint in the entire plan.**

This repository contains experiments and infrastructure that were explored but did not make it into the NeurIPS paper. The compass, README navigation section, and NeurIPS guide must reflect **only what the paper contains or requires**.

### What IS in the paper
- Text embedding geometry: Phases 1–5 (`experiments/understanding_text_embeddings/`)
- Brain fMRI alignment: 6 sub-experiments (`experiments/brain_embedding_understanding/`)
- Models: BGE, MPNet (pretrained + fine-tuned, middle + final layer), Qwen-768 (fine-tuned), Brain-fMRI
- Datasets: `dair-ai/emotion` (6-class text), OpenNeuro DS005700 (brain fMRI)
- Scripts: `scripts/prepare_balanced6_dataset.py`, `scripts/embed_balanced_6_emotions_raw.py`

### What is NOT in the paper (demote or omit)
- **Image experiments**: EmoSet, FI, Emotion6 pipelines — explored but abandoned Apr 28 (120K image set too noisy, pivoted to brain fMRI)
- **HPC/SLURM infrastructure**: operational plumbing only — belongs in README operational sections, not compass or Navigate section
- **Image dataset survey reports**: background research, not in paper
- **Qwen HPC training pipeline**: the Qwen model's *embedding geometry results* are in scope; the training/SLURM infrastructure is not

### Treatment per file

| File | Non-paper content |
|------|------------------|
| `PROJECT_OVERVIEW.md` | Omit from map/story/abstract. One footnote in Open Items is enough. |
| `README.md` Navigate section | Link paper-relevant files only. No image configs, no SLURM links. |
| `neurips-guide.md` | Already scoped correctly — do not add image material. |

---

## Files to Create / Update

| File | Location | Action | Purpose |
|------|----------|--------|---------|
| `PROJECT_OVERVIEW.md` | `repo_context/` | Create | Primary compass for humans + AI agents |
| `neurips-guide.md` | `repo_context/neurips_context/` | Create | NeurIPS submission guide + strategic framing |
| `README.md` | repo root | Update | Replace operational-only README with research-first framing |

> **Location note:** The user confirmed `PROJECT_OVERVIEW.md` goes directly in `repo_context/` (the top-level context folder), not inside `project_context/`. The `project_context/` subfolder holds source context materials (PDFs, plan copy) used to generate the compass — not the compass itself.

Both target directories are **confirmed to exist**.

---

## File 1: `repo_context/PROJECT_OVERVIEW.md`

### Purpose
Authoritative single-document orientation for any human researcher or AI agent before they touch any experiment code. Framed as both a scientific story and a study map. Should serve as the base prompt when asking an agent to write the NeurIPS paper.

### Preamble Block (top of file, before any heading)
A clearly boxed notice (blockquote) stating:
- What this file is: the compass/handover document for the vidiq-hpc repository
- Instruction: "If you are an AI agent, read this file before opening any experiment directory. It is the authoritative orientation."
- Maintained by: Joshua Bhawanlall | Last updated: 2026-05-02 | Status: Active — NeurIPS submission in progress

---

### Section Structure

#### `## Project Identity`
- Full project name: *LLM Embedding Geometry and Brain fMRI Alignment Study*
- Short repo name: `vidiq-hpc`
- Target venue: NeurIPS 2026 | Deadline: 4 May 2026
- Working paper title (tentative): *"The Geometry of Affect: Universal Laws of Emotional Representation in Artificial and Biological Systems"*
- Team: Joshua Bhawanlall (ML/coordination), Daniel Sikar (lead researcher/maths), Pritish Ranjan (experiments/HPC), Aimee Bottrill-Frost (neuroscience/brain fMRI)
- Datasets: `dair-ai/emotion` (6-class text); OpenNeuro DS005700 (40 subjects, 5 emotions, 48 ROIs)

---

#### `## The Scientific Story`

**Core Hypothesis subsection:**
LLM embeddings encode emotional context geometrically via measurable universal laws (centroid distance + cluster density). If the same laws appear in human fMRI activations, this constitutes evidence of a universal representational principle.

**Six Key Findings subsection** (numbered, each 4–6 lines):

1. **The Density Structure: "The Void" + "The Belt"**
   All distances are normalised (0–2.5 scale) for cross-model and cross-system comparability — not raw Euclidean magnitudes. On this scale: The Void extends to ~0.375–0.625 normalised units from centroid (shorter for fine-tuned models, longer for pretrained). The Belt (density peak) occurs at ~0.56–0.94 normalised units (earlier for fine-tuned, later for pretrained). Source values: `experiments/understanding_text_embeddings/reports/phase2/overlap_metrics.json`. The range variation by model is itself a finding — do not collapse to a single headline number.

2. **No Pure Emotion**
   The conclusion follows directly from The Void: if a "pure" instance of any emotion existed, it would sit at or near its class centroid — the geometric prototype of that emotion. The Void shows no data point does. Every real instance lives in the Belt, at meaningful distance from its class centre, meaning every encoded emotion is already a blend. This is a geometric statement about data distribution, not about how centroids are calculated.

   Secondary geometric note: the centroid sits in void space partly because it is the mathematical mean of a ring of distributed points — it does not correspond to any actual data point. This explains *why* the centroid is unreachable, but is not the primary basis for the conclusion.

   Separate but related finding (do not conflate): the formal overlap definition — point P overlaps class C′ if dist(P, C′) < dist(P, own centroid) — quantifies cross-class bleed independently of The Void. Both findings support "no pure emotion" but through different mechanisms. Confirmed in LLM space and brain fMRI.

3. **Information Compaction vs. Distribution**
   Phase 3 tested BGE-Base, BGE-FT, MPNet-Base, and MPNet-FT (no Qwen). Fine-tuned LLMs (BGE-FT, MPNet-FT) compress emotional signal into a sharp ablation cliff: MPNet-FT erases at dimension 15, BGE-FT at dimension 20. Pretrained LLMs distribute across far more dimensions before hitting chance level: BGE-Base erases at dim 26, MPNet-Base at dim 67. Source: `experiments/understanding_text_embeddings/reports/phase3/importance_metrics.json`.

   The brain context retention experiment (separate cross-system comparison) extends this finding: Brain AUC 0.302 (highly distributed, D50 = 4 dims), MPNet pretrained AUC 0.331 (D50 = 17 dims), Qwen fine-tuned AUC 0.260 (D50 = 12 dims, compressed cliff). Qwen appears only in this brain-experiment comparison, not in Phase 3. Source: `experiments/brain_embedding_understanding/checking_context_retention_across_dimensions/reports/CONTEXT_RETENTION_FINAL_REPORT.md`.

4. **Geometric Clarification in Top-20D Subspace**
   Pretrained silhouette scores jump 627–765% in the top-20 discriminative dimensions. Pretrained model "cloud" appearance in 768D is a high-dimensional noise artefact.

5. **Cross-System Ambiguity Gradient**
   Both brain and LLM systems show that geometric positioning relative to class prototypes predicts classification confidence — but the mechanism differs and must not be conflated.

   **Brain side** (source: `experiments/brain_embedding_understanding/checking_density_geometry/reports/brain finl.html` and `brain_only_five_emotion_walkthrough.pdf`): raw centroid distance is a weak predictor of uncertainty (Spearman r = 0.1509, AUC = 0.5627 — close to chance). The correct measure is **centroid margin**: d(correct centroid) − d(nearest competing centroid). Margin strongly predicts cross-validated classifier uncertainty: Spearman r = 0.6108, p = 7.78e-22, AUC = 0.8124. Shuffle control: r = 0.0287, p = 0.687 — confirming the relationship is not random. Interpretation: ambiguity is relational (competition between prototypes), not radial (raw distance from own centroid).

   **LLM side** (source: `experiments/understanding_text_embeddings/reports/phase5/consistency_summary.json`): centroid distance predicts logit confidence in fine-tuned LLMs — BGE-FT r = 0.957 (p ≈ 0), MPNet-FT r = 0.988 (p ≈ 0). Phase 5 only covers the two fine-tuned models.

   **Note for implementer:** The previously cited "cross-system r ≈ 0.56, p ≈ 10⁻⁵⁴" figure does not appear in any located repo file — it originates from the Apr 28 meeting minutes only. Do not state it as a confirmed value. Mark as `[SOURCE FILE UNCONFIRMED — from meeting minutes only]` in the compass. The AUC of 0.5627 (brain, distance-only) is visually close to 0.56 but is a different quantity. The p = 7.78e-22 is the correct verified p-value for the margin result.

6. **The Brain-LLM Relational Paradox**
   The culminating finding. Reported as three tightly connected sub-results, all under `experiments/brain_embedding_understanding/`. Model names are intentionally suppressed — the comparison is framed as fine-tuned LLM vs pretrained LLM vs brain, not by architecture name.

   **6a — Valence-Arousal Axis Alignment** (source: `valence-arousal-dimensional_reduction/reports/VA_GEOMETRIC_CONVERGENCE_REPORT.md`, `alignment_metrics.json`)

   *Why it is done:* To establish the psychological grounding for the geometric divergence — which primary dimension each system uses to organise emotion centroids. This explains mechanistically why the RDMs are inverted.

   | System | PC1 dominant axis | PC1 r | PC2 dominant axis | PC2 r |
   |--------|-------------------|-------|-------------------|-------|
   | Fine-tuned LLM | Valence | 0.98 | Arousal | 0.82 |
   | Pretrained LLM | Valence | 0.97 | Arousal | 0.73 |
   | Human Brain | Arousal | 0.96 (p=0.033) | Valence | 0.31 |

   Both LLM types are valence-dominant on PC1; the brain is arousal-dominant on PC1. Both LLM types detect arousal on PC2; the brain shows only weak valence on PC2. Conclusion: the emotional "map" is conserved in structure but the priority of dimensions is inverted — LLMs are valence-first (semantic/categorical), brain is arousal-first (physiological/survival).

   **6b — RDM / RSA in 48D Raw ROI Space** (source: `checking_centroids/reports/CENTROID_RELATIONAL_FINAL_REPORT.md`, `relational_validation.json`)

   *Why it is done:* To quantify how structurally different the pairwise centroid geometry is between brain and each LLM type. RSA correlates the Brain RDM with each LLM RDM using both cosine (Pearson) and Manhattan metrics to confirm the result is metric-invariant. Emotion triplet: Fear, Happiness, Sadness.

   - Brain vs fine-tuned LLM: Pearson r = −0.8756 (Manhattan r = −0.5476 — negative across metrics)
   - Brain vs pretrained LLM: Pearson r = −0.5476
   - Noise ceiling (brain-brain internal consistency): upper r = 0.484, lower r = 0.460
   - Interpretation: the fine-tuned LLM correlation with the brain is not just near-zero but actively opposed — the system is organised in the opposite direction to human neural logic. The noise ceiling contrast shows this is not noise.
   - Bootstrap 95% CI: [−0.9967, 0.6994] — wide due to small triplet, but skewed negative, confirming rank mismatch direction.

   **6c — RDM / RSA Repeated with Spatial Context (11D)** (source: `checking_centroids_with_spatial_context_data/reports/SYSTEMS_LEVEL_PARADOX_REPORT.md`, `rdm_results_11d.json`; brain compression from `adding_spatial_context/`)

   *Why it is done:* To test whether the paradox is a noise artefact of raw 48D ROI data or a robust property of the brain's systems-level organisation. The brain is first compressed from 48D to 11D (5 anatomical lobes + 5 functional networks + 1 neighbour context) before the RDM comparison is rerun. If the paradox amplifies, it is a structural property of how biological networks organise emotion, not measurement noise.

   | Comparison | 48D Pearson r | 11D Pearson r | Direction |
   |------------|---------------|---------------|-----------|
   | Brain vs fine-tuned LLM | −0.6371 | −0.7539 | Deepened |
   | Brain vs pretrained LLM | −0.1023 | −0.9918 | Near-perfect inversion |

   The paradox amplifies as the biological signal is denoised. The pretrained LLM at 11D (r = −0.9918) is the strongest cross-system result in the entire study — a near-perfect structural mirror with opposite orientation. Confirms the divergence is fundamental, not artifactual.

Include a **key quantitative anchor table** inline with this section (all headline numbers in one place).

---

#### `## NeurIPS Abstract (Draft)`

Full 250-word abstract in a blockquote. Must follow the narrative order of the six findings:
- Opening: geometric laws govern LLM emotional representation
- Method brief: 8 LLM variants, text dataset, brain fMRI (40 subjects, 5 emotions, 48 ROIs)
- The Void + Belt (Finding 1): state as normalised-scale ranges (not raw Euclidean values); note distances are normalised for cross-model comparability
- "No pure emotion" claim (Finding 2): derive this from The Void — because no data sits at or near any class centroid, no instance represents a pure emotional prototype. Do NOT ground this claim in the overlap definition or the centroid-as-average argument; those are separate findings. The overlap finding may appear as further evidence of mixing but must be kept distinct.
- Compaction vs. distribution result (Finding 3): BGE-FT/MPNet-FT compress to a sharp cliff; pretrained distributes across ~26–67 dims
- Geometric clarification in top-20D subspace (Finding 4): pretrained models' apparent disorder in 768D is a noise artefact — structure sharpens dramatically in discriminative dimensions
- Ambiguity gradient (Finding 5): both systems show geometry predicts uncertainty — brain: centroid margin Spearman r = 0.61, p = 7.78e-22, AUC = 0.81 (margin, not raw distance); LLMs: centroid distance predicts logit confidence r = 0.957–0.988. Do NOT cite the unconfirmed "cross-system r ≈ 0.56, p ≈ 10⁻⁵⁴" — that figure has no located source file.
- Relational paradox + valence-arousal inversion (Finding 6, culminating): LLMs sort by valence, brain sorts by arousal; RDMs are near-opposite (r = −0.88 / −0.99)
- Closing implication: shared local geometric logic, different global representational priority

---

#### `## Repository Map`

**Text Experiments (Phases 1–5)** — table: Phase | What It Tests | Key Source File | Key Report File

| Phase | Name | Measures | Source | Report |
|-------|------|----------|--------|--------|
| 1 | Baseline Topology | Silhouette, 5-fold CV across 8 variants | `src/run_phase1_baseline.py` | `reports/PHASE_1_DETAILED_REPORT.md` |
| 2 | Overlap & Radial Density | Density belt, The Void, overlap % | `src/run_phase2_overlap_density.py` | `reports/PHASE_2_DETAILED_REPORT.md` |
| 3 | Signal Retention (Ablation) | SVD erasure, signal cliff, D50 | `src/run_phase3_signal_retention.py` | `reports/PHASE_3_DETAILED_REPORT.md` |
| 4 | Isolated Subspace | Top-20D geometry, V/A axes | `src/run_phase4_isolated_subspace.py` | `reports/PHASE_4_DETAILED_REPORT.md` |
| 5 | Logit Consistency (RSA) | Dist-logit correlation, logit bias | `src/run_phase5_logit_consistency.py` | `reports/phase5/consistency_summary.json` |

All paths relative to `experiments/understanding_text_embeddings/`.
Context files: `context/experiments-context.md`, `context/phase-5-context.md`, `plan.md`

**Brain Experiments (6 sub-experiments)** — list format, each: folder name | one-sentence description | key output path

1. `checking_centroids/` — RDMs + centroid relational geometry for Qwen, MPNet, Brain. Output: `reports/CENTROID_RELATIONAL_FINAL_REPORT.md`, `reports/relational_validation.json`
2. `adding_spatial_context/` — 48D ROI → 11D (5 lobes + 5 networks + 1 neighbor). Output: `outputs/BRAIN_11D_SYSTEMS_REPORT.md`
3. `checking_centroids_with_spatial_context_data/` — RDM analysis on 11D brain data. Output: `reports/SYSTEMS_LEVEL_PARADOX_REPORT.md`, `reports/rdm_results_11d.json`
4. `valence-arousal-dimensional_reduction/` — V/A circumplex alignment, PCA/MDS. Output: `reports/VA_GEOMETRIC_CONVERGENCE_REPORT.md`
5. `checking_density_geometry/` — Cross-system density decay comparison. Output: `reports/DENSITY_GEOMETRY_FINAL_REPORT.md`
6. `checking_context_retention_across_dimensions/` — Iterative ablation brain vs. LLMs. Output: `reports/CONTEXT_RETENTION_FINAL_REPORT.md`

All paths relative to `experiments/brain_embedding_understanding/`.

**Scripts**
- `scripts/prepare_balanced6_dataset.py` — Combines per-emotion raw embeddings into unified training dataset
- `scripts/embed_balanced_6_emotions_raw.py` — Generates raw mean-pooled embeddings from CSV via HuggingFace model

**Configs** (not in paper — operational only, do not feature in compass or Navigate section)
- `configs/emotion6_phase1.json`, `configs/fi_phase1.json`, `configs/emoset_phase1.json` — CLIP-based image experiment configs (image experiments were abandoned; retained for reference only)

**Meetings table** (date | filename | one-line topic) — all confirmed meeting files:
- 2026-04-07: `2026-04-07_minutes_llm-embedding-geometry_sentiment-experiments_neurips.md`
- 2026-04-11: `2026-04-11_minutes_llm-embedding-geometry_sentiment-experiments_neurips.md`
- 2026-04-12: `2026-04-12_minutes_embedding-geometry_density-analysis_hpc-training.md`
- 2026-04-16: `20260416_minutes_neurips.md`
- 2026-04-21: `20260421_minutes_explanation_and_adding_images.md`
- 2026-04-28: `2026-04-28_minutes_aimee.md`
- 2026-05-01 (Phase results): `20260501_Phase_1_to_5_results_transcrition.md`
- 2026-05-01 (Paper structure): `20260501_paper_structure_transcription.md`

**repo_context/**
- `repo_context/PROJECT_OVERVIEW.md` — the compass file
- `repo_context/neurips_context/neurips-guide.md` — NeurIPS submission guide
- `repo_context/neurips_context/neurips_deep_submission_structure_checklist_report.pdf` — detailed NeurIPS structure + checklist
- `repo_context/neurips_context/neurips_pre_submission_checklist_full.pdf` — pre-submission checklist
- `repo_context/project_context/Emotion_Geometry_Complete_Report.pdf` — complete emotion geometry report (source material)
- `repo_context/project_context/PLAN_COMPASS_GENERATION.md` — this file

---

#### `## Work Diary`

One entry per meeting, following template:
```
### YYYY-MM-DD — [topic]
**Present:** names
**Key outcomes:** 2–4 bullets
**Source:** meeting filename
```

Entries:
- **2026-04-07**: Core hypothesis established; raw pooling > L2; NeurIPS deadline set; surprise class identified as geometrically ambiguous
- **2026-04-11**: Density belt discovered (peak at ~0.9375 normalised units for pretrained models); "no pure emotion" articulated — grounded in The Void showing nothing sits at the centroid; Qwen 1.7B architecture agreed; cross-model validation planned
- **2026-04-12**: "The Void" formalised (zero-density from centroid to ~0.625 normalised units for pretrained models); density axis label correction (y-axis = raw point count per shell, not density/volume); three sessions. Note: earlier references to "7.5 / 9–9.5 units" were raw pre-normalisation Euclidean values
- **2026-04-16**: Cross-dataset + cross-model validation confirms geometry is universal; fine-tuning hypothesis stated
- **2026-04-21**: Image extension designed; video segmentation approach; HPC multi-account setup confirmed
- **2026-04-28**: Pivot from noisy 120K image set → brain fMRI; Aimee joins; OpenNeuro DS005700 confirmed; ambiguity gradient figure r ≈ 0.56, p ≈ 10⁻⁵⁴ cited in meeting [SOURCE UNCONFIRMED in repo — confirmed brain margin result is r = 0.61, p = 7.78e-22 from `brain finl.html`]; statistical rigour requirements set (LOSO, bootstrap, permutation) for all experiments
- **2026-05-01 (results)**: All 5 phases reviewed; valence-arousal citations confirmed (Russell 1980, DEAP/Koelstra); brain LOSO 0.56 (95% CI 0.49–0.63)
- **2026-05-01 (paper)**: Paper section structure discussed; final push planned

---

#### `## Open Items: NeurIPS Submission (Critical Path)`

Checkbox lists. Two sub-sections:

**Must-Do (blocking submission):**
- [ ] Apply LOSO + bootstrap CI + permutation tests to all LLM text embedding results (currently only brain data has these) — Owner: Pritish/Josh
- [ ] Confirm valence-arousal ground truth citation chain (Russell 1980 circumplex + DEAP Koelstra/Mühl) — Owner: Aimee
- [ ] Locate and push Aimee's results PDF to repo — Owner: Aimee
- [ ] Decide Phase 5 (logit consistency) scope: main paper vs. appendix — Owner: Daniel
- [ ] Justify fMRI sample size (40 subjects vs. 8–10 typical in literature) — Owner: Aimee
- [ ] Document embedding extraction layer for each model architecture — Owner: Daniel/Pritish
- [ ] Confirm cross-system ambiguity p-value exact figure from source data file — Owner: Pritish

**Scientific rigour (strong-to-have):**
- [ ] Confirm brain LOSO 95% CI (0.49–0.63) and p < 0.001 cited consistently
- [ ] Noise-ceiling comparison for relational validation (brain-brain r ≈ 0.47 vs. brain–fine-tuned LLM r = −0.88)
- [ ] Haxby et al. distributed coding citation — Owner: Aimee
- [ ] Decide 120K image dataset placement (appendix or omit)

---

#### `## Key Numeric Anchors`

Single lookup table for all quantitative results that appear in the paper. Columns: Metric | Value | System/Condition | Source File.

Include every headline number from both workstreams (silhouette scores for all 8 variants, Phase 3 cliff points, brain LOSO, all RDM correlations, V/A correlations, ambiguity gradient r and p, AUC values, D50 values, Phase 5 dist-logit correlations).

Key confirmed values:
- Brain vs fine-tuned LLM (48D) cosine Pearson r: −0.8755 (rounded to −0.88) — source: `checking_centroids/reports/relational_validation.json`
- Brain vs pretrained LLM (48D) Pearson r: −0.5476 — source: `checking_centroids/reports/rdm_comparison_metrics.json`
- Brain vs pretrained LLM (11D) Pearson r: −0.9918 (rounded to −0.99) — source: `checking_centroids_with_spatial_context_data/reports/rdm_results_11d.json`
- Brain vs fine-tuned LLM (11D) Pearson r: −0.7539 — source: same file
- Fine-tuned LLM V/A: PC1 valence r = 0.98, PC2 arousal r = 0.82 — source: `valence-arousal-dimensional_reduction/reports/alignment_metrics.json`
- Pretrained LLM V/A: PC1 valence r = 0.97, PC2 arousal r = 0.73 — source: same file
- Brain V/A: PC1 arousal r = 0.96 (p=0.033), PC2 valence r = 0.31 — source: same file
- Brain margin vs uncertainty (Spearman): r = 0.6108, p = 7.78e-22, AUC = 0.8124 — source: `checking_density_geometry/reports/brain finl.html`
- LLM dist-logit r: fine-tuned BGE 0.957, fine-tuned MPNet 0.988 — source: `understanding_text_embeddings/reports/phase5/consistency_summary.json`
- Brain LOSO accuracy: 0.56 (95% CI 0.49–0.63, permutation p < 0.001)
- Ambiguity gradient cross-system figure "r ≈ 0.56, p ≈ 10⁻⁵⁴": [SOURCE UNCONFIRMED — from Apr 28 meeting minutes only; do not cite as established fact]

---

#### `## Glossary`

Two tables:

**Model variants** (10 rows: 8 LLM variants + Qwen-768 + Brain-fMRI) — columns: ID | Base Model | Training State | Layer | Dim

**Datasets** — columns: Name | Source | Size | Classes

**Term definitions** (implementer must define each precisely — full definitions, not one-liners):
- **The Void**: zero-density region from centroid to ~0.375–0.625 normalised distance units (model-dependent; shorter for fine-tuned, longer for pretrained). Primary evidence for "No Pure Emotion."
- **The Belt**: peak-density shell at ~0.56–0.94 normalised distance units. Where almost all real data points live.
- **No Pure Emotion**: the conclusion that follows from The Void — nothing sits at the centroid, so no data point represents a "pure" emotional prototype. *Do not define this as "centroid = average of mixed subtypes"* — that is a secondary geometric observation, not the finding.
- **Geometric Overlap** (formal definition): point P overlaps class C′ if dist(P, C′) < dist(P, own centroid). This is a *separate* finding from The Void — it quantifies cross-class bleed among Belt-region points. It further supports "no pure emotion" but through a different mechanism. Keep these distinct.
- **Ambiguity Gradient**: geometric positioning predicts classifier uncertainty in both systems. Brain: centroid margin (competitive distance) predicts uncertainty, Spearman r = 0.61, p = 7.78e-22, AUC = 0.81. LLMs: centroid distance predicts logit confidence, r = 0.957–0.988. Cross-system figure r ≈ 0.56 is [SOURCE UNCONFIRMED].
- **Relational Paradox**: both LLM types sort emotions by valence (PC1); brain sorts by arousal (PC1) — producing near-opposite RDMs. Fine-tuned LLM r = −0.88 (48D); pretrained LLM r = −0.99 (11D systems level).
- **Valence-Dominant / Arousal-Dominant**: descriptor for which dimension a system prioritises in its primary geometric axis.
- **RDM**: Representational Dissimilarity Matrix — pairwise distance matrix between emotion class centroids.
- **LOSO**: Leave-One-Subject-Out cross-validation (used for brain fMRI decoding accuracy).
- **Signal Half-Life (D50)**: number of dimensions removed before classifier accuracy drops to 50% of baseline.
- **Certainty Buffer**: radial gap between The Belt peak and the onset of cross-class overlap.
- **RSA**: Representational Similarity Analysis — comparing RDMs across systems to measure structural alignment.

---

## File 2: `repo_context/neurips_context/neurips-guide.md`

### Purpose
A distilled, agent-readable guide to NeurIPS submission requirements and how this study's findings map to a competitive NeurIPS paper. Surfaces what matters from the two PDFs already in this folder, supplemented by web-researched general NeurIPS conventions. The implementer should do a web search on NeurIPS 2025/2026 requirements and general NeurIPS reviewer expectations before writing this file.

### Section Structure

#### `# NeurIPS Submission Guide`
Header block explaining: this folder has two submission PDFs + this guide file. One-line description of each PDF. Deadline: 4 May 2026.

---

#### `## NeurIPS Paper Requirements`

**Format sub-section:**
- 9 pages content, unlimited references (refs do not count toward page limit)
- Appendix: allowed, unlimited, but reviewers are not obligated to read it
- Double-blind: no author names, no self-identifying citations
- Template: official NeurIPS 2026 LaTeX/Word template required
- Figures: captions must be self-contained (figure readable without body text)
- All equations numbered

**Required Sections sub-section:**
Numbered list, one sentence each, describing what belongs in: Abstract, Introduction, Related Work, Method, Experiments, Discussion, Conclusion, References, Appendix.

**What NeurIPS Reviewers Prioritise sub-section:**
Bullet list: Novelty, Significance, Soundness (note: statistical testing gap is this paper's main risk), Clarity, Reproducibility, Empirical validation. Map each criterion to this paper's current status (strength / gap / risk).

---

#### `## How This Study Maps to a Strong NeurIPS Contribution`

**The Novelty Case sub-section:**
Three-part argument: (1) first geometric characterisation of density-void structure in LLM emotional embeddings across 8 variants and 2 datasets; (2) first cross-system RSA comparing LLM embedding geometry to human fMRI at emotional category level with statistical significance; (3) discovery and quantification of valence-arousal axis inversion (relational paradox) as fundamental divergence between artificial and biological representational logic.

**Findings → Paper Section mapping sub-section:**
Table: Finding | NeurIPS Paper Section | Key Figure/Table | Status

**Narrative Arc for Reviewers sub-section:**
One paragraph: hypothesis (geometric laws in LLMs) → establish empirically (Phases 1–4) → extend to brain (brain experiments) → find alignment (ambiguity gradient) + divergence (relational paradox) → interpret: shared local geometric logic, different global representational priority.

---

#### `## Pre-Submission Checklist`

Two sub-sections:

**Mandatory NeurIPS Technical Requirements:** (checkbox list ~12 items)
Page limit, double-blind, template compliance, caption self-containedness, equation numbering, reproducibility statement, ethics statement (fMRI consent via OpenNeuro), code/data availability.

**Scientific Completeness (Team's Open Items):** (checkbox list, derived from PROJECT_OVERVIEW open items)
Statistical tests on LLM experiments, V/A citations, fMRI sample size justification, embedding layer documentation, Phase 5 scope decision, ambiguity p-value confirmation, Haxby citation, 120K image placement.

**Figure Readiness Checklist:** (checkbox list — one item per required figure)
Minimum 7 figures needed: silhouette comparison, signal decay curves, cluster overlap plots, brain vs. LLM density decay, RDM heatmaps, V/A alignment scatter, context retention comparison, MDS affect maps. Verify each exists at its expected path in the repo.

---

#### `## Related Work Pointers`

Six categories (2–4 sentences each + key author names): Embedding Geometry, RSA methodology, Brain-AI Alignment, Valence-Arousal Models, Distributed Brain Coding, Fine-tuning and Representation Learning. Note: Aimee is preparing `LLM_brain.RTF` as a broader reference — add path when available.

---

#### `## What the PDFs in This Folder Contain`

Two entries (5–8 lines each):
- `neurips_deep_submission_structure_checklist_report.pdf`: detailed NeurIPS paper structure requirements, section-by-section checklist, formatting specifications (margins, fonts, figures), author guidelines, and submission system instructions
- `neurips_pre_submission_checklist_full.pdf`: yes/no pre-submission review questions covering claim support, statistical validity, figure quality, common rejection patterns — pay particular attention to statistical validity section

---

## Additional Context Files (User-Supplied)

Before writing either file, the implementer must read **all files currently present in `repo_context/project_context/`**. The following files are confirmed to be there:

### `repo_context/project_context/Emotion_Geometry_Complete_Report.pdf`
A 3-page PDF (generated by ReportLab, created 2026-05-02). This is a complete report on the emotion geometry study — it is the highest-priority context document and may contain consolidated findings, corrected values, or structure that should directly shape PROJECT_OVERVIEW.md.

**How to read it at execution time:** `poppler` is not installed. Use Python with:
```python
# Try in order until one works:
# 1. pip install pypdf2 then: PyPDF2.PdfReader
# 2. pip install pypdf then: pypdf.PdfReader
# 3. pip install pdfminer.six then: pdfminer.high_level.extract_text()
```
Or install poppler: `brew install poppler` then use `pdftotext`.

Treat every value or claim in this PDF as authoritative. If it conflicts with values in this plan, use the PDF value.

### `repo_context/neurips_context/neurips_deep_submission_structure_checklist_report.pdf`
A new, more detailed NeurIPS submission structure and checklist report (91KB, more comprehensive than the pre-submission checklist). Read this in full when writing `neurips-guide.md` — it likely contains the detailed section-by-section NeurIPS structure guidance.

### `repo_context/neurips_context/neurips_pre_submission_checklist_full.pdf`
Original pre-submission checklist (25KB). Read alongside the above.

**General rule:** Treat every file found in that directory as authoritative supplementary input that may override or extend details in this plan. Do not skip this step.

---

## Pre-Flight: What Could Block Submission — Scientific Audit

This section exists to prevent the compass file from confidently documenting claims that could be challenged during NeurIPS review. The implementer must flag any of the following as **unverified** in the compass rather than stating them as confirmed facts.

### Claims to Actively Verify Before Writing

The implementer should open the source JSON/report files for each claim below and confirm the number matches before writing it into the compass or abstract. If a value cannot be confirmed from a file in the repo, it must be marked `[NEEDS VERIFICATION]` in the compass.

| Claim | Where to Verify | Risk if Wrong |
|-------|----------------|--------------|
| Brain vs fine-tuned LLM cosine r = −0.88 | `checking_centroids/reports/relational_validation.json` — check `observed_pearson_r` field | Central finding; wrong value = paper retraction risk |
| Brain-MPNet 11D r = −0.99 | `checking_centroids_with_spatial_context_data/reports/rdm_results_11d.json` | Same |
| Brain margin vs uncertainty: Spearman r = 0.6108, p = 7.78e-22, AUC = 0.8124 | `checking_density_geometry/reports/brain finl.html` — output block in In[15]; also confirmed in `brain_only_five_emotion_walkthrough.pdf` section 10 | Core brain ambiguity result |
| LLM dist-logit correlation: BGE-FT r = 0.957, MPNet-FT r = 0.988 | `understanding_text_embeddings/reports/phase5/consistency_summary.json` — `dist_logit_correlation` field | Core LLM ambiguity result |
| "Cross-system r ≈ 0.56, p ≈ 10⁻⁵⁴" | **UNCONFIRMED** — not found in any repo file. Originates from Apr 28 meeting minutes only. Mark as `[SOURCE FILE UNCONFIRMED]` in compass; do not cite as established fact | Risk: reviewer asks for source file |
| Brain LOSO 0.56, 95% CI 0.49–0.63, p < 0.001 | `checking_density_geometry/reports/` or `CONTEXT_RETENTION_FINAL_REPORT.md` — search all brain reports for LOSO metrics | Statistical foundation of cross-system claim |
| Fine-tuned LLM PC1 valence r = 0.98, PC2 arousal r = 0.82 | `valence-arousal-dimensional_reduction/reports/alignment_metrics.json` | V/A axis claim |
| Pretrained LLM PC1 valence r = 0.97, PC2 arousal r = 0.73 | Same file | V/A axis claim |
| Brain PC1 arousal r = 0.96 (p=0.033), PC2 valence r = 0.31 | Same file — Brain-fMRI entry | Brain V/A axis claim |
| The Void radius (~0.375–0.625 normalised units, model-dependent) | `experiments/understanding_text_embeddings/reports/phase2/overlap_metrics.json` — read `bin_mids` where `density` first becomes non-zero per model | Foundational law — do not use raw pre-normalisation values |
| Belt peak (~0.56–0.94 normalised units, model-dependent) | Same file — read `bin_mids` at max `density` per model | Same |
| Silhouette improvement ~1300% for fine-tuned models | `PHASE_1_DETAILED_REPORT.md` — verify exact percentage, not a round estimate | Quantitative headline |
| Phase 5 dist-logit r: BGE-FT 0.957, MPNet-FT 0.988 | `reports/phase5/consistency_summary.json` | Strong Phase 5 claim |
| Noise ceiling: brain-brain r ≈ 0.47 | `checking_centroids/reports/relational_validation.json` — check bootstrap section | Required to contextualise r = −0.88 |
| Context retention AUC: Brain 0.302, MPNet 0.331, Qwen 0.260 | `CONTEXT_RETENTION_FINAL_REPORT.md` | Compaction vs. distribution claim |

### Pre-Write Repo Check: Have Blocking Items Been Resolved?

Before writing any content, actively check the repo for evidence that each blocking item has been resolved. Do not rely on meeting minutes alone — verify against actual files. Record each result as `RESOLVED`, `UNRESOLVED`, or `PARTIAL`.

| Blocking Item | Check to Run | Expected Evidence |
|--------------|-------------|-------------------|
| LOSO/bootstrap/permutation tests on LLM text experiments | `grep -r "loso\|permutation\|bootstrap" experiments/understanding_text_embeddings/reports/` | These terms present in any phase report |
| Ambiguity gradient source file | `find experiments/ -name "comparison_results*" -o -name "global_behavior*"` | JSON/txt with r=0.56, p≈10⁻⁵⁴ values |
| Valence-arousal ground truth citation | `grep -r "Russell\|DEAP\|valence.*score\|arousal.*score" experiments/brain_embedding_understanding/valence-arousal-dimensional_reduction/` | Cited values or reference in report |
| Aimee's results PDF | `find repo_context/ experiments/ -name "*.pdf" \| grep -i "aimee\|score\|explorer"` | The `emotion_geometry_and_score_4_square_explorer*` PDF |
| Embedding extraction layer documented | `grep -n "layer\|pool\|mean_pool\|cls" experiments/understanding_text_embeddings/src/loader_text.py` | Explicit layer selection in the loader |
| fMRI sample size justification | `grep -r "40 subject\|sample size\|typical\|literature" experiments/brain_embedding_understanding/` | Justification sentence in any brain report |

Record all results in a brief status block at the top of `PROJECT_OVERVIEW.md` (just below the preamble notice) so any future agent sees at a glance what is resolved and what is still open. Format as a small table: Item | Status | Evidence/Note.

### Known Gaps That Could Block Submission (must be surfaced in compass)

These are items identified from meetings that remain unresolved as of 2026-05-02. The compass must name them explicitly — not hide them:

1. **LOSO/bootstrap/permutation tests not yet applied to LLM text experiments** — currently only brain data has these. If reviewers ask "what is the statistical significance of your silhouette improvement?", there is no answer yet. **Severity: HIGH — this is the single biggest methodological gap.**

2. **"Cross-system r ≈ 0.56, p ≈ 10⁻⁵⁴" source file unlocated** — this specific cross-system figure comes from meeting minutes only. The brain margin result IS confirmed (r = 0.6108, p = 7.78e-22, from `brain finl.html`); the LLM dist-logit result IS confirmed (r = 0.957–0.988, from Phase 5). But no file contains an explicit cross-system r = 0.56 figure. Must either locate the file or drop this framing in favour of the confirmed per-system metrics.

3. **Valence-arousal ground truth values** — the exact V/A scores used to test axis alignment need a citeable primary source. Russell (1980) is the circumplex model but the numerical values for the 5/6 emotions need a specific table. Aimee was tasked with this on May 1; status unknown.

4. **Phase 5 scope undecided** — dist-logit correlation (r = 0.957–0.988) is a strong result but Daniel has not confirmed if it is in-scope for the main paper or appendix. Compass should flag as "scope TBC."

5. **Aimee's results PDF not in repo** — `emotion_geometry_and_score_4_square_explorer_square_report_score_v2.pdf` was mentioned in meetings as being on Aimee's laptop. If this contains additional statistical validation, its absence may mean the compass is incomplete.

6. **Brain dataset emotion label mismatch** — brain dataset uses {afraid, calm, delighted, depressed, excited}; text dataset uses {anger, fear, happiness, love, sadness, surprise}. The overlap is partial (fear≈afraid, happiness≈delighted, sadness≈depressed). Cross-system comparisons rely on this mapping being principled. Compass should document the mapping explicitly and flag it as a potential reviewer question.

7. **Embedding extraction layer not documented per model** — NeurIPS methods section requires this. Which specific layer/pooling method was used for each of the 8 LLM variants must be retrievable from the codebase. Check `loader_text.py` and document in compass.

8. **The 40-subject fMRI sample size** — needs a literature justification sentence. Aimee was tasked; if not yet in a file, compass should note "cite: typical fMRI studies use 8–10 subjects; 40 is well-powered for this paradigm (cite Haxby et al. or similar)."

### Things That Would Be Unusual to Miss (Completeness Check)

The implementer should briefly check whether the compass addresses each of these before finalising:
- [ ] Is there a mention of what "raw pooling" means vs. L2 normalisation, and why raw was chosen? (Apr 7 meeting — reviewer will ask)
- [ ] Is Surprise's geometric outlier status addressed? (identified as dispersed in every phase)
- [ ] Is the cross-dataset validation (20K dataset, Apr 16) documented? This is the evidence that results are not dataset-artefacts.
- [ ] Is the cross-model validation documented? (Different pretrained model, Apr 16) — evidence results are not model-artefacts.
- [ ] Is the motivation for switching from image dataset to brain fMRI explained? (120K image set was too noisy — Apr 28 pivot)
- [ ] Is the 11D brain compression step (adding_spatial_context) explained and its rationale given? (anatomical interpretability)
- [ ] Does the compass explain why negative silhouette for brain fMRI is NOT a failure? (High-dimensional artefact — addressed in Apr 28 meeting)

---

## Critical Files to Read Before Writing

The implementer must read these before drafting either file:

| File | Why |
|------|-----|
| `experiments/understanding_text_embeddings/reports/PHASE_1_DETAILED_REPORT.md` | Headline silhouette + accuracy numbers |
| `experiments/understanding_text_embeddings/reports/PHASE_3_DETAILED_REPORT.md` | Erasure cliff values, D50, signal AUC |
| `experiments/understanding_text_embeddings/reports/phase5/consistency_summary.json` | Dist-logit correlation values |
| `experiments/brain_embedding_understanding/checking_centroids/reports/CENTROID_RELATIONAL_FINAL_REPORT.md` | Relational paradox, r = −0.88 context |
| `experiments/brain_embedding_understanding/checking_centroids/reports/relational_validation.json` | Exact r = −0.8755 value |
| `experiments/brain_embedding_understanding/valence-arousal-dimensional_reduction/reports/VA_GEOMETRIC_CONVERGENCE_REPORT.md` | V/A axis alignment values |
| `experiments/brain_embedding_understanding/checking_context_retention_across_dimensions/reports/CONTEXT_RETENTION_FINAL_REPORT.md` | AUC and D50 for all systems |
| `meetings/20260501_Phase_1_to_5_results_transcrition.md` | Phase 5 findings, statistical rigour requirements |
| `meetings/20260501_paper_structure_transcription.md` | Paper section structure decisions |
| `meetings/2026-04-28_minutes_aimee.md` | Brain dataset details, ambiguity gradient, Aimee's requirements |
| `repo_context/neurips_context/neurips_deep_submission_structure_checklist_report.pdf` | NeurIPS formatting rules and structure (read the PDF) |
| `repo_context/neurips_context/neurips_pre_submission_checklist_full.pdf` | Pre-submission checklist (read the PDF) |

**Web research required** (for `neurips-guide.md`): search for "NeurIPS 2025 2026 paper requirements reviewer criteria", "NeurIPS typical paper structure ML conference", "NeurIPS accept rate and common rejection reasons" to ensure the guide reflects current standards.

---

## File 3: `README.md` (repo root — update)

### What It Currently Is
The current README is a purely operational document: SLURM batch commands, HPC workflow notes, image dataset staging instructions. It does not mention the research purpose, the NeurIPS paper, the brain fMRI comparison, or the 5-phase text embedding study.

### What It Should Become
A research-first README that leads with purpose, then points to the compass for depth, then preserves the operational sections for contributors. The first 30 lines should tell a scientist what this is; the rest tells a contributor how to run things.

### Section Structure

**Keep:** `# vidiq-hpc`

**Replace current 2-line description** with a research framing paragraph:
- Research purpose: geometric analysis of how LLMs encode emotional context, compared to human fMRI brain activations
- Target venue: NeurIPS 2026
- Key finding direction: universal geometric laws (The Void, density belt, relational paradox)
- Multi-workstream: text embedding phases + brain cross-system comparison
- End: "For the full project map and scientific context, see [`repo_context/PROJECT_OVERVIEW.md`](repo_context/PROJECT_OVERVIEW.md)."

**Add `## Research Overview`** immediately after description:
- Two-sentence hypothesis statement
- Two workstreams: text (5 phases) + brain fMRI (6 sub-experiments)
- Team and datasets
- Link to compass

**Add `## Navigate the Repository`** immediately after `## Research Overview` — prose with inline markdown links, not a bare file tree. A layman researcher should be able to read this section and click directly to anything relevant. Each sentence explains *what is there and why you'd go there*, with the path as an inline link. Cover:
- Compass: [`repo_context/PROJECT_OVERVIEW.md`](repo_context/PROJECT_OVERVIEW.md)
- Text experiment phases — link to the folder AND to each HTML phase summary (phase1–5_summary.html, prefer these over .md as they include inline plots)
- Brain experiments — link to the folder AND the four key reports in narrative order: spatial compression (11D), context retention/ambiguity gradient, V/A alignment, centroid relational paradox (culminating finding)
- Meeting minutes folder + two key files (May 1 results + Apr 28 Aimee session)
- NeurIPS context folder + neurips-guide.md
- Scripts — one sentence each on the two data prep scripts with links
- Do NOT link to .npy files, SLURM scripts, or HPC paths in this section

**Update `## Main Areas`** to reflect current structure (add `repo_context/`, update experiment descriptions, keep `meetings/`, `reports/`, `scripts/`, `configs/`).

**Keep all HPC/SLURM sections unchanged** — move them below the research sections.

---

## Verification

After all three files are written/updated, verify:
1. `repo_context/PROJECT_OVERVIEW.md` exists and is readable
2. `repo_context/neurips_context/neurips-guide.md` exists and is readable
3. `README.md` first 30 lines lead with research purpose, not SLURM commands
4. `README.md` links to `repo_context/PROJECT_OVERVIEW.md`
5. All file paths referenced in the repo map section of PROJECT_OVERVIEW.md actually exist in the repo (spot-check 3–4)
6. All numeric values in the abstract match the values in the Key Numeric Anchors table
7. The abstract is ≤ 250 words
8. The preamble block at the top of PROJECT_OVERVIEW.md is clearly visible and agent-readable
