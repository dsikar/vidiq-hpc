# Meeting Minutes — Embedding Write-Up Review, Phase 5 Validation, and Brain Geometry Walkthrough

**Date:** 1 May 2026, 16:53  
**Duration:** 3h 13m  
**Attendees:** Daniel Sikar, Pritish Ranjan (PG-Verma), Amy  
**Transcript:** `2026-05-01_1_transcript.md`

---

## Meeting Purpose

- Review the completed text-embedding experiment phases and decide what is paper-ready.
- Capture the new phase 5 validation linking geometric overlap to classifier logits.
- Walk through the brain-data preprocessing and geometry pipeline before combining the brain and LLM stories.

---

## Text Embedding Review

### Phase 1 — Clusters and Centroids

- Fine-tuned models, especially final-layer embeddings, continue to show the best cluster quality and separation.
- The centroid-distance story remains coherent: emotionally similar classes sit closer together, while semantically opposed classes are further apart.
- The reports and HTML summaries are useful for shortlist/review, but not every plot or metric should go into the paper unchanged.

### Phase 2 — Density, Overlap, and Ambiguity

- Radial density decay continues to support the main geometry story: fine-tuned models reach their density peak earlier, consistent with tighter clusters.
- A key clarification was made: the normalization in the embedding-only plots is between the embedding models, not against the brain data.
- The overlap peak tends to appear after the density peak. This supports the idea that ambiguity becomes strongest once the representation moves beyond the densest region of a class.
- The current wording around the "certainty buffer" needs revision. It should not be described as the start of ambiguity; it is better described as the gap between the density peak and the overlap peak.
- Overlap heatmaps are useful, but the visual color scales need to be made comparable across models before they are paper figures.

### Phase 3 — Recursive Direction Removal

- Recursive removal of high-importance directions shows that fine-tuned models begin with stronger class signal but decay faster once the most informative directions are removed.
- This supports the claim that emotional information is concentrated in a relatively small subspace rather than spread evenly across the full embedding space.
- The section currently introduces too much custom terminology. The team agreed this should be simplified or aligned with standard venue/literature phrasing.
- Terms like SVD and related classifier-weight extraction language need clearer definitions in the report.

### Phase 4 — 20-Direction Subspace

- Restricting the analysis to the top 20 learned directions improves silhouette scores, especially for the base/pretrained models.
- This suggests that emotional structure can be recovered more cleanly from a focused subspace than from the full embedding representation.
- An important technical clarification was made: "20 directions" are not the same thing as 20 raw embedding indices. Each direction is a learned combination over the original 768-dimensional space.
- A follow-up check is needed to measure how many original indices are implicated by those extracted directions and to compare this with a random-index removal baseline.
- This line of work may support a greener alternative to fine-tuning: extracting the right semantic subspace from a pretrained model rather than retraining the whole model for a narrow task.

---

## Phase 5 — Logit / Geometry Consistency

- A new validation phase was added to test whether points that geometrically overlap another class also receive stronger classifier evidence for that other class.
- For the fine-tuned models, the number of overlapping samples is low but non-trivial:
  - **BGE-FT Final:** 76 overlaps, about 1.88%
  - **MPNet-FT Final:** 71 overlaps, about 1.76%
- The logit-agreement rate was reported as very high, meaning the classifier generally prefers the same class that the geometric margin says the point is closer to.
- The phase 5 scatter plots suggest an approximately linear relation between geometric margin and logit margin.
- This is one of the strongest results from the meeting: the classifier behaviour appears to track the same ambiguity structure that is visible in the embedding geometry.
- The team agreed that the phase 5 plots need clearer sample-count labelling so it is obvious which plots show all points and which show only overlapping cases.

---

## Brain Data Walkthrough

- The brain data discussion focused on the **NeuroEmo** dataset.
- The retained categories are five emotion conditions: `afraid`, `calm`, `delighted`, `depressed`, and `excited`.
- The final analysis object is not raw fMRI images; it is a subject-by-emotion feature matrix with:
  - **40 participants**
  - **5 retained emotions**
  - **200 total samples**
  - **48 cortical ROI dimensions**
- These 48 dimensions come from the Harvard-Oxford cortical regions of interest and are treated as the high-dimensional brain representation for geometry analysis.
- Repeated blocks per participant/emotion are averaged to yield stable ROI vectors, producing subject-level representations suitable for centroid geometry.
- Feature scaling is necessary so high-variance ROIs do not dominate Euclidean distances.
- Centroids are to be described as class prototypes or average patterns, not as "pure" emotions or literal neural states.
- PCA is only a visual summary; the actual centroid distances, margins, and RDMs are computed in the native high-dimensional space.
- Label-shuffle validation collapses the geometry, supporting the claim that the observed structure is signal rather than an artefact of memorization or arbitrary labeling.
- A key conceptual distinction was reiterated: density is supportive evidence, but margin/uncertainty-style analysis is a stronger validation route for the paper’s main claim.

---

## Cross-System Direction

- The brain analysis is now being organized in phases analogous to the text work:
  - checking density geometry
  - checking centroids
  - checking centroids with spatial context
  - checking context retention across dimensions
  - valence/arousal dimensional reduction
- The team’s aim is not to present text and brain results as two separate stories. The stronger goal is to show that they share compatible geometric structure.
- Phase 5 in text space and margin-based analysis in brain space were discussed as the bridge toward a unified geometry claim.

---

## Paper / Reporting Decisions

- Avoid idiosyncratic terminology unless it is already standard in the relevant literature or venue.
- Add explicit definitions for terms like SVD, RDM, BOLD, geometric margin, and related technical language.
- Make sure captions and prose clearly distinguish between:
  - the original 768-dimensional embedding space
  - the extracted 20-direction subspace
- Add counts and clearer plot annotations wherever sample subsets are being shown.
- Shortlisting of paper figures should happen only after wording and visual consistency issues are fixed.

---

## Action Points

| Owner | Action |
|---|---|
| Pritish | Revise phase 2 wording around normalization, ambiguity gradient, and certainty buffer. |
| Pritish | Fix overlap-heatmap color-scale comparability before using those plots in the paper. |
| Pritish | Simplify phase 3-4 terminology, define SVD more clearly, and distinguish raw dimensions from learned directions throughout the reports. |
| Pritish | Add explicit sample counts to the phase 5 scatter plots and overlap-only plots. |
| Pritish | Quantify how many original indices are implicated by the extracted 20 directions and compare this with a random-index removal baseline. |
| Pritish + Josh | Continue the remaining geometry/classifier follow-up experiments and the crossover work linking the text and brain analyses. |
| Amy | Provide the brain-only walkthrough material, glossary, and notebook/report references into the working repo flow so the preprocessing details are preserved. |
| Daniel | Review the write-up and figures from a reviewer perspective, removing avoidable holes and unnecessary custom jargon. |
| Daniel + team | Run later drafts through multiple critical / contrarian LLM review passes before submission. |

---

## Open Questions

- What is the best paper-safe term for the ambiguity story: ambiguity, uncertainty, probability, or another more standard venue-aligned term?
- How should the 20-direction result be presented without implying that only 20 raw embedding coordinates matter?
- What is the minimum figure set needed to support the unified geometry claim without overcrowding the paper?
