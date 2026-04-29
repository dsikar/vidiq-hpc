# Meeting Minutes — fMRI Brain Data, Ambiguity Gradient, and Statistical Validation

**Date:** 28 April 2026  
**Time:** 6:36 pm (duration approx. 1 h 45 m)  
**Present:** Daniel Sikar, Pritish Ranjan (PG-Verma), Aimee Bottrill-Frost (biomedical science), Josh, Freya Myo, Andrew, Arj  
**Topic:** Introduction of brain fMRI dataset; ambiguity gradient cross-system comparison; statistical validation of LLM and brain geometry results; signal decay ablation experiments

---

## 1. Introduction of Aimee Bottrill-Frost

- Aimee joined the team bringing a **Biomedical Science / Neuroscience** background.
- Her contribution is the statistical treatment and neuroscientific interpretation of the brain fMRI data and its comparison with the LLM embedding results.
- The initial motivation for her involvement was that the 120 K image sentiment dataset proved too noisy for clear cluster separation; the team pivoted to the fMRI / LLM cross-system comparison as a stronger research direction.
- Aimee had previously done unpublished analysis on the fMRI data and found structure that matched the patterns Pritish had been observing in LLM embedding space.

---

## 2. fMRI Brain Dataset

### 2.1 Source

- **Portal:** [OpenNeuro](https://openneuro.org)
- **Dataset ID:** `DS005700`
- **Name:** Neural MO — fMRI Dataset for Emotion Recognition
- **Published / Last updated:** approximately 10 months before this meeting (i.e. mid-2025)
- Download command (recorded in notebook — see §7.2):

  ```
  openneuro.py download dataset DS005700 target_directory
  ```

### 2.2 Structure

- **40 subjects** × **5 emotions**: afraid, calm, delighted, depressed, excited
- **48 features per subject-emotion observation** — each feature corresponds to an ROI (Region of Interest) activation level, mapping to named brain areas including the amygdala, hippocampus, insula, angular gyrus, and prefrontal cortex.
- Activations across all 48 ROIs are recorded **concurrently** for each observation; there is no single dominant region.
- Aimee filtered the raw OpenNeuro data to the 48 emotion-relevant ROI columns and assigned emotion target labels, producing the CSV used for all subsequent analysis.

### 2.3 Dataset Size Compared with the Literature

- Pritish raised a concern about the dataset being small relative to the LLM data (≈ 20 000 examples).
- **Aimee clarified:** most published fMRI emotion studies use 8–10 subjects; 40 subjects is above the field standard. This should be stated explicitly in the paper to pre-empt reviewer objections about sample size.

**Action — Aimee / Daniel:** Add a sentence in the methods / data section citing typical fMRI study sizes and asserting that 40 subjects is above standard for this domain.

---

## 3. Ambiguity Gradient — Core Novel Finding

Aimee provided the following written definition (read aloud into the transcript record):

> *"The ambiguity gradient measures whether emotional uncertainty increases as a brain activity pattern moves away from the prototypical pattern for its labelled emotion. In our case, each emotion centroid represents an average distributed ROI activation pattern. Samples close to that centroid are more typical and easier to classify, while samples further away are more mixed and ambiguous. This suggests that emotional ambiguity is not just noise — it is structured geometrically in the brain's representational space. This is an important cross-system result: LLMs show a similar ambiguity-distance relationship. Even though brain and LLM may differ in global category structure and redundancy, they appear to share a local geometric principle — uncertainty increases with distance from emotional prototypes."*

- This mirrors the finding already observed in the LLM work: the centroid of an emotion class represents the **purest** instance of that emotion, not the most extreme.
- The result is described as potentially novel because prior literature has examined representational similarity between systems but has not characterised how **uncertainty behaves as a function of representational distance**.

### 3.1 Quantitative Result

- **Cross-system correlation:** ≈ 0.56
- **Permutation p-value:** ≈ 10⁻⁵⁴ (effectively zero — near-zero probability that the observed correlation is due to chance)
- Normalization of brain embeddings (L2 / Z-score) was applied before comparison because the raw 48-dimensional activation values are not inherently scaled; Pritish noted the same justification applies to not relying on embedding magnitude in the LLM work.

---

## 4. Statistical Validation — Brain Data (Aimee's Standalone Results)

Aimee described the following tests, run independently on her laptop on the brain data:

| Test | Result |
|---|---|
| LOSO decoding accuracy | 0.56 (chance: 0.20 for 5 classes) |
| 95 % bootstrap CI | 0.49 – 0.63 |
| Permutation test | p < 0.001 (highly significant) |

- **Interpretation (Aimee's written paragraph read into record):**
  > *"Although low-dimensional visualisations of the brain's representational space do not exhibit clearly separated clusters, this does not imply an absence of structure. In a high-dimensional space (48 dimensions) the emotional states form overlapping but statistically separable distributions. This is confirmed by LOSO decoding, which achieves an accuracy of 0.56, substantially above the five-class chance level of 0.20, with a 95 % bootstrap CI of 0.49–0.63. A permutation test further demonstrated that this performance is highly significant (p < 0.001), with real accuracy far exceeding the null distribution. The apparent lack of distinct clusters in 2D projections arises from dimensionality reduction and substantial inter-subject variability, which blurs boundaries when aggregated. Biologically this reflects the distributed and multifunctional nature of neural systems, where emotional states are encoded as graded, overlapping patterns of network activity rather than discrete isolated categories."*

- This statistical result **reconciles the negative silhouette score** (Phase 1, see §5) with actual decodability: the clusters are not visually separable in 2D but are statistically separable in 48D.
- Connects to prior work by **James V. Haxby** (reference to be supplied by Aimee) on cognitive and emotional states encoded as distributed patterns rather than in single regions.

**Action — Aimee:** Compile and send full reference list (neuroscience literature) to Daniel and Pritish, particularly for the Haxby citation and the prototype-theory / representational-similarity work.

---

## 5. Experiment Results — File References

All experiment HTML reports below are on **Pritish's MacBook** in the `understanding_text_embeddings` experiment folder. These are **to be pushed to GitHub** (see action points).

### 5.1 Phase 1 — Unified Global Geometry (Redo)

**File:** `experiments/understanding_text_embeddings/reports/phase_one_summary_of_all.HTML`

Content:

- Table titled **"Unified Phase 1 Global Geometry (redo)"** — intrinsic emotion geometry across 5 LLM configurations plus brain fMRI, with L2 and Z-score normalization applied.
- LLM configurations in the table:
  1. **BGE balanced** — pre-trained BGE model, balanced dataset, 768-dim
  2. **BGE DSAIRI** — pre-trained BGE model, DSAIRI dataset, 768-dim
  3. **MPNet balanced** — pre-trained MPNet model, balanced dataset, 768-dim
  4. **Qwen native** — fine-tuned Qwen model, 2048-dim → PCA → 768-dim
  5. **Qwen fine-tuned** — fine-tuned Qwen model (native embeddings)
  6. **Brain fMRI** — Amy's dataset, 48-dim, 40 subjects × 5 emotions
- **Silhouette score** column: Qwen fine-tuned scores highest (tightly packed, well-separated clusters); pre-trained models lower; **brain fMRI is negative** (visually overlapping, not discretely clustered — explained by high dimensionality reduction artefacts, not absence of structure; see §4).
- **Centroid PCA visualisation** also included: LLMs show separable clouds; brain data shows distributed, overlapping patterns consistent with negative silhouette.

### 5.2 Phase 2 — Linear Probing and 1D Directionality

**File:** `experiments/understanding_text_embeddings/reports/phase_2_1_score_summary.HTML`

Content:

- A regression model was trained on each set of embeddings to predict the emotion label; this tests how well the emotion context is preserved in the embedding.
- Key results:

  | Configuration | Accuracy |
  |---|---|
  | Best pre-trained LLM | 95.29 – 95.59 % |
  | Qwen fine-tuned (best) | ≈ 97 % |
  | Brain fMRI (48-dim) | ≈ 63 % |

- All accuracies substantially above chance (20 % for 5 classes; ≈ 17 % for 6 classes).
- Brain fMRI lower accuracy is expected given small dataset (≈ 200 samples) and 48-dim space; the 63 % is still meaningful.
- **1D projection test:** bringing embeddings down to a single dimension still retained significant emotion-predictive accuracy — suggests emotional context is concentrated in very few directions.

- **Methodology note:** embeddings used for probing were from the **test split only** (model was fine-tuned on train split, embeddings extracted on held-out test set). This was confirmed during the meeting and should be stated explicitly in the methods section.

### 5.3 Phase 3 — Signal Decay and Redundancy (Ablation Test)

**File:** `experiments/understanding_text_embeddings/reports/phase_3_summary.HTML`

Content:

- After Phase 2, a priority-ranked list of the 768 (or 48) embedding dimensions was obtained from the trained linear probe's feature weights.
- An **ablation / signal decay** test was then run: iteratively remove the top-5 most important dimensions, retrain the probe, record accuracy, repeat.
- Results by model type:

  | Model type | Behaviour |
  |---|---|
  | Qwen fine-tuned | Sharp exponential drop; most context packed into top ≈ 5–10 dimensions |
  | Pre-trained (BGE, MPNet) | Linear / gradual decay; emotion context distributed broadly across all 768 dims |
  | Brain fMRI | Slow, shallow decay; content spread across all 48 dims; only ≈ 23 % drop after removing top few |

- Threshold: accuracy at or below ≈ 0.20 (5 classes) / ≈ 0.15 (6 classes) = chance; annotated as dotted lines on the plot.
- **Interpretation:** fine-tuning compacts emotional context into a small number of dominant directions (≈ 5–10 out of 768); pre-trained models distribute it broadly; the brain similarly distributes it, which is consistent with the distributed-coding interpretation from neuroscience.
- The BGE DSAIRI line is noisier (known noisy dataset) — the team agreed to leave it in and note the data quality in the discussion rather than remove it.

### 5.4 Global Behaviour Comparison (Brain vs. LLM density decay)

**File:** `video_understanding/human_brain_emotion_exports/global_behavior_comparison/comparison_results.txt`

Content:

- Density-decay pattern (number of data points per radial-distance band from centroid) was computed for both LLM embedding space and brain fMRI space and the two curves were compared.
- The file contains: **Pearson correlation coefficient**, **P-values**, **permutation test results**, and **bootstrap test results**.
- The density-decay shapes are visually similar between LLM and brain (rise then fall), but the brain curve is less smooth due to the smaller dataset.
- Cross-system correlation and p-value for the ambiguity gradient are also recorded here.

### 5.5 Brain Dataset Notebook

**File:** `video_understanding/human_brain_emotion_exports/untitled_1_2.HTML`

- Copy of the analysis notebook.
- **Line 8** contains the OpenNeuro download call:

  ```python
  openneuro.py download dataset DS005700 target_directory
  ```

- This line provides the citable source identifier for the dataset in the paper.

### 5.6 Aimee's Results PDF

**File (not yet in repo):** `emotion_geometry_and_score_4_square_explorer_square_report_score_v2.pdf`
*(Filename approximated from dictation; exact filename to be confirmed by Aimee.)*

- Created: **27 April 2026**
- Header: **"Emotion Geometry Across Brain and LLM Systems"**
- Currently held on Aimee's laptop and shared with Pritish via WhatsApp / AirDrop.
- Contains: brain decoding validation; ambiguity gradient cross-system comparison; redundancy / signal decay from a brain-data perspective; references.
- Results 1–5 in that document map to the statistical tests described in §3 and §4.

**Action — Aimee / Pritish:** Aimee to export the PDF and source notebooks / CSVs and send to Pritish; Pritish to push all files to the GitHub repository in a suitable location (suggest `experiments/brain_fmri/` or alongside the `understanding_text_embeddings` folder).

---

## 6. Statistical Treatment — Application to Previous LLM Results

A critical discussion point was that the statistical rigour Aimee applied to the brain fMRI data **must also be applied to the previous LLM text embedding experiments** before submission.

Aimee summarised the requirement:

> *"We need statistical significance built in. Otherwise you could find something amazing, but if reviewers can't see that this is robust and generalises — because you've done permutation tests, shuffle tests, etc. — they'll throw it out."*

The specific gap identified: previous LLM experiments relied on visual inspection of plots and basic metrics; they lack formal statistical testing of cluster validity, separability, and the ambiguity-gradient relationship.

### Tests to be applied (or confirmed) across previous LLM experiments:

1. **Silhouette score** — already added in Phase 1 redo (§5.1); confirm it covers all model configurations.
2. **LOSO / cross-validated decoding accuracy** — done for brain (§4); should be added for each LLM configuration.
3. **Bootstrap confidence intervals** on decoding accuracy — as per brain analysis.
4. **Permutation test** on decoding accuracy — as per brain analysis.
5. **Ambiguity gradient correlation** (radial distance vs. classification uncertainty) with **permutation p-value** — already computed for brain-vs-LLM cross-system comparison; should also be reported for each individual LLM configuration.
6. **Pearson / Spearman correlation** between density-decay curves across configurations.

**The exact set of tests to apply, and any adjustments for the LLM data characteristics, is subject to Aimee's advice.**

**Action — Aimee:** Advise Pritish and Josh on exactly which statistical tests from the brain analysis pipeline should be ported to the LLM text experiments and whether any modifications are needed given the larger / different data structure.

**Action — Pritish / Josh:** Once advised by Aimee, implement the statistical tests on all prior LLM text embedding results and update the Phase 1–3 HTML reports and any existing findings reports accordingly.

---

## 7. Additional Discussion Points

### 7.1 Embedding Normalisation Justification

- **Rationale for L2 / Z-score normalisation:** LLM embeddings hold context for many things beyond emotions; their magnitude does not encode emotion intensity alone. Therefore normalising removes a misleading signal.
- Same logic applies to the brain data (raw activation values are not intrinsically scaled).
- This reasoning should be explicitly stated in the paper as a methodological justification.

### 7.2 Loss of Spatial Information in Brain Embeddings

- When the 48 ROI activation values are treated as a flat embedding vector, the **spatial adjacency** of brain regions is lost.
- In LLMs, positional encodings preserve sequence order; no analogous mechanism exists here.
- The team noted this as a **limitation** and **future-work item**: incorporating a form of positional or graph-based encoding that preserves known brain region adjacency could improve the brain embedding representation.

### 7.3 Image Dataset (120 K)

- The sentiment image dataset with 120 K samples showed very noisy embeddings with poor cluster separation.
- Proposed treatment: retain in paper as an **appendix result** showing that the approach was attempted on a noisier domain — providing evidence of scope and honesty rather than hiding a negative result.

### 7.4 Feature Importance Weights (Future Work)

- The linear probe trained in Phase 2 contains weights for each of the 768 (or 48) dimensions.
- These weights could be unpacked (the model is stored as a `.npy` / binary file) to produce a ranked importance table showing exactly which embedding dimensions drive emotion prediction.
- This would complement the signal-decay ablation curve and provide a richer figure for the paper.
- Flagged as **future / follow-up work**.

### 7.5 1D Projection — Extension

- The remarkable result that 1D projection retains significant emotion-predictive accuracy (§5.2) should be extended to 5, 10, 15, 20 dimensions to characterise the full decay curve.
- This was also identified as near-term experimental work rather than the current critical path.

---

## 8. Action Points Summary

| # | Owner | Action |
|---|---|---|
| 1 | Aimee | Export CSV files and all brain analysis source files; send to Pritish for GitHub push |
| 2 | Aimee / Pritish | Confirm exact filename of results PDF and push to GitHub |
| 3 | Aimee | Compile and send full neuroscience reference list (Haxby et al. and others) |
| 4 | Aimee | Advise team on which statistical tests from the brain pipeline to apply to previous LLM results, and any adjustments needed |
| 5 | Pritish | Push all `understanding_text_embeddings` experiment HTML reports and supporting files to GitHub |
| 6 | Pritish / Josh | Apply agreed statistical tests (LOSO / bootstrap / permutation) to all prior LLM text embedding experiments |
| 7 | Pritish / Josh | Quantify previous multi-dataset visualisation results with silhouette scores and proper numerical metrics |
| 8 | Aimee / Daniel | Add a sentence citing typical fMRI study sizes to justify the 40-subject dataset size |
| 9 | Daniel / Pritish | Ensure the methods section records the exact embedding extraction point for each model |
| 10 | Team | Decide final placement of the 120 K image dataset results (main body vs. appendix) |

---

## 9. Key File and Data Reference Summary

| Description | Path / Identifier |
|---|---|
| Brain dataset source | OpenNeuro portal, dataset ID `DS005700` |
| Dataset full name | Neural MO — fMRI Dataset for Emotion Recognition |
| Download reference (notebook) | `video_understanding/human_brain_emotion_exports/untitled_1_2.HTML`, line 8 |
| Cross-system comparison stats | `video_understanding/human_brain_emotion_exports/global_behavior_comparison/comparison_results.txt` |
| Phase 1 global geometry report | `experiments/understanding_text_embeddings/reports/phase_one_summary_of_all.HTML` |
| Phase 2 linear probing report | `experiments/understanding_text_embeddings/reports/phase_2_1_score_summary.HTML` |
| Phase 3 signal decay report | `experiments/understanding_text_embeddings/reports/phase_3_summary.HTML` |
| Aimee's results PDF | `emotion_geometry_and_score_4_square_explorer_square_report_score_v2.pdf` *(on Aimee's laptop; to be pushed)* |
