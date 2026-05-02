# Meeting Minutes — LLM Embedding Geometry: Sentiment Experiments & NeurIPS Paper

**Date:** 11 April 2026, 18:35  
**Duration:** 1h 59m  
**Attendees:** Daniel Sikar, Pritish Ranjan (PG-Verma)  
**Transcript:** `2026-04-11_transcript_llm-embedding-geometry_sentiment-experiments_neurips.docx`  
**NeurIPS submission deadline:** 4 May 2026

---

## Recap: Work Completed Since Last Meeting

Pritish presented the latest visualisations and findings from the `dair-ai/emotion` multiclass experiment. The following was reviewed:

### Density Analysis — Per-Class Scatter Plots and Density Decay Curves

- Embeddings are 768-dimensional (BAAI/bge-base-en-v1.5, raw mean pooling).
- PCA reduces to 2D for visualisation only; all metrics computed in native 768D.
- **Counterintuitive finding confirmed:** density does NOT peak at the centroid. It peaks at a "belt" at approximately 9–9.5 units Euclidean distance from the centroid, then decays. This pattern holds for every emotion class.
- Across all classes, the peak density falls in the range 8.5–10.5 units from the centroid.
- Example: for joy and anger, peak density ≈ 350 points per unit volume at distance ≈ 9–9.5; the 10%-of-peak threshold (density ≈ 35) falls at approximately distance 10.
- The centroid, therefore, represents not the location of "most intense anger" but the geometric centre of all subtypes of a class (e.g. anger-from-love, anger-from-fear, etc.). The belt around it holds the highest-density region.
- A 10%-threshold dotted line was drawn as a reference for potential outlier identification; this is exploratory and not yet a firm conclusion.

### Inter-Class Overlap Analysis

- Bar chart "Anger vs Fear Bin Overlap Counts" (and analogues for other pairs) shows how many data points from one class fall within a given radius of the other class's centroid.
- **Key finding:** overlap between classes peaks in the region just *after* the density peak — i.e. in the low-density tail, not the high-density belt.
- This means inter-class confusion concentrates in sparse, boundary-region embeddings.

### Balanced Dataset

- The original `dair-ai/emotion` dataset is class-imbalanced (anger and joy have many more examples than surprise).
- Balanced by random undersampling to the size of the smallest class (surprise). Same strategy was already applied to the pairwise binary plots.
- Implication for future training: undersampling removes data; an alternative of data augmentation or oversampling should be discussed.

### All-Classes Cluster Plot

- A single scatter plot was reviewed showing all six emotion classes simultaneously (PCA 2D, balanced dataset).
- Observations confirmed:
  - Joy and love centroids are close; both are far from anger, sadness, and fear.
  - Anger, sadness, and fear form a compact negative cluster.
  - Surprise lies geometrically between the two groups.
- Interpretation of surprise: *surprise is a reaction rather than an emotion*, which explains its ambiguous embedding position.
- Caveat: the PCA projection may distort inter-centroid distances. The 768D Euclidean distances have not yet been measured and may tell a different story.

### Visualisation Gap Noted

- Sophia Mirvoda's data-visualisation workshop was referenced: stacked or grouped bar charts can show total overlap count broken down by contributing class. This would improve the overlap bar charts.

---

## Discussions

### On the Nature of the Centroid

The centroid is computed as the mean embedding of all examples labelled as a given class. Because sentences labelled "anger" may simultaneously carry context of fear, love, or sadness, the centroid averages across all these subtypes. The belt, not the centroid, is where the most densely packed and contextually "pure" instances lie.

The embedding space preserves contextual overlap between classes that the original one-hot labels discard. This is one of the paper's core claims.

### On Dimensionality Reduction

PCA is used only to visualise. Centroid positions, density calculations, and inter-class distances are all computed in native 768D space. Conclusions drawn from the 2D plots are treated as hypotheses to be confirmed numerically in 768D.

### On Magnitude Preservation

A core methodological principle was restated: in all stages of analysis, raw quantities that preserve both magnitude and direction should be preferred over normalised or softmax-scaled quantities.

- Raw embeddings were previously shown to outperform L2-normalised embeddings for geometry-based analysis.
- The same principle extends to the planned fine-tuned classifier experiment: **logits** (the final pre-softmax layer outputs) should be recorded and analysed rather than softmax probabilities. Logits preserve the absolute score magnitude; softmax collapses it into a relative distribution. Inter-class statements such as "anger scores higher in absolute terms than love" can only be made from logits.

### On Evaluation Metrics for the Paper

BLEU (used in "Attention Is All You Need" for translation) was cited as a domain-appropriate, community-standard metric. The paper will need to justify its evaluation approach explicitly. Anthropic reportedly emphasises evaluation rigour in interviews. Current approach (silhouette, Davies-Bouldin, centroid distances, confusion matrices) is geometry-first; a community-standard classification metric should complement this. Deep-research task assigned to clarify what evaluation metrics are standard in embedding geometry papers.

### On Bengio (2017) Label-Randomisation Idea

Referenced: Sammy Bengio et al. (2017) showed that over-parameterised networks memorise random labels to near-100% accuracy, demonstrating memorisation over generalisation. The analogous experiment for embedding geometry: assign random labels to a text dataset and ask whether the embedding positions shift or stay the same. The hypothesis is that embeddings reflect semantic content regardless of label; fine-tuning with corrupted labels would be needed to observe geometry distortion. Agreed this is a future/separate paper, not in scope for NeurIPS 4 May deadline.

### On the Paper Title

Working title inspired by Pritish's earlier paper "Knowing What the Neural Network Doesn't Know": the new paper could be framed as "Knowing Only What the Neural Network Knew" — the embedding space reveals information the network has encoded but which remains opaque in the final label output.

### On HPC and Timeline

- All text experiments should be complete by end of April to leave ~3 days for writing.
- Image experiments should mirror the text pipeline exactly; segmentation can be treated as pre-processing and is not expected to be slow.
- First image experiment: without segmentation, as a baseline; then with segmentation.
- HPC is needed for the fine-tuned classifier experiments (larger models, full-dataset training).
- Plan: begin HPC setup as soon as possible.
- Overleaf paper template: Daniel to set up and share.

---

## Action Points

### Pritish

| # | Action | Notes | Deadline |
|---|--------|-------|----------|
| P1 | Compute pairwise Euclidean inter-centroid distance matrix in native 768D for all six emotion classes; present as a table | Confirm whether PCA-based proximity impressions hold in 768D | 14 Apr |
| P2 | Cross-model validation: repeat the density-decay and overlap experiments with at least two alternative embedding models; verify that the belt pattern and overlap region are model-agnostic | Scale of distances may differ; the pattern should not | 18 Apr |
| P3 | Improve overlap bar charts: replace or supplement current single-class bars with stacked/grouped bars showing per-class components and totals (Sophia-workshop style) | Improves legibility for paper | 16 Apr |
| P4 | Multi-panel scatter plots: produce an 8–10 panel figure with one 2-class pairwise scatter per panel + one final panel showing all centroids with high-transparency point cloud | Required for paper figures section | 18 Apr |
| P5 | Fine-tuned classifier: implement the architecture (pre-trained transformer → 768D FC → 5-class logit head), train on `dair-ai/emotion` train split, evaluate on validation/test split; record 768D embeddings and logits (not softmax) for every example | See experiments plan document | 25 Apr |
| P6 | Logit-geometry correlation: for each validation example, check whether the rank order of logit scores matches the rank order of Euclidean distances from class centroids | Core hypothesis test for the fine-tuned experiment | 27 Apr |
| P7 | Image pipeline: start single-image embedding extraction (no segmentation first), mirror text metrics and visualisations | Required for cross-modality claim | 25 Apr |

### Daniel

| # | Action | Notes | Deadline |
|---|--------|-------|----------|
| D1 | Set up Overleaf paper template; share link with Pritish | Sections: intro, related work, methodology, experiments, results, discussion, conclusion, future work | 13 Apr |
| D2 | HPC access: coordinate getting experiments running on HPC; prioritise the fine-tuned classifier jobs | Larger models (1.6–2B params) need HPC | 14 Apr |
| D3 | Research evaluation metrics standard in embedding-geometry and sentence-representation papers; summarise candidates beyond silhouette and Davies-Bouldin | Reviewers will ask about this | 16 Apr |
| D4 | Draft mathematical formulation section: centroid, belt radius, overlap volume, cross-modality invariance claim | Build on 7 Apr minutes draft | 18 Apr |
| D5 | Draft introduction and related-work sections | Incorporate existing literature survey | 23 Apr |
| D6 | Integrate image results and cross-modality argument | Depends on P7 | 27 Apr |
| D7 | Full paper draft for joint review | | 29 Apr |
| D8 | Final revisions and submission | | 3 May |

---

## Submission Timeline

```
Apr 13  Overleaf template shared (D1)
Apr 14  Inter-centroid distance table (P1) + HPC setup (D2)
Apr 16  Improved bar charts (P3) + evaluation metrics research (D3)
Apr 18  Cross-model validation (P2) + multi-panel scatter plots (P4) + maths formulation draft (D4)
Apr 23  Related work + intro drafted (D5)
Apr 25  Fine-tuned classifier trained + embeddings/logits recorded (P5) + image pipeline running (P7)
Apr 27  Logit-geometry correlation results (P6) + cross-modality section drafted (D6)
Apr 29  Full draft complete (D7)
May 3   Final submission (D8)
May 4   NeurIPS deadline
```

---

## Future Work (Out of Scope for NeurIPS)

- **Label-randomisation experiment** (Bengio 2017 analogue): train with corrupted labels and observe whether embedding geometry is disturbed or preserved; expected to become a separate paper.
- **Video modality**: extend the cross-modality claim beyond text and single images.
- **Data augmentation / oversampling** for the surprise class instead of undersampling all other classes.
