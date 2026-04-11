# Next Experiments Plan — dair-ai/emotion Phase 2

## Purpose

This document defines the next stage of experiments for the `dair-ai/emotion` multiclass geometry study, extending the completed embedding-analysis work into three new directions agreed at the 11 April 2026 meeting:

1. **Quantitative centroid geometry** — pairwise distances in native 768D space.
2. **Cross-model validation** — verify the belt-density pattern is model-agnostic.
3. **Fine-tuned classifier with logit-geometry correlation** — move from tokenizer-only embeddings to a trained classification head and relate the model's internal scores to the embedding geometry.

These experiments are needed for the NeurIPS paper (deadline 4 May 2026). All text experiments must complete by 30 April.

---

## Principle: Preserve Magnitude and Direction

In every measurement taken in these experiments, use raw quantities that retain both magnitude and direction. Do not normalise or apply softmax unless an explicit normalised variant is being studied.

Rationale: the geometry analysis depends on Euclidean distances and absolute score magnitudes. Normalisation projects points onto a hypersphere and discards the radial component; softmax collapses absolute logit scores into a relative distribution. Both operations destroy information that is load-bearing for the paper's claims.

Specifically:
- Embeddings: use raw mean-pooled vectors (established as best in the binary and multiclass ablations).
- Logits: record raw pre-softmax scores, not softmax outputs.
- Distances: use Euclidean distance in native N-dimensional space, not cosine similarity unless explicitly comparing.

---

## Experiment 1: Inter-Centroid Euclidean Distance Matrix

### Goal

Compute pairwise Euclidean distances between all six class centroids in native 768D space and present as a table.

### Motivation

The all-clusters scatter plot (PCA 2D) suggests that joy and love are close, that anger/sadness/fear cluster together, and that surprise sits between the two groups. These impressions are based on 2D projections that discard ~96% of variance. The distance matrix in 768D will either confirm or revise these impressions and will be a concrete, citable result for the paper.

### Procedure

1. Load the raw mean-pooled embeddings for the full balanced validation split.
2. Compute the per-class centroid as the mean embedding vector for each of the six classes (sadness, joy, love, anger, fear, surprise).
3. Compute the 6×6 Euclidean distance matrix across all centroid pairs.
4. Present as a symmetric table; also present as a heatmap.
5. Rank the pairs by distance (closest to furthest).

### Deliverables

- `artifacts/metrics/centroid-distance-matrix.json` — raw 6×6 matrix.
- `artifacts/plots/centroid-distance-heatmap.png` — heatmap.
- Short findings paragraph for the paper (e.g. "joy–love: X units; anger–joy: Y units; ...").

### Notes

- Do not use cosine similarity here; Euclidean distance is what the density-radius analysis is based on and the two metrics must be consistent.
- If the 768D ordering differs substantially from the PCA-2D ordering, document this explicitly; it strengthens the argument for always working in native space.

---

## Experiment 2: Cross-Model Validation

### Goal

Repeat the density-decay and inter-class overlap experiments with at least two alternative embedding models and verify that the belt-density pattern is not an artefact of `BAAI/bge-base-en-v1.5`.

### Motivation

The hypothesis is that the belt pattern (density peaking at ~9–9.5 units from the centroid, overlap peaking just beyond that) is a universal property of LLM embedding geometry, not a quirk of one model. Demonstrating it across multiple model families strengthens the paper's generality claim.

### Model Candidates

Select at least two from:

| Model | Params | Notes |
|-------|--------|-------|
| `sentence-transformers/all-mpnet-base-v2` | ~110M | Already run in model-selection stage; embeddings may already exist |
| `sentence-transformers/all-MiniLM-L6-v2` | ~22M | Smaller; useful to check whether size matters |
| `intfloat/e5-base-v2` | ~109M | Different training recipe from BGE |
| `thenlper/gte-base` | ~109M | Good multilingual coverage; different architecture family |

Use raw mean pooling for all candidates (consistent with the established default).

### Procedure

1. For each candidate model, generate raw mean-pooled embeddings for the full balanced validation split of `dair-ai/emotion`.
2. Compute:
   - Per-class centroid in the model's native embedding space.
   - Density-decay curve (density per unit volume vs. distance from centroid) for at least two class pairs (e.g. anger vs. fear; joy vs. sadness).
   - Inter-class overlap count by distance bin for the same pairs.
3. Record the distance at which density peaks and the distance at which overlap peaks for each model.
4. Compare across models: do peak-density distances all fall in a consistent range? Does the overlap-peaks-after-density-peak pattern hold?

### Deliverables

- `artifacts/metrics/cross-model-density-peaks.json` — table of peak distances per model per class.
- `artifacts/plots/cross-model-density-decay/` — one density-decay overlay plot per class pair, with one line per model.
- Findings paragraph: confirm or revise the universality claim.

### Notes

- The absolute distance scale will differ between models (different embedding norms). Compare the *pattern*, not the raw numbers.
- If one model breaks the pattern, investigate why before concluding it is an exception.

---

## Experiment 3: Fine-Tuned Classifier — Embeddings and Logits

### Goal

Train a classification head on top of a pre-trained transformer, then record the 768D intermediate embeddings and the pre-softmax logit scores for each validation/test example. Analyse whether logit magnitudes correlate with Euclidean distances from class centroids in the embedding space.

### Motivation

All experiments so far use the tokenizer's embedding output directly, without any task-specific training signal. This experiment adds a trained classification head and asks:

- Does fine-tuning make the class clusters tighter and more separated?
- Do the model's logit scores (how confident it is about each class) correlate with the geometric proximity of each embedding to the class centroids?

If the answer to both is yes, this is a strong result: the geometry visible in the raw embedding space is reinforced by training, and the model's internal confidence scores are geometrically meaningful.

### Architecture

```
Input sentence
    ↓
Tokenizer  (same tokenizer as current experiments; do not change)
    ↓
Pre-trained transformer backbone  (frozen initially; see variants below)
    ↓
768-dimensional FC layer  ← record embeddings here
    ↓
5-class logit layer (one score per class)  ← record logits here
    ↓
Softmax  (used for training loss only; do not use logit scores after this point)
```

The 768D FC layer output is the embedding to analyse geometrically. The 5-class layer output (before softmax) is the logit vector to correlate with centroid distances.

### Model Size

- Start with the same base model already in use (`BAAI/bge-base-en-v1.5`, ~600M params).
- If HPC capacity allows, also run a larger model (~1.6–2B parameters). For from-scratch training, use a small model (~10–50M params, e.g. a 6-layer transformer) to keep the training tractable on the available data size.

### Training Variants

Run in this order; stop if compute becomes limiting:

| Variant | Backbone | New layers |
|---------|----------|------------|
| A | Frozen BAAI/bge-base-en-v1.5 | 768D FC + 5-class head trained |
| B | Unfrozen BAAI/bge-base-en-v1.5 (fine-tuned end-to-end) | 768D FC + 5-class head trained |
| C | Small transformer trained from scratch (~10–50M params) | Full model |

### Data Split

- Train on the training split.
- Evaluate on the validation split (and test split if available).
- Do not evaluate on the training split for the logit-geometry analysis; the model will have memorised training examples and logit scores will trivially be high.

### What to Record

For every example in the evaluation split:

- 768D embedding vector (from the FC layer).
- 5-class logit vector (raw, before softmax).
- Ground-truth label.
- Model-predicted label (argmax of logits).

Save these as numpy arrays alongside labels for downstream analysis.

### Logit-Geometry Correlation Analysis

After collecting embeddings and logits:

1. Compute the per-class centroid from the recorded 768D embeddings.
2. For each example, compute its Euclidean distance to every class centroid (a 5-element distance vector).
3. Rank the five classes by Euclidean distance (nearest to furthest).
4. Rank the five classes by logit score (highest to lowest).
5. Compute rank correlation (Spearman) between the two orderings for each example.
6. Report:
   - Mean rank correlation across all examples.
   - Mean rank correlation broken down by ground-truth class.
   - Examples where rank ordering is correct vs. inverted.

### Expected Findings

- The class with the highest logit score should be the class whose centroid is closest to the embedding.
- If the model is well-trained and not overfitting, rank correlation should be high.
- Classes that are geometrically close (joy/love, anger/sadness/fear) will produce ambiguous logit vectors; rank correlation will be lower for these pairs.

### Deliverables

- `src/run_finetune_classifier.py` — training script for variants A, B, C.
- `src/run_logit_geometry_correlation.py` — analysis script.
- `artifacts/embeddings/finetune-*/` — recorded embeddings per variant.
- `artifacts/metrics/logit-centroid-correlation.json` — Spearman rank correlation summary.
- `artifacts/plots/logit-vs-distance/` — scatter plots of logit score vs. centroid distance per class.
- Findings paragraph for the paper.

---

## Experiment 4: Improved Overlap Visualisations

### Goal

Replace the current single-class overlap bar charts with stacked/grouped bars that show both the total overlap count and the per-contributing-class breakdown at each distance bin.

### Motivation

The current bar charts show, for a given class pair (e.g. anger vs. fear), how many anger points fall in the fear region. A stacked bar chart would additionally show, for each distance bin, how much of the total overlap comes from anger points vs. fear points vs. points from other classes. This is more informative and publication-ready.

### Format

- x-axis: distance from centroid (binned).
- y-axis: overlap count.
- Bars: stacked, one colour per contributing class.
- Total bar height = total overlap at that distance.

Produce one chart per class pair featured in the paper.

### Deliverables

- `artifacts/plots/overlap-stacked/` — one stacked bar chart per class pair.

---

## Experiment 5: Multi-Panel Pairwise Scatter Plots

### Goal

Produce a single publication-ready figure containing 8–10 panels, each showing one 2-class pairwise PCA scatter plot, plus one final panel showing all centroids with a high-transparency full-class scatter.

### Motivation

The paper needs figures that let readers see the class separation at a glance. Showing all six classes in one plot is visually crowded. A multi-panel figure with one comparison per panel, arranged on one page, is the standard format for this kind of result.

### Format

- Each panel: PCA 2D scatter, two classes only, centroids marked, density-peak arcs shown.
- Suggested pairs: joy vs. anger; joy vs. sadness; joy vs. fear; joy vs. surprise; love vs. anger; love vs. surprise; anger vs. fear; sadness vs. fear.
- Final panel: all centroids + full scatter with point alpha ≈ 0.1.
- Arrange as a 3×3 or 2×5 grid on a single figure.

### Deliverables

- `artifacts/plots/pairwise-panels/pairwise-scatter-grid.png` — the full multi-panel figure.
- Individual panel files in `artifacts/plots/pairwise-panels/individual/` if needed.

---

## Execution Order and Timeline

| # | Experiment | Owner | Deadline |
|---|------------|-------|----------|
| 1 | Inter-centroid distance matrix | Pritish | 14 Apr |
| 4 | Improved overlap bar charts | Pritish | 16 Apr |
| 5 | Multi-panel scatter plots | Pritish | 18 Apr |
| 2 | Cross-model validation | Pritish | 18 Apr |
| 3 | Fine-tuned classifier (variant A first) | Pritish + HPC | 25 Apr |
| 3 | Logit-geometry correlation analysis | Pritish | 27 Apr |

Variants B and C of the fine-tuned classifier are stretch goals; proceed only if HPC time and the April deadline allow.

---

## Notes for the Paper

- Always state which experiment (tokenizer-only vs. fine-tuned) produced each result.
- Report all geometry metrics in native 768D; PCA-based observations go in supplementary or figures only.
- Use logits, not softmax, throughout the classifier analysis.
- The cross-model validation result is the evidence for the universality claim; without it, the paper's generality is limited to one model.
