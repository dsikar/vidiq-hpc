# Text Embedding Model and Validation Guide

## Purpose

This document records the current recommendation for:

- which text embedding models to use for `vidiq-hpc`
- how to construct embeddings from text datasets
- how to validate the resulting embedding geometry
- which caveats to keep in mind when interpreting results

It is intended to be reused as prompt context for later implementation, analysis, and reporting work.

## Scope

This recommendation is based on:

- the existing text-dataset reports in this repository
- a full pass over the local literature set in `vidiq/lit-survey/gemini`
- the extracted title / abstract / conclusion report in `vidiq/reports/title-abstract-conclusion.md`

The immediate target use case is sentence-level embeddings for clean text datasets such as:

- SST-2
- Subjectivity
- GYAFC
- other short-text datasets with simple semantic contrasts

## Executive Decision

For the first experiment pipeline, use a contrastively trained sentence encoder as the primary embedding model.

Do not use raw hidden states from a generative decoder LLM as the main baseline for the first pass.

The first production choice should be:

- `BAAI/bge-base-en-v1.5`

Strong alternative:

- `sentence-transformers/all-mpnet-base-v2`

Optional comparison family for a later second-stage baseline:

- a SimCSE-style sentence model
- an open LLM hidden-state baseline such as Gemma or Llama with the same pooling recipe

## Why This Is the Current Recommendation

The literature sweep supports the following conclusions:

- Vanilla BERT-style sentence vectors are not reliably good for similarity and clustering unless the model is trained for sentence embedding tasks.
- Contrastive or siamese objectives consistently improve sentence-level geometry for similarity, clustering, and retrieval.
- Embedding spaces often show anisotropy, dominant common directions, or other geometry pathologies.
- A single metric such as cosine similarity is not sufficient for validation.
- Performance on STS-style tasks alone does not guarantee strong clustering or dataset geometry behavior.

In practice, this means the first pipeline should favor:

- sentence encoders over raw autoregressive LLM hidden states
- explicit pooling and normalization rules
- explicit geometry diagnostics
- multiple validation metrics rather than one headline score

## Recommended Model Stack

### Tier 1: Primary Model

Use:

- `BAAI/bge-base-en-v1.5`

Reason:

- strong sentence embedding baseline
- practical model size
- good off-the-shelf behavior for semantic similarity and clustering
- simpler and safer for first-pass geometry analysis than raw decoder LLM states

### Tier 2: Practical Alternate

Use:

- `sentence-transformers/all-mpnet-base-v2`

Reason:

- mature sentence-transformers integration
- strong general-purpose sentence embedding behavior
- useful as a robustness comparison against BGE

### Tier 3: Later Comparison Models

Add only after the first pipeline is working:

- a SimCSE-family encoder
- one decoder LLM hidden-state baseline

Reason:

- useful for comparison
- not the safest first choice for clean geometry experiments
- adds ambiguity around pooling, layer selection, and similarity interpretation

## Embedding Construction

### Unit of Embedding

Create one embedding per text example.

For the current datasets, the canonical unit is:

- one sentence
- one short review snippet
- one short formality example

Do not start with token-level embeddings as the primary experimental object.

### Pooling Rule

Use masked mean pooling over the final hidden states.

Default rule:

1. tokenize the text
2. run the encoder
3. take the final hidden state for all non-padding tokens
4. compute the attention-mask-weighted mean

Do not default to:

- raw `CLS` only
- first-token pooling unless the model explicitly expects it
- unpooled token sets as the primary representation

### Normalization Rule

Always save at least two versions of each embedding:

1. pooled embedding
2. L2-normalized pooled embedding

Also save a third version for geometry analysis:

3. mean-centered, then L2-normalized embedding

Reason:

- centering may materially reduce common-direction drift
- normalized vectors are easier to compare with cosine and Euclidean-on-unit-sphere analyses
- storing all three avoids recomputation and makes ablation easier

### Suggested Saved Outputs

For each dataset split, save:

- `embeddings_raw.npy`
- `embeddings_l2.npy`
- `embeddings_centered_l2.npy`
- `labels.npy`
- `texts.jsonl`
- metadata describing:
  - model name
  - tokenizer
  - pooling rule
  - normalization steps
  - dataset name
  - split name
  - timestamp

## Recommended Experimental Workflow

### Phase 1: Dataset Preparation

- load the dataset
- remove empty rows
- keep original text and label
- inspect class balance
- record length statistics
- sample a manageable but representative subset if the dataset is large

### Phase 2: Embedding Extraction

- extract sentence embeddings with the primary encoder
- save raw, normalized, and centered-normalized embeddings
- log batch size, max sequence length, truncation policy, and device

### Phase 3: Geometry Validation

Run all validation on both:

- L2-normalized embeddings
- mean-centered + L2-normalized embeddings

This comparison is important because geometry improvements may come from post-processing rather than the encoder alone.

### Phase 4: Comparative Baselines

After the first model is stable:

- repeat on `all-mpnet-base-v2`
- compare outcomes
- only then add a decoder-LLM hidden-state baseline if still needed

## Core Validation Metrics

### Cluster Quality

Use:

- silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score

Purpose:

- test whether labeled groups form compact and separated clusters

Notes:

- compute on the same representation variants
- report sensitivity to the number of clusters if using unsupervised methods

### Distance Structure

Use:

- centroid cosine distance
- centroid Euclidean distance
- average intra-class distance
- average inter-class distance
- ratio of intra-class to inter-class distance

Purpose:

- measure whether classes are geometrically separated
- estimate whether a simple semantic axis is present

### Linear Separability

Use:

- logistic regression
- linear SVM

Report:

- train / test accuracy
- macro F1
- ROC-AUC when appropriate

Purpose:

- test whether the geometry is linearly usable

Interpretation:

- high linear probe performance is stronger evidence than a pretty 2D plot

### Neighborhood Quality

Use:

- k-nearest-neighbor classification accuracy
- class purity among nearest neighbors

Purpose:

- test whether local geometry preserves label semantics

### Dimensionality and Spectrum

Use:

- PCA explained variance ratios
- top principal component dominance
- effective rank if convenient

Purpose:

- measure whether the space is dominated by a few directions
- support interpretation of anisotropy and shared-direction drift

### Isotropy / Uniformity Diagnostics

Use:

- IsoScore if implemented
- average pairwise cosine as a rough diagnostic only
- variance spectrum diagnostics

Important:

- do not rely on brittle isotropy claims from a single simple metric
- treat isotropy as a diagnostic, not a success metric by itself

### Representation Similarity Across Models

If multiple models are compared, use:

- CKA
- RSA-style distance-matrix comparisons

Purpose:

- compare whether different encoders produce similar geometry on the same dataset

## Distance Metrics to Use

### Primary

Use first:

- cosine distance
- Euclidean distance on L2-normalized embeddings

Reason:

- standard
- easy to interpret
- enough for the first experimental pass

### Secondary

Consider later:

- Mahalanobis distance
- Word Mover's Distance style comparisons
- optimal transport style distances
- DDR-style similarity

Reason:

- potentially informative
- more expensive or more specialized
- not necessary for the first implementation

### Important Caution About Cosine

Do not assume cosine similarity is universally meaningful.

The literature suggests cosine can be influenced by representation scaling, regularization, and geometry artifacts. It should remain a primary baseline metric, but not the only one.

## Visualization Guidance

Use:

- PCA first
- UMAP second
- t-SNE only if needed

Always:

- color by label
- show class centroids
- report projection quality caveats
- avoid making strong claims from 2D projections alone

Recommended plots:

- PCA scatter with centroids
- UMAP scatter with centroids
- histogram of projection onto class-difference vector
- pairwise distance histograms for intra-class vs inter-class pairs

## Failure Modes to Watch For

### Lexical Shortcut Separation

Risk:

- clusters may separate because of superficial lexical markers instead of semantics

Mitigation:

- inspect nearest neighbors
- examine misclassified examples
- compare across multiple datasets with different confounds

### Dominant Common Direction

Risk:

- the space may be driven by one or a few global directions

Mitigation:

- compare raw vs centered embeddings
- inspect PCA spectrum

### Over-reading 2D Projections

Risk:

- PCA / UMAP / t-SNE can create misleading visual separation

Mitigation:

- pair plots with numerical metrics
- never use plots as the sole evidence

### Model-Specific Metric Fragility

Risk:

- one encoder looks good under one metric and weak under another

Mitigation:

- use a metric panel, not a single score

### Dataset Confounds

Risk:

- subjectivity, formality, and sentiment datasets may include topic or style leakage

Mitigation:

- document known confounds
- compare multiple datasets

## Recommended First Experiment Matrix

### Datasets

Start with:

- SST-2
- Subjectivity
- GYAFC

Optional later:

- emotion or politeness datasets

### Models

Run:

1. `BAAI/bge-base-en-v1.5`
2. `sentence-transformers/all-mpnet-base-v2`

Optional later:

3. SimCSE-family model
4. one decoder-LLM hidden-state baseline

### Representation Variants

For each model, evaluate:

1. raw pooled
2. L2-normalized
3. mean-centered + L2-normalized

### Metrics Panel

For each run, report:

- silhouette score
- Davies-Bouldin index
- centroid cosine distance
- centroid Euclidean distance
- intra-class / inter-class distance ratio
- linear probe accuracy and macro F1
- kNN accuracy
- PCA explained variance for top components
- optional IsoScore

## Default Prompt Context Summary

If this document is used in later prompts, the default assumptions should be:

- use sentence-level embeddings
- start with `BAAI/bge-base-en-v1.5`
- use masked mean pooling
- save raw, normalized, and centered-normalized embeddings
- validate with both cluster metrics and supervised separability metrics
- do not trust cosine alone
- do not trust 2D plots alone
- treat raw decoder-LLM hidden states as a later comparison, not the primary first model

## Current Bottom Line

The safest first implementation for `vidiq-hpc` is:

- a contrastive sentence encoder
- sentence-level mean-pooled embeddings
- explicit normalization and centering variants
- a validation stack that combines cluster, distance, probe, and spectrum metrics

This is the recommended baseline against which any later LLM hidden-state or more exotic geometry method should be compared.
