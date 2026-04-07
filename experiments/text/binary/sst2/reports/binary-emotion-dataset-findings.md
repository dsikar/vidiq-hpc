# SST-2 Binary Emotion Dataset Findings

## Experiment Summary

This document records the completed findings for the first binary text experiment on SST-2.

Experiment scope:

- dataset: `glue/sst2`
- task: binary sentiment geometry analysis
- final selected model: `BAAI/bge-base-en-v1.5`
- compared embedding variants:
  - raw mean pooled
  - L2-normalized mean pooled
  - mean-centered plus L2-normalized mean pooled

Evaluation policy:

- all quantitative metrics were computed on the original `768`-dimensional embeddings
- PCA and nonlinear projections were used only for visualization

## Dataset Snapshot

- train embeddings shape: `(67349, 768)`
- validation embeddings shape: `(872, 768)`
- dtype: `float32`
- `max_length`: `64`
- batch size: `32`
- device: `cpu`
- validation labels:
  - negative: `428`
  - positive: `444`

Tokenizer metadata from the selected BGE run:

- average tokenized length: `40.02`
- median tokenized length: `40`
- truncation count: `160`
- truncation rate: `0.0024`

Interpretation:

- truncation was negligible under the current configuration
- there is no immediate need to increase `max_length` for SST-2

## Model Selection Finding

The initial model-selection stage compared:

- `BAAI/bge-base-en-v1.5`
- `sentence-transformers/all-mpnet-base-v2`

Decision:

- choose `BAAI/bge-base-en-v1.5`

Reason:

- `all-mpnet-base-v2` was slightly stronger on linear probe performance
- `bge-base-en-v1.5` was stronger on neighborhood quality and clustering-oriented geometry
- the project objective is embedding-space analysis, not only probe accuracy

## Embedding Variant Results

### Raw Mean Pooling

- logistic regression accuracy: `0.8876`
- logistic regression macro F1: `0.8874`
- kNN@1 accuracy: `0.8417`
- kNN@5 accuracy: `0.8670`
- silhouette: `0.0468`
- centroid cosine distance: `0.0771`
- intra / inter distance ratio: `0.9099`

Confusion matrix:

- negative correctly classified: `369 / 428`
- positive correctly classified: `405 / 444`

### L2 Mean Pooling

- logistic regression accuracy: `0.8830`
- logistic regression macro F1: `0.8829`
- kNN@1 accuracy: `0.8475`
- kNN@5 accuracy: `0.8658`
- silhouette: `0.0468`
- centroid cosine distance: `0.0768`
- intra / inter distance ratio: `0.9100`

Confusion matrix:

- negative correctly classified: `369 / 428`
- positive correctly classified: `401 / 444`

### Centered + L2 Mean Pooling

- logistic regression accuracy: `0.8830`
- logistic regression macro F1: `0.8828`
- kNN@1 accuracy: `0.8257`
- kNN@5 accuracy: `0.8635`
- silhouette: `0.0468`
- centroid cosine distance: `1.9974`
- intra / inter distance ratio: `0.9102`

Confusion matrix:

- negative correctly classified: `364 / 428`
- positive correctly classified: `406 / 444`

## Main Finding

The best current SST-2 embedding representation is:

- `BAAI/bge-base-en-v1.5` with raw mean pooling

Reason:

- best logistic probe result
- best `k = 5` neighborhood result
- best overall balance across probe quality, neighborhood quality, and geometry

Important nuance:

- the three variants are close
- raw mean pooling is better, but not by a huge margin
- this means post-processing is not the main source of performance on this dataset

## Geometry Interpretation

What the metrics suggest:

- SST-2 has a clear polarity structure in the embedding space
- the structure looks more like a strong sentiment axis than two perfectly separated clusters
- the silhouette score is low in absolute terms, which means the classes overlap substantially even though they are predictively separable
- this matches the dataset itself, where many review snippets are short, fragmentary, or sentiment-ambiguous

What the visualizations suggest:

- PCA shows strong separation along the first principal direction
- sentiment-axis histograms show clear mean separation between negative and positive examples
- the class distributions still overlap noticeably
- the nonlinear visualization was generated with `t-SNE` in this environment as a fallback diagnostic view

Visual artifacts:

- [raw PCA](/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/text/binary/sst2/artifacts/plots/bge-variant-visuals/raw/pca-2d.png)
- [raw sentiment axis](/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/text/binary/sst2/artifacts/plots/bge-variant-visuals/raw/sentiment-axis-hist.png)
- [projection summary](/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/text/binary/sst2/artifacts/plots/bge-variant-visuals/projection-summary.json)

## Final Decision

Use this as the default binary SST-2 setup:

- model: `BAAI/bge-base-en-v1.5`
- embedding unit: one SST-2 row
- pooling: masked mean pooling
- default representation: raw mean pooled embedding
- sequence length: `64`
- metrics: all on original `768D` vectors
- dimensionality reduction: visualization only

## What Remains

The binary experiment is complete enough to move forward.

The next meaningful additions are:

- `TF-IDF + logistic regression` baseline
- optional `CLS` pooling comparison
- optional multiclass pipeline work on `dair-ai/emotion`
