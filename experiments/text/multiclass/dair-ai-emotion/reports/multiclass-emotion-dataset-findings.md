# dair-ai/emotion Multiclass Findings

## Experiment Essentials

- dataset: `dair-ai/emotion`
- final model: `BAAI/bge-base-en-v1.5`
- embeddings: raw, L2-normalized, centered+L2 mean pooled vectors on the full train and validation splits
- validation labels: six emotions (sadness, joy, love, anger, fear, surprise)

## Quantitative Outcome

- raw mean pool:
  - logistic reg macro F1: `0.697`
  - kNN@5 macro F1: `0.647`
  - silhouette: `0.0038`
  - Davies-Bouldin: `7.91`
- L2 mean pool:
  - logistic reg macro F1: `0.655`
  - kNN@5 macro F1: `0.650`
- centered+L2 mean pool:
  - logistic reg macro F1: `0.676`
  - kNN@5 macro F1: `0.636`
- centroids remain near each other (same/cross distance ratio ≈ `0.953`)
- the best metric balance is still the raw embeddings, so they remain the default

## Confusion Insights

- `joy` vs `love`, and the `anger`/`fear`/`sadness` trio show the most overlap (see `metrics/confusion-matrix.json`)
- `surprise` is the weakest class, frequently confused with multiple others
- these patterns match the dataset narrative and explain the tiny silhouette scores

## Geometry Interpretation

- PCA plots show a dominant axis but no six-way clean clustering
- t-SNE fallback outlines partially overlapping blobs, which is expected when classes share affective meaning
- centroid heatmaps confirm the pairwise distances listed in the summary
- the overlaps are real (not just visualization artifacts), as the metrics were computed in the native 768D space

## Fine-Tuned Qwen Stage

- first supervised fine-tuning result now exists at `experiments/text_model/runs/tmqb0010_17763/`
- integrated dataset-level bridge run: `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/`
- model: `Qwen/Qwen3-1.7B`
- held-out balanced-CSV evaluation accuracy: `0.9804` on `4038` examples

This result shows that the supervised classifier stage is operational and produces the expected embeddings, logits, centroids, and training metrics. It does **not** yet replace the earlier BGE-stage findings as the geometry baseline.

Important caveats:

- the Qwen run is evaluated on a held-out split from the balanced CSV workflow, not on the earlier BGE validation setup
- the Qwen label schema uses `happiness`, while the earlier BGE-stage reports use `joy`
- any cross-stage geometry comparison must therefore document the label mapping explicitly rather than assuming direct equivalence

Current interpretation:

- BGE ablation stage remains the geometry-first baseline
- Qwen fine-tuned stage is the supervised follow-on stage for logit/geometry analysis
- further plotting and geometric comparison are still needed before claiming that fine-tuning improves class structure

## Decision

- keep `BAAI/bge-base-en-v1.5` raw mean pooling as the multiclass default
- reference L2 and centered-L2 variants for stress tests or data shifts
- the next additions should be:
  1. a TF-IDF + logistic regression lexical baseline
  2. optional CLS pooling or `max_length=128` check if truncation changes behaviour
