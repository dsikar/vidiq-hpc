# Qwen Fine-Tuned Run Phase 3 Implementation Report

## Status

Phase 3 has been implemented locally and run successfully against the tracked Qwen fine-tuned source run:

- source run: `experiments/text_model/runs/tmqb0010_17763/`
- bridge run: `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/`

This report documents what was added, what outputs were generated, and what conclusions are justified at this stage.

## Implemented Files

Added:

- `experiments/text/multiclass/dair-ai-emotion/src/analyze_qwen_logit_geometry.py`
- `experiments/text/multiclass/dair-ai-emotion/reports/qwen-finetune-logit-geometry-findings.md`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e/logit-geometry-summary.json`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e/per-class-logit-distance-correlations.json`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e/nearest-centroid-vs-prediction.json`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e/distance-rank-agreement.json`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/logit-geometry/true-class-logit-vs-distance.png`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/logit-geometry/predicted-class-logit-vs-distance.png`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/logit-geometry/distance-margin-vs-logit-margin.png`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/logit-geometry/rank-correlation-by-class.png`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/logit-geometry/nearest-centroid-confusion-heatmap.png`

Updated:

- `experiments/text/multiclass/dair-ai-emotion/README.md`
- `experiments/text/multiclass/dair-ai-emotion/reports/qwen-finetune-first-findings.md`
- `experiments/text/multiclass/dair-ai-emotion/reports/multiclass-emotion-dataset-findings.md`

## What The Script Does

The Phase 3 script:

- reads the dataset-level bridge run rather than hardcoding source artifact locations
- resolves source artifact paths relative to the bridge run directory
- reconstructs held-out evaluation labels using the same `StratifiedShuffleSplit` configuration as training
- validates row counts, class counts, centroid dimensionality, and logit shape
- computes both Euclidean and cosine centroid distances in native embedding space
- compares raw pre-softmax logits against centroid-distance structure
- writes dataset-level metrics and plots without duplicating raw arrays

The implementation is parameterized by bridge run, so future fine-tuned runs can use the same flow.

## Main Results

From `tmqb0010_17763`:

- evaluation accuracy: `0.9804`
- evaluation size: `4038`
- mean per-example rank agreement:
  - Euclidean: `0.7189`
  - cosine: `0.7159`
- predicted class equals nearest centroid:
  - Euclidean: `0.9941`
  - cosine: `0.9926`
- true label is nearest centroid:
  - Euclidean: `0.9789`
  - cosine: `0.9792`
- global true-class logit vs distance correlation:
  - Euclidean: `-0.5802`
  - cosine: `-0.5307`
- global outlier rate under the primary metric: `0.0097`

These results support a conservative within-run claim:

- the Qwen fine-tuned model's logits are strongly aligned with its exported embedding geometry on the held-out balanced split

These results do **not** support:

- a claim that Qwen geometry is better than the earlier BGE stage
- a direct cross-stage comparison that silently equates `joy` and `happiness`

## Validation Notes

The reconstruction checks passed for this run:

- `eval_embeddings.npy` rows matched `eval_logits.npy` rows
- reconstructed evaluation labels matched artifact row count
- per-class held-out counts matched the expected balanced split:
  - `673` examples per class
- `eval_logits.npy` had `6` columns matching `num_labels`
- centroid dimensionality matched embedding dimensionality

The output JSON uses repo-relative provenance paths, not machine-specific absolute paths.

## Reusability

Future results should be straightforward to incorporate.

The intended pattern is:

1. keep the raw training run in `experiments/text_model/runs/<run_id>/`
2. add a lightweight bridge run in `experiments/text/multiclass/dair-ai-emotion/runs/`
3. rerun:
   - `src/plot_qwen_finetune_run.py`
   - `src/analyze_qwen_logit_geometry.py`
4. update the dataset-level findings reports if the new run is important enough to surface

This means new fine-tuned runs can be added without changing the repo structure again.

## Verification

Executed successfully:

- `python3 -m py_compile experiments/text/multiclass/dair-ai-emotion/src/analyze_qwen_logit_geometry.py`
- `.venv/bin/python experiments/text/multiclass/dair-ai-emotion/src/analyze_qwen_logit_geometry.py`

## Remaining Boundaries

Still out of scope after Phase 3:

- new training runs
- HPC workflow changes
- direct BGE vs Qwen superiority claims
- any cross-stage quantitative comparison that does not explicitly resolve the `joy` / `happiness` mismatch

## Recommendation

This implementation is structurally complete and scientifically narrow enough to review.

The next sensible step is an audit of the implementation artifacts and findings note before commit/push.
