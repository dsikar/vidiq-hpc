# Qwen Fine-Tuned First Findings

## Run Summary

- source run: `experiments/text_model/runs/tmqb0010_17763/`
- integrated run bridge: `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/`
- model: `Qwen/Qwen3-1.7B`
- training data: `balanced_emotions_6classes.csv`
- training regime: balanced 6-class CSV, batch size `8`, learning rate `5e-5`, `10` epochs
- evaluation result: accuracy `0.9804` on `4038` held-out examples

## What This Run Gives Us

This is the first completed supervised fine-tuning stage for the multiclass emotion work. Unlike the earlier BGE ablation runs, this stage trains a task-specific classifier and exports:

- per-example evaluation embeddings
- per-example raw logits
- class centroids
- training metrics by epoch
- run metadata tied to the HPC job

That makes it the first run in this repo that directly supports the planned logit-geometry analysis.

## Interpretation

The result is operationally strong:

- training loss falls steadily across the 10 epochs
- evaluation accuracy is already very high by epoch 1 and ends at `0.9804`
- the run completed cleanly and produced all expected analysis artifacts

The result is also narrower than the earlier BGE-stage findings:

- evaluation was performed on a held-out split from the balanced CSV workflow
- this is not the same setup as the earlier BGE embedding-only validation stage
- the number therefore should not be treated as directly comparable to the earlier BGE macro-F1 and clustering metrics

In other words, this run establishes that the supervised Qwen stage works and that its logits are internally consistent with its exported embedding geometry, but it does not yet show whether the fine-tuned geometry is better, tighter, or more interpretable than the earlier BGE baseline.

## Label-Schema Caveat

The Qwen fine-tuned stage uses the label set:

- `anger`
- `fear`
- `happiness`
- `love`
- `sadness`
- `surprise`

The earlier BGE-stage reports use:

- `sadness`
- `joy`
- `love`
- `anger`
- `fear`
- `surprise`

The important mismatch is:

- `happiness` in the Qwen balanced CSV
- `joy` in the earlier BGE reports

These should not be silently treated as identical in cross-stage analysis. Any later geometry comparison must either map them explicitly or explain the semantic difference.

## Integration Decision

For repo structure, the correct approach is:

- keep the raw Qwen training artifacts in `experiments/text_model/runs/tmqb0010_17763/`
- expose the stage inside the dataset-level multiclass experiment via the lightweight bridge run `run-201-qwen3-1-7b-finetune-10e`
- keep future fine-tuned-stage plots and narrative under `experiments/text/multiclass/dair-ai-emotion/`

This avoids large-file duplication while making the fine-tuned stage visible alongside the BGE-stage work.

## Plot Output

The dataset-level plotting stage writes publication-facing outputs under:

- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/`

That directory now contains:

- `train-loss-vs-epoch.png`
- `eval-accuracy-vs-epoch.png`
- `eval-embeddings-pca-2d.png`
- `eval-embeddings-tsne-2d.png`
- `centroid-distance-heatmap.png`
- `projection-summary.json`

## Phase 3 Output

The dataset-level logit-geometry stage now writes analytical outputs under:

- `experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e/`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/logit-geometry/`

The main within-run signals are:

- mean per-example rank agreement between logits and centroid proximity:
  - Euclidean: `0.7189`
  - cosine: `0.7159`
- predicted class equals nearest centroid:
  - Euclidean: `0.9941`
  - cosine: `0.9926`
- global true-class logit vs distance correlation:
  - Euclidean: `-0.5802`
  - cosine: `-0.5307`

These numbers support a conservative claim of within-run geometry/logit coherence for the fine-tuned Qwen stage.

## Immediate Next Steps

1. Inspect the generated Phase 3 plots under `artifacts/plots/qwen-finetune-10e/logit-geometry/` for class-specific disagreement patterns and outliers.
2. Review `qwen-finetune-logit-geometry-findings.md` before making any stronger paper-facing claim.
3. Keep the earlier BGE stage as the geometry-first baseline until cross-stage comparison is explicitly label-aligned.
4. Treat any future Qwen run the same way:
   - create a bridge run
   - rerun Phase 2 plotting
   - rerun Phase 3 logit-geometry analysis
