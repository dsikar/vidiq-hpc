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

In other words, this run establishes that the supervised Qwen stage works, but it does not yet show whether the fine-tuned geometry is better, tighter, or more interpretable than the earlier BGE baseline.

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

## Immediate Next Steps

1. Add dataset-level plots for this run under `artifacts/plots/`.
2. Inspect the saved embeddings and centroids to determine whether fine-tuning tightened class structure.
3. Add a logit-geometry analysis step before making any claim about geometry improvement.
4. Keep the earlier BGE stage as the geometry-first baseline until those comparisons are complete.
