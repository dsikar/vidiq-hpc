This implementation report is the result of actioning `prompts/13-qwen-off-the-shelf-validation-embeddings.md`.

# 13 Qwen Off-The-Shelf Validation Embeddings Implementation Notes

## Scope

The requested change was to support extraction of off-the-shelf pretrained Qwen embeddings for the tracked `dair-ai/emotion` validation split, without reusing the historical fine-tuned hold-out reconstruction path.

The correct implementation surface was:

- `experiments/text_model/train_multiclass.py`

The downstream bridge/parity workflow was not modified because it only repackages already-generated fine-tuned artifacts.

## Files Changed

Modified:

- `experiments/text_model/train_multiclass.py`

Added:

- `hpc/extract_qwen_validation_embeddings.slurm`
- `prompts/13-qwen-off-the-shelf-validation-embeddings.md`

## Implementation Summary

`experiments/text_model/train_multiclass.py` now supports two modes:

- `--mode train`
- `--mode extract-only`

The default remains `train`, so the existing fine-tuning SLURM jobs keep their current behavior.

For processed JSONL data, the script now also supports:

- `--data-split train`
- `--data-split validation`

When `--mode extract-only` is used with:

```bash
--data-root experiments/text/multiclass/dair-ai-emotion --data-split validation
```

the script reads directly from:

- `experiments/text/multiclass/dair-ai-emotion/data/processed/validation/texts.jsonl`

and does not build a new stratified split.

## Extract-Only Behavior

In `extract-only` mode, the script:

- loads pretrained `Qwen/Qwen3-1.7B`
- runs a forward pass over the tracked validation split
- extracts embeddings from the final transformer hidden state at the last token position
- casts embeddings to `float32`
- skips training, optimizer setup, checkpoint saving, archive model writes, tokenizer archive writes, and classifier-logit export

This keeps the extracted representation aligned with the existing fine-tuned Qwen embedding definition while clearly separating pretrained-only runs from fine-tuned runs.

## Output Artifacts

The new extract-only mode writes these artifacts under:

- `experiments/text_model/runs/<run_id>/analysis/validation_embeddings.npy`
- `experiments/text_model/runs/<run_id>/analysis/validation_metrics.json`
- `experiments/text_model/runs/<run_id>/analysis/validation_centroids.json`

It also writes:

- `experiments/text_model/runs/<run_id>/run_metadata.json`

`validation_metrics.json` records:

- dataset size
- embedding dimension
- source split
- model name
- mode = `extract-only`
- `pretrained_only = true`

`run_metadata.json` now also distinguishes:

- `mode`
- `fine_tuning_performed`
- `pretrained_only`
- processed split path used

## Backward Compatibility

The existing training path remains responsible for:

- `analysis/eval_embeddings.npy`
- `analysis/eval_logits.npy`
- `analysis/eval_metrics.json`
- `analysis/centroids.json`

No existing training artifact names were changed.

## Validation Notes

Local checks completed:

- `python3 -m py_compile experiments/text_model/train_multiclass.py`
- `bash -n hpc/extract_qwen_validation_embeddings.slurm`

Both passed.

I also confirmed that the tracked validation JSONL already contains `label` values, so centroid generation for the validation split can be performed directly from the processed validation data.

## Local Limitation

I could not run a full local extraction pass from this checkout because the default local `python3` environment is missing `numpy`, so runtime verification of the actual output array shape was not completed locally.

That means the implementation is syntactically validated and wired to the correct data split, but the final execution check still needs to happen in the proper project environment or on Hyperion.

## Recommended Run

The intended batch entrypoint for the new workflow is:

- `hpc/extract_qwen_validation_embeddings.slurm`

It launches:

```bash
python experiments/text_model/train_multiclass.py \
    --mode extract-only \
    --data-root experiments/text/multiclass/dair-ai-emotion \
    --data-split validation \
    --run-root experiments/text_model/runs \
    --run-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
    --batch-size 8 \
    --max-length 256 \
    --model-path Qwen/Qwen3-1.7B
```

## HPC Directive

Run this SLURM job on Hyperion:

- `hpc/extract_qwen_validation_embeddings.slurm`

Once the job completes and the real `<run_id>` is known, the first files that need to be pushed back to remote are:

- `experiments/text_model/runs/<run_id>/analysis/validation_embeddings.npy`
- `experiments/text_model/runs/<run_id>/analysis/validation_metrics.json`
- `experiments/text_model/runs/<run_id>/analysis/validation_centroids.json`
- `experiments/text_model/runs/<run_id>/run_metadata.json`

These are the Prompt 13 source-run outputs that matter for the repo and paper workflow.

Do not push back:

- `outputs/`
- `experiments/text_model/runs/<run_id>/model/`
- `experiments/text_model/runs/<run_id>/tokenizer/`

If the Prompt 13 code changes were made on HPC rather than locally, also push:

- `experiments/text_model/train_multiclass.py`
- `hpc/extract_qwen_validation_embeddings.slurm`
- `prompts/13-qwen-off-the-shelf-validation-embeddings.md`
- `reports/13-qwen-off-the-shelf-validation-embeddings-implementation-notes.md`

## Assumptions

- The processed validation split is the source of truth for this paper request.
- Validation labels in `texts.jsonl` are correct and can be used to compute `validation_centroids.json`.
- Pretrained extraction runs should not emit synthetic or meaningless classifier logits.
