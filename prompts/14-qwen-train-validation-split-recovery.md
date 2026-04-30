Recover and serialize the exact train/validation split used by the historical Qwen fine-tuning run so the same split can be reused for BGE and MPNet experiments.

## User request to satisfy

Pritish asked for the separation of training and validation data used to train Qwen, so that he can fine-tune BGE and MPNet on the exact same split and keep the experiments uniform.

The goal is not a new split. The goal is to recover the exact historical split used by the tracked Qwen run and write it out in a reusable form.

## Correct source run

Use the tracked Qwen source run:

- `experiments/text_model/runs/tmqb0010_17763/`

Read:

- `experiments/text_model/runs/tmqb0010_17763/run_metadata.json`

That run metadata records the exact split parameters used by the Qwen run:

- source CSV: `experiments/text/multiclass/dair-ai-emotion/data/raw/balanced_emotions_6classes.csv`
- text column: `cleaned_text`
- label column: `emotion`
- `test_size = 0.2`
- `seed = 42`

## Important reconstruction rule

Do not approximate the split by reading some other processed dataset.

Reconstruct the split by matching the historical Qwen training pipeline behavior in:

- `experiments/text_model/train_multiclass.py`

Specifically:

1. Read the raw CSV.
2. Apply the same row filtering logic as `load_texts_and_labels_from_csv(...)`.
3. Preserve the same row order that the Qwen training script saw.
4. Use the same label mapping logic as the original run.
5. Rebuild the split using `StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`.

The output must be the exact split implied by the historical Qwen run, not a hand-made or manually sampled version.

## Deliverables

Create reusable split artifacts under a dedicated directory, for example:

- `experiments/text/multiclass/dair-ai-emotion/data/splits/qwen_tmqb0010_17763/`

Write at minimum:

- `train.csv`
- `validation.csv`
- `split_metadata.json`

Each CSV should contain enough information for downstream reuse. Include:

- original row order or source row index
- `sentence`
- `emotion`
- `cleaned_text`

If a row was skipped by the original loader logic, it must not appear in either split file.

## What metadata must record

`split_metadata.json` should clearly document:

- source run id: `tmqb0010_17763`
- source CSV path
- reconstruction method
- text column used
- label column used
- `test_size`
- `seed`
- filtered row count
- training row count
- validation row count
- per-class counts for train and validation

Also include a brief note that this split is intended to be reused for the BGE and MPNet fine-tuning experiments to maintain cross-model comparability.

## Validation checks

Before finishing, verify and report:

1. `train.csv` row count + `validation.csv` row count = filtered row count.
2. No row appears in both splits.
3. The validation size matches the Qwen run expectation from the historical split parameters.
4. Class balance is stratified across train and validation.
5. The reconstruction logic is explicitly tied back to `tmqb0010_17763/run_metadata.json`.

## Reporting

Write a short report under `reports/` prefixed with `14-`, for example:

- `reports/14-qwen-train-validation-split-recovery.md`

The report should state:

- that it is the result of actioning this prompt
- which source run was used
- how the split was reconstructed
- where the reusable split files were written
- the final train and validation counts
- any assumptions or edge cases encountered

## Return format

Return:

- the files created or modified
- the exact location of the recovered train/validation split files
- the final train and validation counts
- confirmation that this is the same split definition used by the historical Qwen run
