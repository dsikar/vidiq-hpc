This report is the result of actioning `prompts/14-qwen-train-validation-split-recovery.md`.

# 14 Qwen Train Validation Split Recovery

## Source Run

The recovered split is tied to the historical Qwen source run:

- `experiments/text_model/runs/tmqb0010_17763/`

The reconstruction parameters were read from:

- `experiments/text_model/runs/tmqb0010_17763/run_metadata.json`

Key parameters from that run:

- source CSV: `experiments/text/multiclass/dair-ai-emotion/data/raw/balanced_emotions_6classes.csv`
- text column: `cleaned_text`
- label column: `emotion`
- `test_size = 0.2`
- `seed = 42`

## Reconstruction Method

The split was rebuilt to match the historical Qwen training pipeline, not the separate tracked processed `16000/2000` dataset split.

The recovery used the same logic as `experiments/text_model/train_multiclass.py`:

1. read the raw balanced CSV
2. apply the same CSV row filtering logic as `load_texts_and_labels_from_csv(...)`
3. preserve the same filtered row order seen by the Qwen training run
4. rebuild the same sorted label mapping
5. run `StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)`

To make this reproducible, a helper script was added:

- `scripts/recover_qwen_train_validation_split.py`

## Output Files

The recovered reusable split files were written to:

- `experiments/text/multiclass/dair-ai-emotion/data/splits/qwen_tmqb0010_17763/train.csv`
- `experiments/text/multiclass/dair-ai-emotion/data/splits/qwen_tmqb0010_17763/validation.csv`
- `experiments/text/multiclass/dair-ai-emotion/data/splits/qwen_tmqb0010_17763/split_metadata.json`

Each CSV includes:

- `source_row_index`
- `sentence`
- `emotion`
- `cleaned_text`

## Final Counts

After applying the historical Qwen CSV filtering logic:

- filtered row count: `20190`
- training row count: `16152`
- validation row count: `4038`

Per-class counts:

- filtered: `3365` per class
- train: `2692` per class
- validation: `673` per class

These counts are exactly consistent with a balanced six-class dataset and `test_size = 0.2`.

## Validation Checks

The generated `split_metadata.json` records and confirms:

- `train.csv` row count + `validation.csv` row count = filtered row count
- no row appears in both splits
- validation row count matches the historical Qwen expectation
- stratification is preserved across all six classes
- reconstruction is explicitly tied back to `tmqb0010_17763/run_metadata.json`

Note on `wc -l`:

- `train.csv` shows `16153` lines because it includes one header row plus `16152` data rows
- `validation.csv` shows `4039` lines because it includes one header row plus `4038` data rows

## Assumptions And Edge Cases

- The historical Qwen run definition is the source of truth, not the separately tracked processed `train/validation` dataset folders under `data/processed/`.
- Rows skipped by the original CSV loader because of missing text or missing label were excluded from both recovered split files.
- The recovered split is intended to be reused directly for BGE and MPNet fine-tuning so those runs stay aligned with the Qwen experiment.

## Conclusion

This is the same split definition used by the historical Qwen run `tmqb0010_17763`, reconstructed from the recorded run metadata and the original CSV filtering plus `StratifiedShuffleSplit` logic.
