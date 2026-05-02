# vidiq-hpc

High-performance computing experiments for the `vidiq` embedding-geometry project.

This repo currently covers two main workstreams:

- text experiments for embedding-geometry analysis on sentiment and emotion datasets
- HPC fine-tuning runs for a Qwen-based multiclass emotion classifier

## Main Areas

- `experiments.md`
  repo-level map of experiment families, models, datasets, and output locations
- `experiments/text/`
  dataset-specific text experiment pipelines, reports, configs, and outputs
- `experiments/text_model/`
  Qwen-based multiclass classifier training code and requirements
- `hpc/`
  SLURM batch scripts for Hyperion runs, including smoke tests and long-run jobs
- `reports/`
  survey reports and research notes, including the current image-dataset survey reports
- `meetings/`
  meeting minutes and transcripts
- `prompts/`
  reusable research prompts, including the image-dataset search prompt

## Text-Model Training

The main multiclass training entrypoint is:

- `experiments/text_model/train_multiclass.py`

Detailed implementation and paper-method notes:

- [experiments/text_model/README.md](experiments/text_model/README.md)

Current default backbone:

- `Qwen/Qwen3-1.7B`

The script supports:

- processed dataset roots via `--data-root`
- direct CSV training via `--csv-path`
- SLURM-derived run ids
- archived large artifacts with local run-directory symlinks
- export of evaluation embeddings, logits, centroid summaries, and metrics

## Hyperion / SLURM

Key batch scripts:

- `hpc/check_model_access.slurm`
  fast smoke test for Qwen tokenizer/model availability
- `hpc/train_multiclass.slurm`
  main multiclass training job
- `hpc/run_qwen_bge_parity.slurm`
  rebuilds the Qwen parity embedding/metric/plot bundle from the tracked Qwen bridge artifacts
- `hpc/train_multiclass_frozen_backbone.slurm`
  frozen-backbone variant
- `hpc/image_embedding_emotion6.slurm`
  smaller-dataset image sanity benchmark on Emotion6
- `hpc/image_embedding_fi.slurm`
  smaller-dataset image benchmark on FI
- `hpc/image_embedding_emoset.slurm`
  legacy EmoSet-oriented image job, kept for comparison rather than immediate priority
- `hpc/train_multiclass_balanced_{10e,50e,100e,250e,500e,1000e}.slurm`
  balanced 6-class CSV runs
- `hpc/train_multiclass_unbalanced_{10e,50e,100e,250e,500e,1000e}.slurm`
  unbalanced 20-class CSV runs

Standard Hyperion flow:

```bash
git pull
sbatch hpc/check_model_access.slurm
sbatch hpc/train_multiclass_balanced_10e.slurm
```

## Batch Commands

Smoke test:

```bash
sbatch hpc/check_model_access.slurm
```

Training:

```bash
sbatch hpc/train_multiclass.slurm
sbatch hpc/train_multiclass_frozen_backbone.slurm
sbatch hpc/train_multiclass_balanced_10e.slurm
sbatch hpc/train_multiclass_balanced_50e.slurm
sbatch hpc/train_multiclass_balanced_100e.slurm
sbatch hpc/train_multiclass_balanced_250e.slurm
sbatch hpc/train_multiclass_balanced_500e.slurm
sbatch hpc/train_multiclass_balanced_1000e.slurm
sbatch hpc/train_multiclass_unbalanced_10e.slurm
sbatch hpc/train_multiclass_unbalanced_50e.slurm
sbatch hpc/train_multiclass_unbalanced_100e.slurm
sbatch hpc/train_multiclass_unbalanced_250e.slurm
sbatch hpc/train_multiclass_unbalanced_500e.slurm
sbatch hpc/train_multiclass_unbalanced_1000e.slurm
```

Qwen parity:

```bash
sbatch hpc/run_qwen_bge_parity.slurm
```

Image Sentiment:

```bash
sbatch --export=DATASET=emotion6 hpc/prepare_image_dataset.slurm
sbatch hpc/image_embedding_emoset.slurm
sbatch hpc/image_embedding_emotion6.slurm
sbatch hpc/image_embedding_fi.slurm
```

For the image workflow on Hyperion:

- dataset preparation now has its own dedicated SLURM surface:
  - `sbatch --export=DATASET=emotion6 hpc/prepare_image_dataset.slurm`
  - `sbatch --export=DATASET=fi hpc/prepare_image_dataset.slurm`
  - if you already have a manually acquired source dataset elsewhere, pass it with `SOURCE_ROOT`, for example:
    - `sbatch --export=DATASET=emotion6,SOURCE_ROOT=/path/to/manual/emotion6 hpc/prepare_image_dataset.slurm`
    - `sbatch --export=DATASET=fi,SOURCE_ROOT=/path/to/manual/fi hpc/prepare_image_dataset.slurm`
- the next sanity-check job should be:
  - `sbatch hpc/image_embedding_emotion6.slurm`
- the next follow-on smaller benchmark job should be:
  - `sbatch hpc/image_embedding_fi.slurm`
- `Emotion6` staged dataset root defaults to `/users/aczd097/archive/vidiq-hpc/data/image/emotion6`
- `FI` staged dataset root defaults to `/users/aczd097/archive/vidiq-hpc/data/image/fi`
- `EmoSet` staged dataset root defaults to `/users/aczd097/archive/vidiq-hpc/data/image/emoset`
- Hugging Face dataset cache defaults to `/users/aczd097/sharedscratch/huggingface/datasets`
- the image runner is now dataset-config driven via:
  - `configs/emoset_phase1.json`
  - `configs/emotion6_phase1.json`
  - `configs/fi_phase1.json`
- on Hyperion, image run directories now append `SLURM_JOB_ID`, so the effective run ids are shaped like:
  - `emotion6_phase1_clip_<jobid>`
  - `fi_phase1_clip_<jobid>`
- `Emotion6` and `FI` currently expect staged local data and do not claim automatic dataset download support
- the preparation script can optionally copy from a manually acquired source root into the staged target root, then validate the dataset layout:
  - `python3 scripts/prepare_image_dataset.py --dataset emotion6 --source-root /path/to/manual/source`
  - `python3 scripts/prepare_image_dataset.py --dataset fi --source-root /path/to/manual/source`
- if manual acquisition is still required, the preparation step fails clearly instead of pretending the dataset is available
- `EmoSet` still prefers a staged local dataset if present, and otherwise falls back to the public Hugging Face EmoSet mirror
- the current image config also runs a linear-probe evaluation on the generated embeddings and saves classification metrics under `experiments/image/runs/<run_name>/artifacts/metrics/`
- `Emotion6` and `FI` each run from their own dedicated SLURM batch file; they are not meant to share one mixed dataset job

Expected staged local layout for `Emotion6` and `FI`:

```text
<data_root>/
  metadata.csv
  images/
    ...
```

`metadata.csv` should include:

- one image path column:
  - `image_path`, `filepath`, `path`, `image`, or `filename`
- one label column:
  - `emotion`, `label_name`, `label`, `class`, or `category`

Canonical labels:

- `Emotion6`: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`
- `FI`: `amusement`, `anger`, `awe`, `contentment`, `disgust`, `excitement`, `fear`, `sadness`

Operational order for `Emotion6` and `FI`:

1. run the dataset preparation job
2. confirm the staged root contains `metadata.csv`, `images/`, and a `preparation_manifest.json`
3. run the main image embedding job

Concrete Hyperion sequence:

```bash
sbatch --export=DATASET=emotion6 hpc/prepare_image_dataset.slurm
sbatch hpc/image_embedding_emotion6.slurm

sbatch --export=DATASET=fi hpc/prepare_image_dataset.slurm
sbatch hpc/image_embedding_fi.slurm
```

## Pushing Results Back

When an HPC job finishes, review the generated files first:

```bash
git status --short
```

Typical tracked result locations in this repo are:

- `experiments/text/.../runs/`
- `experiments/text/.../artifacts/plots/`
- `experiments/text/.../artifacts/metrics/`
- `experiments/image/runs/<run_name>/`
- `experiments/text_model/runs/<run_id>/analysis/`
- `experiments/text_model/runs/<run_id>/run_metadata.json`
- `experiments/text_model/runs/<run_id>/train_metrics.json`

Typical ignored or non-remote result locations are:

- `outputs/`
- `experiments/text/.../artifacts/embeddings/*.npy`
- `experiments/text/multiclass/dair-ai-emotion/artifacts/embeddings/*.npy`
- `experiments/embeddings_field/.../artifacts/embeddings/*.npy`
- `experiments/text_model/runs/<run_id>/model/`
- `experiments/text_model/runs/<run_id>/tokenizer/`
- `/users/aczd097/archive/vidiq-hpc/data/image/...`
- `/users/aczd097/sharedscratch/huggingface/...`

For an image run such as `emotion6_phase1_clip_<jobid>` or `fi_phase1_clip_<jobid>`, you will usually want to stage:

```bash
git add \
  experiments/image/runs/<run_name>/artifacts/centroids.json \
  experiments/image/runs/<run_name>/artifacts/density.json \
  experiments/image/runs/<run_name>/artifacts/embeddings.pt \
  experiments/image/runs/<run_name>/artifacts/labels.pt \
  experiments/image/runs/<run_name>/artifacts/label_map.json \
  experiments/image/runs/<run_name>/artifacts/metrics/classification-summary.json \
  experiments/image/runs/<run_name>/artifacts/metrics/confusion-matrix.json \
  experiments/image/runs/<run_name>/artifacts/metrics/train-history.json \
  experiments/image/runs/<run_name>/artifacts/models/linear_probe.pt \
  experiments/image/runs/<run_name>/logs/run_metadata.json
git status --short
```

If you changed the setup on Hyperion while producing the run, also stage the corresponding code/config files, for example:

```bash
git add \
  configs/emotion6_phase1.json \
  configs/fi_phase1.json \
  hpc/image_embedding_emotion6.slurm \
  hpc/image_embedding_fi.slurm \
  scripts/run_image_embeddings.py \
  src/image_experiments/config.py \
  src/image_experiments/datasets.py
git status --short
```

Do not push:

- `outputs/`
- `preparation_manifest.json` under Hyperion archive dataset roots
- staged raw image datasets under `/users/aczd097/archive/vidiq-hpc/data/image/...`
- Hugging Face caches under `/users/aczd097/sharedscratch/huggingface/...`
- unrelated scratch files

After implementing or revising the preparation workflow itself, the files to push are the helper surfaces, for example:

```bash
git add \
  scripts/prepare_image_dataset.py \
  hpc/prepare_image_dataset.slurm \
  README.md
git status --short
```

For a normal result push:

```bash
git add experiments/ README.md
git status --short
git commit -m "Add latest experiment results"
git push
```

For a Qwen training run, you will usually want to stage the tracked source-run outputs under `experiments/text_model/runs/<run_id>/` and any dataset-level bridge, metrics, plots, or report updates under `experiments/text/multiclass/dair-ai-emotion/`.

For the Qwen parity workflow, you will usually want to stage:

```bash
git add \
  experiments/text/multiclass/dair-ai-emotion/runs/ \
  experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/ \
  experiments/text/multiclass/dair-ai-emotion/artifacts/plots/ \
  experiments/text/multiclass/dair-ai-emotion/artifacts/embeddings/*.json
git status --short
git commit -m "Add Qwen parity results"
git push
```

Use `git status --short` before committing so you do not accidentally try to push ignored HPC scratch output or unrelated local edits.

Run outputs are written under:

- `experiments/text_model/runs/<run_id>/`

Tracked run artifacts include:

- `analysis/`
- `run_metadata.json`
- `train_metrics.json`

Ignored heavyweight artifacts:

- `model/`
- `tokenizer/`

## Current Tracked Training Result

The first tracked balanced fine-tuning run is:

- `experiments/text_model/runs/tmqb0010_17763/`

It includes:

- `analysis/centroids.json`
- `analysis/eval_embeddings.npy`
- `analysis/eval_logits.npy`
- `analysis/eval_metrics.json`
- `run_metadata.json`
- `train_metrics.json`

Additional Qwen long-run results from the Hyperion `50e`, `100e`, `250e`, `500e`, and `1000e` jobs should be incorporated through the dataset-level bridge workflow documented in:

- `experiments/text/multiclass/dair-ai-emotion/README.md`
- `reports/qwen-finetune-phase3-implementation-report-codex.md`

## Image Dataset Survey Work

The current image-extension planning material includes:

- prompt: `prompts/04-image-sentiment-dataset-search.md`
- reports:
  - `reports/chatgpt-image-datasets.md`
  - `reports/claude-image-datasets.md`
  - `reports/deepseek-image-datasets.md`
  - `reports/gemini-image-datasets.md`
  - `reports/grok-image-datasets.md`
  - `reports/lechat-image-datasets.md`

These reports are intended to support selection of the first image dataset for the cross-modality extension discussed in the 16 April meeting minutes.

## Environment

Original local note:

- Pritish: run this project inside `gemini_env`.

For Hyperion jobs, the current SLURM scripts activate:

- `/users/aczd097/sharedscratch/venvs/main/bin/activate`
