# 16 Smaller Image Datasets Plan

This report replaces the earlier implicit `EmoSet`-first assumption for the next image-embedding experiments.

## Why EmoSet is being deprioritized

The current repo already has an `EmoSet`-based image pipeline, but the latest outcome from that direction is that the resulting embedding geometry was too noisy and did not produce the cluster separation we want.

That makes `EmoSet` a poor immediate next benchmark for the current objective:

- establish whether the image pipeline can recover clean emotion structure at all
- measure whether a smaller, cleaner label surface improves centroid separation
- avoid burning additional Hyperion time on a noisy first-stage corpus

`EmoSet` is not being deleted from the repo. It is being moved out of the immediate critical path.

## Why Emotion6 and FI are the next targets

### Emotion6

`Emotion6` is the next sanity-check dataset because it is small and easier to stage, and the survey material describes it as having stronger human annotation quality than most large social-image corpora.

Use it to answer the narrow question:

- does the current image embedding pipeline show any meaningful emotion separation on a smaller, cleaner dataset?

### FI

`FI` is the next main smaller benchmark because it remains image-level, emotion-labeled, and materially smaller than `EmoSet` while still being large enough to matter as a practical benchmark.

Use it to answer the next question:

- once the pipeline is sanity-checked on `Emotion6`, does a somewhat larger but still cleaner emotion dataset produce better separation than `EmoSet`?

### OASIS

`OASIS` remains a good later validation/control dataset, but it should not be the first operational target in this pivot. It is better used after `Emotion6` and `FI` to validate manifold behavior against a more normative affect surface.

## Experiment order

The immediate run order should be:

1. `Emotion6` full-image extraction + geometry + linear-probe evaluation
2. `FI` full-image extraction + geometry + linear-probe evaluation
3. `OASIS` later, only if the first two runs justify adding a normative control

That order keeps the first Hyperion job cheap and diagnostic.

## What can be reused unchanged

The current image framework already provides the main surfaces we need:

- `scripts/run_image_embeddings.py`
- `src/image_experiments/embeddings.py`
- `src/image_experiments/geometry.py`
- `src/image_experiments/training.py`
- structured run output under `experiments/image/runs/<run_name>/`
- frozen-embedding linear-probe evaluation artifacts

The main missing piece was not evaluation logic anymore. It was dataset generalization and disciplined run surfaces.

## What needed to change

The repo needed these specific changes:

1. make dataset selection explicit in the image config surface
2. stop assuming the image pipeline is effectively EmoSet-only
3. add honest staged-local dataset loaders for:
   - `Emotion6`
   - `FI`
4. add initial configs:
   - `configs/emotion6_phase1.json`
   - `configs/fi_phase1.json`
5. add one dedicated Hyperion `.slurm` job per dataset:
   - `hpc/image_embedding_emotion6.slurm`
   - `hpc/image_embedding_fi.slurm`
6. update repo docs so they state:
   - the next job to run
   - the expected staged dataset layout
   - which files should be pushed back to remote

## Dataset roles

- quick sanity benchmark: `Emotion6`
- main smaller emotion benchmark: `FI`
- later validation/control: `OASIS`

## What is implemented now vs deferred

Implemented now:

- config-level dataset selection
- staged-local `Emotion6` support
- staged-local `FI` support
- a separate dataset preparation / validation script
- a separate Hyperion `.slurm` wrapper for dataset preparation
- dedicated config files for `Emotion6` and `FI`
- dedicated `.slurm` files for `Emotion6` and `FI`
- updated README / experiment inventory guidance

Deferred:

- fully automated acquisition of `Emotion6` or `FI`
- any claim that `FI` can be downloaded directly by this repo without manual staging
- any move to `OASIS`
- any segmented-object follow-up via `EmoVerse`

## Hyperion jobs to run

The jobs should run on the same Hyperion class of setup as the latest Qwen extraction job:

- partition: `gpu-a100`
- GPU: `a100`
- shared venv: `/users/aczd097/sharedscratch/venvs/main/bin/activate`

Each dataset has its own dedicated batch file:

- `Emotion6`: `hpc/image_embedding_emotion6.slurm`
- `FI`: `hpc/image_embedding_fi.slurm`

### Next job to run

Run this first:

```bash
sbatch --export=DATASET=emotion6 hpc/prepare_image_dataset.slurm
```

That preparation job launches:

```bash
python3 scripts/prepare_image_dataset.py --dataset emotion6
```

Then run:

```bash
sbatch hpc/image_embedding_emotion6.slurm
```

That main job launches:

```bash
python3 scripts/run_image_embeddings.py --config configs/emotion6_phase1.json
```

Expected main run name:

- `emotion6_phase1_clip_<jobid>`

After that is reviewed, run:

```bash
sbatch --export=DATASET=fi hpc/prepare_image_dataset.slurm
```

Then run:

```bash
sbatch hpc/image_embedding_fi.slurm
```

That main job launches:

```bash
python3 scripts/run_image_embeddings.py --config configs/fi_phase1.json
```

Expected run name:

- `fi_phase1_clip_<jobid>`

## Expected staged dataset layout

Both `Emotion6` and `FI` currently expect staged local data in this shape:

```text
<data_root>/
  metadata.csv
  images/
    ...
```

`metadata.csv` should contain:

- an image path column:
  - `image_path`, `filepath`, `path`, `image`, or `filename`
- a label column:
  - `emotion`, `label_name`, `label`, `class`, or `category`

Canonical labels:

- `Emotion6`: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise`
- `FI`: `amusement`, `anger`, `awe`, `contentment`, `disgust`, `excitement`, `fear`, `sadness`

## What to push back to remote

After a stable image run finishes, push the run bundle under:

- `experiments/image/runs/<run_name>/`

For the first stable `Emotion6` run, that means:

- `experiments/image/runs/emotion6_phase1_clip_<jobid>/artifacts/centroids.json`
- `experiments/image/runs/emotion6_phase1_clip_<jobid>/artifacts/density.json`
- `experiments/image/runs/emotion6_phase1_clip_<jobid>/artifacts/embeddings.pt`
- `experiments/image/runs/emotion6_phase1_clip_<jobid>/artifacts/labels.pt`
- `experiments/image/runs/emotion6_phase1_clip_<jobid>/artifacts/label_map.json`
- `experiments/image/runs/emotion6_phase1_clip_<jobid>/artifacts/metrics/classification-summary.json`
- `experiments/image/runs/emotion6_phase1_clip_<jobid>/artifacts/metrics/confusion-matrix.json`
- `experiments/image/runs/emotion6_phase1_clip_<jobid>/artifacts/metrics/train-history.json`
- `experiments/image/runs/emotion6_phase1_clip_<jobid>/artifacts/models/linear_probe.pt`
- `experiments/image/runs/emotion6_phase1_clip_<jobid>/logs/run_metadata.json`

If the setup was edited on Hyperion while producing that run, also push the setup files:

- `configs/emotion6_phase1.json`
- `hpc/image_embedding_emotion6.slurm`
- `scripts/run_image_embeddings.py`
- `src/image_experiments/config.py`
- `src/image_experiments/datasets.py`

Equivalent logic applies to `FI` using `fi_phase1_clip_<jobid>`.

Do not push:

- `outputs/`
- `preparation_manifest.json` under archive dataset roots
- staged raw image datasets under `/users/aczd097/archive/vidiq-hpc/data/image/...`
- Hugging Face caches under `/users/aczd097/sharedscratch/huggingface/...`
- unrelated scratch files

## Remaining blockers

The main remaining blockers are operational rather than architectural:

1. `Emotion6` and `FI` still need to be staged locally on Hyperion.
2. `FI` access may remain request-based depending on the exact source used.
3. The first stable `experiments/image/` output bundle still needs to be produced and reviewed before the push guidance is exercised for real.
