Write a concrete repository plan and begin the implementation work needed to pivot the image-embedding experiments away from the noisy EmoSet-first assumption and toward smaller, cleaner emotion-labeled datasets.

Repository root: `/home/daniel/git/vidiq/vidiq-hpc`

## Why this prompt exists

The current image workstream was originally planned around `EmoSet-118K`, but recent experiment results indicate that EmoSet is too noisy for the current goal:

- cluster separation was weak
- the embeddings did not form clean emotion structure
- we now want smaller datasets with cleaner labels that have a better chance of producing separable embedding geometry

The next practical candidates are:

1. `Emotion6` as the smallest clean sanity-check dataset
2. `FI` as the main smaller image-emotion dataset
3. optionally `OASIS` later as a normative validation/control set, but **do not make OASIS the main implementation target in this task**

## Source-of-truth context to inspect first

Before changing anything, inspect these files:

- `work-diary.md`
- `experiments.md`
- `reports/claude-image-datasets.md`
- `reports/chatgpt-image-datasets.md`
- `reports/deepseek-image-datasets.md`
- `reports/gemini-image-datasets.md`
- `reports/image-embedding-audit-report.md`
- `prompts/08-image-embedding-experiment-implementation.md`

Then inspect the current image code surface:

- `src/image_experiments/config.py`
- `src/image_experiments/datasets.py`
- `src/image_experiments/embeddings.py`
- `src/image_experiments/geometry.py`
- `src/image_experiments/io_utils.py`
- `src/image_experiments/training.py`
- `scripts/run_image_embeddings.py`
- `configs/emoset_phase1.json`
- `hpc/image_embedding_emoset.slurm`
- `README.md`
- `hpc/extract_qwen_validation_embeddings.slurm`

Important: the repository may have evolved beyond the older audit report. Do not assume functionality is missing until you inspect the current code.

## What you need to do

### Part A: Write a concrete repo plan

Write a short, concrete planning report under `reports/` with a clear name prefixed by `16-`, for example:

- `reports/16-smaller-image-datasets-plan.md`

The plan must cover:

1. why `EmoSet` is being deprioritized for the next stage
2. why `Emotion6` and `FI` are the next implementation targets
3. what experiment order should be used
4. what parts of the current image framework can be reused unchanged
5. what code/config changes are required to support multiple image datasets cleanly
6. which dataset should be treated as:
   - quick sanity benchmark
   - main smaller emotion benchmark
   - later validation/control
7. what should be implemented now versus deferred
8. the exact Hyperion SLURM job(s) that should be used for the next image runs
9. which generated files should be pushed back to remote and which should not

The plan should be specific to this repository, not a generic research memo.

### Part B: Start the setup work

Begin the implementation needed to support `Emotion6` and `FI` in the existing image pipeline.

The goal of this task is **initial repo setup**, not necessarily fully running both datasets end-to-end if access or staging is blocked.

#### Required implementation direction

Prefer extending the current architecture rather than replacing it.

The desired result is that the image pipeline can support dataset-specific runs via config rather than being effectively hardwired to EmoSet.

At minimum, implement whatever is needed so that the repo has a credible, concrete starting point for:

- `Emotion6` extraction + geometry runs
- `FI` extraction + geometry runs
- optional linear-probe evaluation if the existing code already supports it cleanly

#### Expected work items

Inspect the current code and implement the appropriate subset of these changes:

1. generalize dataset selection in the image config / runtime surface
2. add dataset classes or dataset-loading pathways for:
   - `Emotion6`
   - `FI`
3. create initial config files for those datasets, for example:
   - `configs/emotion6_phase1.json`
   - `configs/fi_phase1.json`
4. create or update HPC batch files if needed so these configs are runnable on Hyperion
5. ensure the setup is honest about data access constraints:
   - if a dataset cannot be auto-downloaded legally or practically, support staged local data and document that clearly
   - do not fake automated download support where the real repo cannot provide it
6. document the expected on-disk dataset layout for both datasets
7. keep outputs under the established `experiments/image/runs/<run_name>/` structure

### Part B.1: Hyperion execution requirement

The prompt must result in a clear answer to:

- which SLURM file should be run next on Hyperion
- what command shape it launches
- which partition / cluster setup it uses
- how the dataset runs are separated from one another

Use the latest Qwen extraction job as the reference pattern:

- `hpc/extract_qwen_validation_embeddings.slurm`

The new image job should be designed to run on the same Hyperion environment and same class of cluster setup as that recent Qwen job unless repo evidence shows a strong reason to do otherwise.

Dataset separation is required:

- `Emotion6` must run from its own dedicated `.slurm` batch file
- `FI` must run from its own dedicated `.slurm` batch file
- do not combine both datasets into one generic batch file for this task unless the repo already has a clean wrapper pattern and the dedicated dataset files still exist
- the separation between datasets must be obvious from the filenames, config paths, and run names

The plan must explicitly state that each dataset runs from one `.slurm` batch file of its own.

Be explicit in the plan and implementation about:

- the SLURM filename(s) to run
- the config each SLURM file points to
- whether the job is intended for `Emotion6` or `FI`
- the expected run name
- the fact that each dataset has its own batch file and is not sharing a single mixed dataset job

If you create new SLURM files, name them clearly, for example:

- `hpc/image_embedding_emotion6.slurm`
- `hpc/image_embedding_fi.slurm`

### Part B.2: Push-back-to-remote requirement

The prompt must also result in a clear answer to:

- which files should be pushed back to remote after an HPC image run finishes
- which files should not be pushed

Follow the repository’s existing tracked-versus-ignored conventions. Be explicit about:

- tracked run outputs under `experiments/image/runs/<run_name>/`
- any configs / scripts / SLURM files that should also be committed if they were changed on HPC
- any heavyweight, cache, scratch, or local-only files that should not be pushed

If the implementation introduces new run artifacts, document which of those are intended to be tracked in Git and which are not.

### Part C: Keep the first experiments disciplined

The setup should reflect the fact that we are trying to maximize geometric separability, not dataset size.

So the implementation and plan should encourage:

- starting with `Emotion6` as a small sanity-check corpus
- using `FI` as the main smaller benchmark
- beginning with the cleanest label surface available
- optionally supporting coarser grouped evaluation if the dataset taxonomy makes that useful

Do **not** pivot this task toward:

- large noisy social-image datasets
- face-only emotion datasets as the main benchmark
- art-only datasets as the primary benchmark
- a major architectural rewrite of the current image framework

## Constraints

- Reuse the existing `src/`, `scripts/`, `configs/`, and `hpc/` structure.
- Keep path handling robust for Hyperion.
- Preserve the existing output layout under `experiments/image/runs/`.
- Do not make misleading claims about downloadability or licensing.
- If a dataset requires manual acquisition or request-based access, implement the staging path and document the requirement explicitly.
- Prefer small, concrete changes that move the repo toward runnable Emotion6/FI experiments.
- The plan must explicitly state the next SLURM job to run on Hyperion and the files to push back to remote after the run.
- Each dataset must have clear operational separation, including one dedicated `.slurm` file per dataset.

## Deliverables

At minimum, produce:

- one new report under `reports/` describing the concrete repo plan
- the code/config/script changes needed to begin Emotion6 and FI support
- any README or documentation updates needed to explain the new setup
- clear instructions for:
  - the next Hyperion SLURM job to run
  - the files to push back to remote after that run
- clearly separated SLURM execution surfaces for:
  - `Emotion6`
  - `FI`

## Acceptance criteria

Your work is successful if:

- the repo contains a clear, concrete plan for the smaller-dataset pivot
- the image pipeline is no longer effectively tied only to EmoSet
- there are initial configs for `Emotion6` and `FI`
- the repo documents how each dataset is expected to be staged or loaded
- the repo gives a clear Hyperion run instruction for the next image job
- the repo gives a clear list of which resulting files should be pushed to remote
- `Emotion6` and `FI` do not share one ambiguous batch file; each has its own clear `.slurm` entrypoint
- any new implementation matches the existing repo patterns and does not overclaim unsupported functionality

## Final output

When finished, report:

- the plan file you wrote
- the exact files changed
- what `Emotion6` support now exists
- what `FI` support now exists
- which SLURM job should be run next on Hyperion
- which files should be pushed back to remote after that run
- any remaining blockers for actually running those datasets on Hyperion
