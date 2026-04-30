Write the implementation needed to stage the smaller image datasets before the main image embedding jobs run.

Repository root: `/home/daniel/git/vidiq-hpc`

## Why this prompt exists

The repo now has separate image experiment configs and SLURM jobs for:

- `Emotion6`
- `FI`

However, those jobs currently assume the datasets are already staged locally under the expected Hyperion data roots. We now need a separate preparation step that fetches, downloads, or otherwise stages the required dataset files before the main image experiment jobs are run.

The goal is to make dataset availability explicit and operationally reproducible.

## Context to inspect first

Before editing anything, inspect:

- `prompts/16-smaller-image-datasets-plan-and-setup.md`
- `reports/16-smaller-image-datasets-plan.md`
- `README.md`
- `configs/emotion6_phase1.json`
- `configs/fi_phase1.json`
- `hpc/image_embedding_emotion6.slurm`
- `hpc/image_embedding_fi.slurm`
- `src/image_experiments/datasets.py`
- `scripts/run_image_embeddings.py`

Also inspect any current repo guidance around Hyperion / SLURM environment setup and tracked vs ignored outputs.

## Task

Implement a separate dataset-preparation surface for the smaller image datasets.

## Required outcomes

### 1. Add a dedicated download / staging script

Create a separate script whose job is to ensure required dataset files are available before the main image run starts.

The script should live under `scripts/`, for example:

- `scripts/prepare_image_dataset.py`

Its purpose is:

- accept a dataset target such as `emotion6` or `fi`
- prepare the dataset under the expected staged local root
- verify that the required files exist after preparation
- fail clearly if the dataset cannot legally or practically be auto-downloaded

Important:

- do not silently pretend a dataset was downloaded if it actually still requires manual acquisition
- if `Emotion6` or `FI` must be manually staged, the script should perform validation and emit a precise error explaining what is missing
- if partial automation is possible, implement only what is actually defensible from repo evidence

The script should make the expected staged dataset layout explicit, including:

- `metadata.csv`
- `images/`

It should be able to validate that the dataset root is ready for:

- `configs/emotion6_phase1.json`
- `configs/fi_phase1.json`

### 2. Add a dedicated SLURM wrapper for the preparation step

Create a `.slurm` script that runs the new preparation script on Hyperion before the main image experiment job.

The batch file should live under `hpc/`, for example:

- `hpc/prepare_image_dataset.slurm`

If the repo structure and clarity are better served by separate dataset-specific preparation batch files, that is also acceptable, but be explicit and keep the naming unambiguous.

At minimum, the SLURM wrapper must:

- run on the same Hyperion environment class as the recent Qwen extraction job and the new smaller image jobs
- activate the shared environment
- resolve `REPO_ROOT` robustly
- set `PYTHONPATH` appropriately
- prepare the dataset root before exit

The SLURM wrapper should be clearly documented as the preparation step that must run before the main image embedding job.

### 3. Make the operational order explicit

Update the docs so they clearly state the order:

1. run the dataset preparation `.slurm` job
2. verify the staged dataset root is ready
3. run the main image embedding `.slurm` job

This should be documented for at least:

- `Emotion6`
- `FI`

### 4. Keep tracked vs non-tracked outputs explicit

Document which preparation-step outputs should be pushed back to remote and which should not.

Examples:

- helper scripts, configs, and `.slurm` files should be tracked if changed
- raw staged datasets under Hyperion archive storage should not be pushed
- scratch logs and `outputs/` should not be pushed

## Constraints

- Reuse the current repo structure: `scripts/`, `hpc/`, `configs/`, `src/`.
- Do not collapse dataset preparation into the main image embedding script for this task; the point is to create a separate preparation surface.
- Be honest about licensing and access limitations.
- Prefer explicit validation over fake automation.
- Keep the Hyperion setup aligned with the current repo conventions.

## Deliverables

At minimum, produce:

- one new preparation script under `scripts/`
- one new `.slurm` script under `hpc/`
- any README or report updates needed to explain the new preparation workflow

## Acceptance criteria

Your work is successful if:

- the repo contains a dedicated dataset preparation script
- the repo contains a dedicated `.slurm` wrapper for that preparation step
- the docs clearly explain that preparation runs before the main image experiment jobs
- the implementation is honest about what can and cannot be auto-downloaded
- the preparation workflow is clearly defined for `Emotion6` and `FI`

## Final output

When finished, report:

- the files changed
- what the preparation script does
- which `.slurm` job should be run to prepare datasets
- how that preparation job connects to the main `Emotion6` and `FI` image jobs
- which files should be pushed back to remote after implementing the preparation workflow
