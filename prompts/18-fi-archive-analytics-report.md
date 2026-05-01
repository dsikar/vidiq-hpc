Write an analytics report that inspects the FI dataset archive on Hyperion and prepares the information needed for downstream staging and processing.

Repository root: `/users/aczd097/git/vidiq-hpc`

## Context

The FI archive is now present on Hyperion at:

- `/users/aczd097/archive/vidiq-hpc/data/image/fi/emotion_dataset.tar`

We do **not** want vague commentary. We need a concrete report that can be used to decide the exact downstream data-preparation steps required to turn this archive into the staged dataset layout expected by the repo.

The current repo expects the staged FI dataset root to look like:

```text
/users/aczd097/archive/vidiq-hpc/data/image/fi/
  metadata.csv
  images/
```

The current FI loader expects canonical labels:

- `amusement`
- `anger`
- `awe`
- `contentment`
- `disgust`
- `excitement`
- `fear`
- `sadness`

This task is intended to be executed on Hyperion against the real archive, and the resulting report is intended to be committed and pushed so it can later be pulled down to the local checkout for further analysis and decision-making.

## Files to inspect first

Before analyzing the archive, inspect:

- `reports/16-smaller-image-datasets-plan.md`
- `README.md`
- `scripts/prepare_image_dataset.py`
- `src/image_experiments/datasets.py`
- `configs/fi_phase1.json`
- `hpc/prepare_image_dataset.slurm`
- `hpc/image_embedding_fi.slurm`

## Task

Analyze the FI archive on Hyperion and write a report under `reports/` prefixed with `18-`, for example:

- `reports/18-fi-archive-analytics-report.md`

The report must be specific, operational, and based on actual archive evidence.

This is not just an ephemeral analysis task. The resulting report is meant to become a tracked repo artifact.

## What the report must contain

### 1. Archive structure summary

Describe the actual structure of `/users/aczd097/archive/vidiq-hpc/data/image/fi/emotion_dataset.tar`, including:

- top-level directory name
- class directory names
- whether the class names match the repo’s expected canonical labels
- whether there are non-image files present
- whether there is any existing metadata file in the archive

### 2. Quantitative inventory

Provide concrete counts, such as:

- total number of files
- total number of images
- image count per class
- whether any classes are empty
- whether any file extensions differ from expectations

If there are obvious irregularities, call them out.

### 3. Staging gap analysis

Compare the archive contents against the repo’s expected staged layout.

Be explicit about:

- what already exists in the archive that can be reused directly
- what is missing and must be generated
- whether `metadata.csv` must be created from directory structure
- whether images need to be flattened or copied into a unified `images/` tree
- whether the current loader could consume the extracted directory as-is or not

### 4. Downstream preparation requirements

State the exact downstream data work implied by the archive analysis.

For example:

- generate `metadata.csv`
- normalize image paths
- create `images/` directory in staged root
- preserve or transform class-folder structure
- validate label mapping against canonical FI labels

This section should make the next engineering task obvious.

### 5. Recommended processing plan

Recommend the concrete next step the repo should implement after this report.

That recommendation should be specific enough to become the next prompt or coding task.

Examples of acceptable outcomes:

- add a script that extracts the tar and generates `metadata.csv`
- extend `prepare_image_dataset.py` to support tar-based FI staging
- keep archive extraction separate from metadata generation

Do not implement the next step in this task; just analyze and report.

### 6. Hyperion command notes

Include the exact inspection commands you used on Hyperion so a future agent can reproduce the analysis.

### 7. End-to-end handoff workflow

The report must include the exact operational handoff sequence needed to get the archive-analysis script and its resulting report from local development through Hyperion execution and back to local analysis.

This section must be concrete and written as an explicit ordered procedure.

It must cover these exact stages:

1. commit the script locally
2. push to remote
3. pull from HPC
4. run on HPC
5. commit results on HPC
6. push back to remote
7. pull locally (here, the local machine)

For each stage, the report should state:

- what file(s) are involved
- what command shape should be used
- what output artifact is expected

The report outcome must explicitly state:

- the repo-relative report path
- the absolute report path on Hyperion

The absolute path is required because the next step will be to pull the committed change down locally and use that exact report for further decision making.

## Constraints

- Base the report on actual archive inspection, not assumptions.
- Do not extract the full archive unless that is genuinely necessary for analysis.
- Prefer header/listing analysis and lightweight counting first.
- Keep the report concrete and useful for engineering follow-up.
- Do not modify the dataset archive itself in this task.

## Deliverable

Produce:

- one report under `reports/` with the archive analytics and downstream staging implications

## Final output

When finished, report:

- the report file path
- the absolute path to that report on Hyperion
- the main structural findings
- the exact downstream preparation task you recommend next
