Copy Pritish's image files from `~/Downloads` into a local repo images directory, create that directory if it does not already exist, then inspect the copied images and write a report describing the process shown in them.

## Goal

We need a simple, reproducible workflow that:

1. finds the relevant Pritish image files in `~/Downloads`
2. copies them into this repository under a local images directory
3. creates the target directory if it does not exist
4. analyzes the visual content of the copied images
5. writes a clearly named report in `reports/` describing the process presented in the images

## Required file operations

Create a local image directory if it does not already exist. Use:

- `images/`

If you need a subdirectory for clarity, use:

- `images/pritish/`

Do not leave the only copies in `~/Downloads`. The repo-local copies are the working source for the report.

## Image selection

Find the relevant Pritish image files in:

- `~/Downloads`

Use reasonable judgement to identify the intended files. Prefer files that are clearly part of one process sequence or one experiment/process explanation rather than unrelated screenshots.

If multiple candidate groups exist, choose the set that is most likely to be the process Pritish wants documented, and state the assumption in the report.

## Copy behavior

Copy the selected files into the local repo images directory.

Keep original filenames unless there is a strong reason to normalize them. If you rename anything, document the mapping in the report.

## Analysis task

After copying the files, inspect the copied images and describe the process they present.

The report should focus on:

- what overall process the images appear to document
- the order of steps shown
- any tools, interfaces, or artifacts visible in the sequence
- important decisions, transitions, or outputs visible in the images
- uncertainties or ambiguities where the screenshots are incomplete

Do not write a vague caption list. Synthesize the screenshots into a coherent process description.

## Report output

Write the report to `reports/` with a clear descriptive name. Use a filename along the lines of:

- `reports/15-pritish-image-process-report.md`

The first line of the report should state that it is the result of actioning this prompt.

## Report structure

The report should include:

- which files were copied from `~/Downloads`
- where they were copied to under `images/`
- the inferred process shown in the images
- a step-by-step summary of that process
- any assumptions, missing context, or ambiguities

## Deliverables

Return:

- the image files copied into the repo
- the destination image directory used
- the report path created under `reports/`
