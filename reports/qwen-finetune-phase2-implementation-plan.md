# Qwen Fine-Tuned Run Phase 2 Implementation Plan

## Peer Review Request

If you are reviewing this plan, please write your review as a separate markdown file in the same directory:

- `reports/`

Use a filename that includes your reviewer name, for example:

- `qwen-finetune-phase2-plan-review-<reviewer-name>.md`

Examples:

- `qwen-finetune-phase2-plan-review-claude.md`
- `qwen-finetune-phase2-plan-review-gemini.md`

Please include:

- whether you agree or disagree with the proposed Phase 2 approach
- any structural or scientific risks
- whether Phase 2 should proceed now
- any missing acceptance criteria
- whether any parts of this plan should be delayed or split further

This Phase 2 plan should not be executed until those peer reviews have been read and assessed.

---

## Purpose

This document defines **Phase 2** of the Qwen fine-tuned run integration.

Phase 1 has already established the dataset-level bridge structure this phase depends on:

- the raw training run remains under `experiments/text_model/runs/tmqb0010_17763/`
- a dataset-level bridge run now exists under `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/`
- the fine-tuned stage is now represented in the dataset-level findings reports

Phase 2 is limited to:

- dataset-level plotting
- dataset-level plot metadata
- lightweight documentation updates needed to make those plots discoverable

Phase 2 does **not** include:

- new training runs
- changes to the HPC training workflow
- deeper logit-geometry interpretation
- cross-stage scientific claims beyond descriptive plotting

Those remain out of scope until a later phase.

---

## Current Situation

The completed Qwen run already provides the raw ingredients for visualization:

- `train_metrics.json`
- `analysis/eval_embeddings.npy`
- `analysis/eval_logits.npy`
- `analysis/centroids.json`
- `analysis/eval_metrics.json`

The dataset-level bridge run points to those artifacts, but the dataset-level experiment root still lacks:

- plot files derived from the Qwen run
- a plot metadata summary
- a standard location for fine-tuned-stage visual outputs

As a result, the fine-tuned stage is now narratively integrated, but not visually integrated.

---

## Goal

Make the fine-tuned Qwen stage visually consistent with the rest of the multiclass experiment root by adding publication-facing plots and lightweight metadata under:

- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/`

This should let a reader inspect the multiclass dataset root and immediately find:

- the fine-tuned run summary
- the fine-tuned findings note
- the fine-tuned plots

without having to browse the raw `experiments/text_model/` run directory.

---

## Proposed Outputs

Phase 2 should create:

### Plot directory

- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/`

### Core plot files

- `train-loss-vs-epoch.png`
- `eval-accuracy-vs-epoch.png`
- `eval-embeddings-pca-2d.png`
- `eval-embeddings-tsne-2d.png`
- `centroid-distance-heatmap.png`

### Plot metadata

- `projection-summary.json`

This JSON should describe:

- source run id
- source files used
- number of evaluation points
- embedding dimensionality
- class labels used
- plotting configuration
- generated output file names

---

## Why These Plots

### 1. Training loss vs epoch

Purpose:

- show that optimization behaved sensibly across the 10 epochs
- make the training trajectory visible at dataset level

Source:

- `experiments/text_model/runs/tmqb0010_17763/train_metrics.json`

### 2. Evaluation accuracy vs epoch

Purpose:

- show when the model saturated
- contextualize the headline `0.9804` result

Source:

- `experiments/text_model/runs/tmqb0010_17763/train_metrics.json`

### 3. PCA projection of evaluation embeddings

Purpose:

- give a low-dimensional visual summary of the learned embedding organization
- provide direct visual comparison with earlier multiclass projection plots

Source:

- `experiments/text_model/runs/tmqb0010_17763/analysis/eval_embeddings.npy`

### 4. t-SNE projection of evaluation embeddings

Purpose:

- provide a nonlinear local-neighborhood view of the fine-tuned embeddings
- complement PCA rather than replace it

Source:

- `experiments/text_model/runs/tmqb0010_17763/analysis/eval_embeddings.npy`

### 5. Centroid-distance heatmap

Purpose:

- expose the class-centroid structure directly from the fine-tuned run
- make the supervised stage comparable at a descriptive level with the existing BGE centroid summaries

Source:

- `experiments/text_model/runs/tmqb0010_17763/analysis/centroids.json`

Implementation note:

- `centroids.json` contains one centroid vector per class
- it does **not** contain a precomputed pairwise distance matrix
- the Phase 2 plotting script must therefore:
  1. load centroid vectors from `centroids.json`
  2. load class labels from `run_metadata.json`
  3. compute the pairwise centroid-distance matrix explicitly
  4. render the heatmap from that computed matrix

---

## Structural Recommendation

### Keep plotting code separate from the HPC training entrypoint

Do **not** add plotting directly into:

- `experiments/text_model/train_multiclass.py`

Reason:

- the training script should remain responsible for training and raw artifact export
- Phase 2 is about dataset-level presentation
- keeping plotting separate reduces coupling and avoids retraining just to regenerate plots

### Add a dedicated plotting script under the dataset experiment root

Recommended new script:

- `experiments/text/multiclass/dair-ai-emotion/src/plot_qwen_finetune_run.py`

Responsibilities:

- read the bridge run metadata
- load the source artifacts from the text-model run
- generate the Phase 2 plots
- write them into the dataset-level artifacts tree
- write `projection-summary.json`

Implementation requirements:

- resolve bridge artifact paths relative to the bridge run directory, not the current working directory
- derive class labels from bridge/source metadata, not from a hardcoded constant
- preserve the existing `joy` / `happiness` caveat in metadata and documentation
- keep plot styling aligned with the existing multiclass plotting conventions where practical

This mirrors the existing pattern already used for multiclass visualization scripts.

---

## Input Contract

The plotting script should accept either:

1. a bridge run directory
2. or a source text-model run id

Preferred default:

- bridge run directory

Example target:

- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/`

Reason:

- Phase 1 established the bridge as the dataset-level source of truth for integration
- Phase 2 should build on that instead of bypassing it

Path-resolution rule:

- any relative path stored in bridge metadata must be resolved relative to the bridge run directory itself

---

## Required Safeguards

### 1. No artifact duplication

The plotting step must not copy:

- `eval_embeddings.npy`
- `eval_logits.npy`

into the dataset-level `artifacts/` tree.

Only derived plots and lightweight metadata should be written there.

### 2. Explicit label-schema warning

Any metadata written by the plotting stage must preserve the existing warning:

- Qwen stage uses `happiness`
- earlier BGE-stage materials use `joy`

Phase 2 plots are descriptive only. They must not claim direct classwise equivalence to the earlier BGE stage without explicit mapping.

Label-loading rule:

- class labels for Phase 2 plots must be read from bridge/source metadata
- do **not** reuse the hardcoded label order from earlier BGE-only plotting code

### 3. Re-runnable without retraining

The plotting script must be usable as a pure post-processing step.

If a plot style changes, we should be able to regenerate outputs from existing tracked artifacts only.

### 4. Graceful plotting fallbacks

If an optional plotting dependency is missing, the script should either:

- fall back to an available alternative
- or skip the affected plot cleanly with an explicit message

Phase 2 should not require retraining or manual artifact rewriting just because one optional plotting backend is unavailable.

### 5. Stable output paths

All outputs for this stage should be written into:

- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/`

Do not spread Qwen fine-tuned plots across multiple unrelated artifact folders.

---

## Documentation Changes

Phase 2 should update, at minimum:

- `experiments/text/multiclass/dair-ai-emotion/README.md`

to mention:

- the bridge run
- the dedicated Qwen fine-tuned plotting stage
- the output plot directory

Optionally update:

- `experiments/text/multiclass/dair-ai-emotion/reports/qwen-finetune-first-findings.md`

to reference the new plot files after they exist.

---

## What Phase 2 Must Not Do

### Do not introduce scientific comparison claims

Phase 2 may generate plots that visually invite comparison with the BGE stage, but it must not claim:

- improved geometry
- tighter clustering
- better class separation
- better interpretability

unless those claims are backed by explicit analysis in a later phase.

### Do not add logit-distance correlation yet

Even though logits already exist, Phase 2 should stop at descriptive plotting.

Any logit-geometry correlation work belongs to the later, explicitly analytical phase.

### Do not alter the bridge run schema unless necessary

Phase 1 has already created the bridge metadata. Phase 2 should consume that structure, not redesign it.

---

## Proposed Implementation Order

1. Add `src/plot_qwen_finetune_run.py`.
2. Add the output directory:
   - `artifacts/plots/qwen-finetune-10e/`
3. Generate:
   - training loss plot
   - evaluation accuracy plot
   - PCA plot
   - t-SNE plot
   - centroid heatmap
4. Write `projection-summary.json`.
5. Update the dataset-level README.
6. Optionally update the fine-tuned findings report with links to the generated plots.

---

## Acceptance Criteria

Phase 2 should be considered successful when:

1. the dataset-level plot directory exists:
   - `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/`
2. all five core plots are generated there
3. `projection-summary.json` is generated there
4. no large raw `.npy` files are duplicated into the dataset-level artifacts tree
5. the plotting script can regenerate the outputs from existing tracked artifacts
6. the dataset-level README mentions the fine-tuned plotting stage
7. the label-schema caveat remains visible in the metadata or report layer
8. the centroid heatmap is computed from centroid vectors rather than assumed to exist as a precomputed matrix
9. the centroid heatmap uses label names on both axes, not integer ids
10. the plotting script resolves bridge-relative artifact paths correctly
11. the plot metadata records the exact label list used for the Qwen stage

---

## Immediate Recommendation

Proceed to audit this Phase 2 plan first.

If approved, implement Phase 2 exactly as a plotting-and-documentation stage only.
