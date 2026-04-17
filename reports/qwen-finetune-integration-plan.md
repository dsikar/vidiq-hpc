# Qwen Fine-Tuned Run Integration Plan

## Peer Review Request

If you are reviewing this plan, please write your review as a separate markdown file in the same directory:

- `reports/`

Use a filename that includes your reviewer name, for example:

- `qwen-finetune-integration-plan-review-<reviewer-name>.md`

Examples:

- `qwen-finetune-integration-plan-review-codex-a.md`
- `qwen-finetune-integration-plan-review-claude.md`
- `qwen-finetune-integration-plan-review-gemini.md`

Please include:

- whether you agree or disagree with the proposed integration approach
- any structural risks or better alternatives
- whether Phase 1 should proceed now
- whether Phase 2 and Phase 3 should be delayed
- any missing acceptance criteria or implementation details

This integration plan should not be executed until those peer reviews have been read and assessed.

## Purpose

This note explains how the first completed fine-tuned Qwen multiclass run should be integrated into the existing `dair-ai/emotion` experiment structure without creating a second disconnected results tree.

This is a planning document only. It does not change the current experiment layout.

---

## Current Situation

We now have a completed balanced fine-tuning run:

- `experiments/text_model/runs/tmqb0010_17763/`

Tracked artifacts currently include:

- `analysis/centroids.json`
- `analysis/eval_embeddings.npy`
- `analysis/eval_logits.npy`
- `analysis/eval_metrics.json`
- `run_metadata.json`
- `train_metrics.json`

The run achieved:

- evaluation accuracy: `0.9804`
- evaluation set size: `4038`
- model: `Qwen/Qwen3-1.7B`
- data source: `balanced_emotions_6classes.csv`
- label schema: `anger, fear, happiness, love, sadness, surprise`

Important note:

- the earlier BGE-stage reports use `joy`, while the fine-tuned balanced CSV uses `happiness`
- any cross-stage comparison must either map `joy -> happiness` explicitly or explain why they should not be treated as identical

At the same time, the rest of the multiclass emotion work already has an established structure under:

- `experiments/text/multiclass/dair-ai-emotion/`

That dataset-specific root already contains:

- `reports/`
- `configs/`
- `artifacts/`
- `runs/`
- existing BGE embedding-stage run folders and summaries

---

## The Integration Problem

The fine-tuned Qwen result is scientifically part of the same `dair-ai/emotion` multiclass study, but structurally it sits outside that study root in:

- `experiments/text_model/`

This creates three problems:

1. **Split narrative**
   The BGE embedding-stage results live under the dataset-specific experiment root, but the fine-tuned classifier stage lives elsewhere.

2. **Split reporting**
   The existing reports in `experiments/text/multiclass/dair-ai-emotion/reports/` do not naturally “see” the fine-tuned run unless we manually cross-reference a second tree.

3. **Split plotting / downstream analysis**
   Existing multiclass plots and metrics are organised around the dataset-specific experiment root. The Qwen run produces useful artifacts, but not in the same shape or location.

The result is not wrong, but it is hard to maintain and hard to explain.

---

## Design Goal

We want one coherent experiment story:

- Stage 1: embedding-generation / embedding-ablation results
- Stage 2: geometry analysis and visualization
- Stage 3: fine-tuned classifier + logit/geometry analysis

All of those should be discoverable from the `dair-ai/emotion` experiment root, even if large training artifacts remain physically stored elsewhere.

---

## Recommendation

### Keep the raw training run where it is

Do **not** move the current Qwen training run out of:

- `experiments/text_model/runs/tmqb0010_17763/`

Reason:

- that directory already matches the training code
- it already has model/tokenizer archive handling
- it is already wired to the HPC workflow
- moving it would create unnecessary migration risk

### Add a dataset-level integration layer under `dair-ai-emotion`

The clean fix is to make the dataset-specific experiment root explicitly reference and summarise the fine-tuned stage.

That means:

1. keep `experiments/text_model/` as the training system of record
2. expose fine-tuned-stage metadata and summaries inside:
   - `experiments/text/multiclass/dair-ai-emotion/runs/`
   - `experiments/text/multiclass/dair-ai-emotion/reports/`
   - optionally `experiments/text/multiclass/dair-ai-emotion/artifacts/`

This preserves the existing HPC training workflow while making the dataset-specific experiment root the main analysis interface.

---

## Proposed Target Structure

### 1. Add a bridge run folder under the dataset experiment root

Create a new run folder such as:

- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/`

This folder should **not** duplicate the large arrays. It should contain lightweight integration metadata only:

- `config.json`
- `artifacts.json`
- `run-metadata.json`
- `progress.json` or `summary.json`

These files should point to or summarise:

- the source training run id: `tmqb0010_17763`
- model: `Qwen/Qwen3-1.7B`
- training regime: balanced CSV, 10 epochs, batch size 8
- key metrics: eval accuracy, dataset size, label schema
- the exact label mapping used in training
- paths to tracked artifacts in `experiments/text_model/runs/tmqb0010_17763/`

Implementation note:

- use repository-relative paths where practical, not machine-specific absolute paths

Purpose:

- the fine-tuned run now appears in the same `runs/` area as the other multiclass stages
- downstream reports can reference one dataset-local path
- no heavyweight duplication is introduced

### 2. Add a dedicated findings report for the fine-tuned stage

Create a new report such as:

- `experiments/text/multiclass/dair-ai-emotion/reports/qwen-finetune-first-findings.md`

This report should summarize:

- what the run was
- why it was done
- achieved evaluation accuracy
- what artifacts were produced
- what is still missing for full geometry interpretation

This is the natural place to discuss:

- whether 98.0% accuracy is meaningful or suspiciously high
- whether the balanced CSV task is easier than the earlier BGE geometry setup
- how this result compares conceptually with the BGE embedding-only stage

### 3. Update the main multiclass findings report

Update:

- `experiments/text/multiclass/dair-ai-emotion/reports/multiclass-emotion-dataset-findings.md`

Add a new section:

- `Fine-Tuned Qwen Stage`

That section should make the distinction explicit:

- BGE ablation stage: geometry-first embedding study
- Qwen fine-tuned stage: supervised classifier stage with trainable task head
- the `joy` / `happiness` label mismatch between stages

This keeps the repo’s main findings narrative in one place.

### 4. Add fine-tuned-stage plots into the dataset-level artifacts tree

The current Qwen run stores raw numeric analysis outputs, but no publication-facing plots yet.

The next plotting step should write into:

- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/`

Suggested initial plots:

- training loss vs epoch
- evaluation accuracy vs epoch
- PCA projection of `eval_embeddings.npy`
- t-SNE or UMAP projection of `eval_embeddings.npy`
- centroid-distance heatmap from `centroids.json`
- logit-vs-distance scatter plots once correlation analysis is added

Reason:

- all public-facing plots stay under the dataset experiment root
- the text-model run folder remains a raw run artifact store

---

## What Should Not Be Done

### Do not copy `eval_embeddings.npy` and `eval_logits.npy` into multiple places

Those arrays are already tracked once. Duplicating them inside `artifacts/` or another run folder adds maintenance cost and bloats the repo for no scientific benefit.

### Do not merge `experiments/text_model/` into `experiments/text/multiclass/dair-ai-emotion/` right now

That would be a deeper refactor of the training system and is not justified before the paper deadline.

### Do not rewrite the previous BGE run structure

The existing BGE run layout is already consistent enough. The Qwen integration should adapt to it, not force a broad reorganisation.

---

## Proposed Implementation Order

### Phase 1: lightweight integration

1. Create a dataset-level bridge run folder for `tmqb0010_17763`.
2. Add a fine-tuned findings report.
3. Update the main multiclass findings report with a new fine-tuned stage section.

This phase is low risk and makes the result discoverable immediately.

### Phase 2: plotting integration

4. Add a plotting script that reads:
   - `train_metrics.json`
   - `analysis/eval_embeddings.npy`
   - `analysis/eval_logits.npy`
   - `analysis/centroids.json`
5. Write plots into:
   - `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/`

This phase makes the run consistent with the existing reports/plots culture.

### Phase 3: deeper geometry integration

6. Add logit-geometry correlation outputs in the dataset-local artifacts tree.
7. Compare the fine-tuned embedding geometry against the BGE embedding-only baseline in one consolidated report.

This phase is the scientifically interesting one, but it should come after the structural cleanup.

---

## Why This Is the Best Tradeoff

This plan gives us:

- one coherent dataset-level story
- no duplication of large files
- no risky refactor of the training system
- clear separation between:
  - raw HPC training artifacts
  - dataset-level reports and plots

In other words:

- `experiments/text_model/` remains the execution layer
- `experiments/text/multiclass/dair-ai-emotion/` becomes the interpretation layer

That separation matches the way the repo is already evolving.

---

## Acceptance Criteria

The integration should be considered successful when:

1. a reader inspecting only `experiments/text/multiclass/dair-ai-emotion/` can tell that a fine-tuned Qwen stage exists
2. the fine-tuned run is represented in the dataset-level `runs/` area
3. the main multiclass findings report mentions the fine-tuned stage explicitly
4. plots derived from the fine-tuned run are written into the dataset-level artifacts tree
5. no large artifact duplication is introduced
6. the bridge metadata records the exact training label schema used by the Qwen run
7. the findings report states clearly that the 98.0% accuracy comes from a held-out split of the balanced CSV workflow
8. the main findings update explicitly documents the `joy` / `happiness` label mismatch

---

## Immediate Recommendation

Proceed with **Phase 1 only** first:

- add the bridge run folder
- add a dedicated fine-tuned findings report
- update the main multiclass findings report

Then let the audit review whether the repo should go further into Phase 2 and Phase 3.
