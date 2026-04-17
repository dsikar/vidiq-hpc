# Qwen Fine-Tuned Run Phase 3 Implementation Plan

## Peer Review Request

If you are reviewing this plan, please write your review as a separate markdown file in the same directory:

- `reports/`

Use a filename that includes your reviewer name, for example:

- `qwen-finetune-phase3-plan-review-<reviewer-name>.md`

Examples:

- `qwen-finetune-phase3-plan-review-claude.md`
- `qwen-finetune-phase3-plan-review-gemini.md`

Please include:

- whether you agree or disagree with the proposed Phase 3 scope
- any scientific or structural risks
- whether the plan is ready to implement now
- any missing safeguards, validation checks, or acceptance criteria
- whether any part of the proposed comparison logic should be delayed further

This Phase 3 plan should not be executed until those peer reviews have been read and assessed.

---

## Purpose

This document defines **Phase 3** of the Qwen fine-tuned run integration.

Phase 1 made the fine-tuned run discoverable from the dataset-level experiment root.

Phase 2 made the fine-tuned run visually discoverable by adding dataset-level plots under:

- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/`

Phase 3 is the first explicitly analytical stage. Its job is to test whether the fine-tuned model's raw pre-softmax scores are aligned with the embedding geometry exported by the same run.

This is scientifically sensitive work. The implementation therefore needs stricter scope control than Phase 1 or Phase 2.

---

## Scope

Phase 3 is limited to:

- within-run logit-geometry analysis for the completed Qwen fine-tuned run
- dataset-level analytical outputs derived from tracked source artifacts
- dataset-level metrics, plots, and a findings note
- an implementation structure that can be reused for later fine-tuned runs without reshaping the repo again

Phase 3 does **not** include:

- new training runs
- changes to the HPC training workflow
- copying large arrays into the dataset-level artifacts tree
- claims that the Qwen stage is "better" than the earlier BGE stage
- direct cross-stage quantitative comparison unless the `joy` / `happiness` issue is handled explicitly and conservatively

The core analytical question is:

- do raw Qwen logits track native-space geometric proximity to class centroids on the held-out evaluation split

---

## Preconditions

Phase 3 should assume the following already exist:

- source run:
  - `experiments/text_model/runs/tmqb0010_17763/`
- dataset-level bridge run:
  - `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/`
- Phase 2 plot directory:
  - `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/`

The plan also depends on these facts remaining explicit:

- the Qwen run uses `happiness`
- earlier BGE materials use `joy`
- those labels must not be silently merged in any comparative analysis

---

## Current Situation

The completed fine-tuned run already exposes the raw ingredients needed for Phase 3:

- `analysis/eval_embeddings.npy`
- `analysis/eval_logits.npy`
- `analysis/centroids.json`
- `analysis/eval_metrics.json`
- `run_metadata.json`
- `train_metrics.json`

Phase 2 already added presentation-layer plots and `projection-summary.json`, but those outputs are descriptive rather than analytical.

What is still missing is a dataset-level answer to questions like:

- whether the nearest centroid usually matches the highest logit
- whether the true-class logit falls as true-centroid distance grows
- whether ambiguous classes show weaker geometry/logit agreement than cleaner classes
- whether the run provides evidence of geometry-confidence alignment rather than just high accuracy

---

## Goal

Produce a reusable Phase 3 analysis layer that:

1. loads the bridge run and source run artifacts
2. reconstructs the evaluation labels in a reproducible way
3. computes native-space distances from each evaluation embedding to every class centroid
4. compares those distances against the raw pre-softmax logits for the same examples
5. writes metrics and publication-facing plots into the dataset-level artifacts tree
6. writes a short findings note under the dataset-level reports tree

The key design requirement is that future runs should slot into the same structure cleanly.

In practice that means the implementation should be parameterized by bridge run path or run id, rather than hardcoded specifically for `tmqb0010_17763`.

---

## Proposed Outputs

### New analysis script

- `experiments/text/multiclass/dair-ai-emotion/src/analyze_qwen_logit_geometry.py`

This should be a post-processing script, not part of the training entrypoint.

### Metrics directory

- `experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/qwen-finetune-10e/`

Expected files:

- `logit-geometry-summary.json`
- `per-class-logit-distance-correlations.json`
- `nearest-centroid-vs-prediction.json`
- `distance-rank-agreement.json`

### Plot directory

- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/logit-geometry/`

Expected files:

- `true-class-logit-vs-distance.png`
- `predicted-class-logit-vs-distance.png`
- `distance-margin-vs-logit-margin.png`
- `rank-correlation-by-class.png`
- `nearest-centroid-confusion-heatmap.png`

### Findings report

- `experiments/text/multiclass/dair-ai-emotion/reports/qwen-finetune-logit-geometry-findings.md`

This report should summarize the Phase 3 results conservatively and should state clearly what is descriptive evidence versus what would require further runs.

### Output schema requirements

The output JSON files should have stable top-level keys before implementation begins.

`logit-geometry-summary.json` should include at least:

- `bridge_run`
- `source_run_id`
- `source_files`
- `label_mapping`
- `evaluation`
- `distance_metrics`
- `agreement_metrics`
- `significance`
- `generated_files`
- `notes`

`per-class-logit-distance-correlations.json` should include at least:

- `label_names`
- `primary_metric`
- `secondary_metric`
- `per_class`
- `global`

Each entry in `per_class` should include at least:

- `label`
- `count`
- `spearman_r`
- `spearman_p`
- `true_class_logit_distance_correlation`
- `outlier_count`

`nearest-centroid-vs-prediction.json` should include at least:

- `label_names`
- `counts_matrix`
- `normalized_matrix`
- `overall_match_rate`
- `per_class_match_rate`

`distance-rank-agreement.json` should include at least:

- `label_names`
- `primary_metric`
- `mean_spearman_r`
- `mean_spearman_p`
- `per_class_mean_spearman_r`
- `per_class_mean_spearman_p`
- `margin_summary`

---

## Data Sources

The Phase 3 analysis should read:

- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/artifacts.json`
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/run-metadata.json`
- `experiments/text_model/runs/tmqb0010_17763/analysis/eval_embeddings.npy`
- `experiments/text_model/runs/tmqb0010_17763/analysis/eval_logits.npy`
- `experiments/text_model/runs/tmqb0010_17763/analysis/centroids.json`
- `experiments/text_model/runs/tmqb0010_17763/analysis/eval_metrics.json`
- `experiments/text_model/runs/tmqb0010_17763/run_metadata.json`

Important requirements:

- resolve source artifact paths relative to the bridge run directory, not the current working directory
- derive class labels from metadata, not from a hardcoded label list
- use raw logits exactly as exported; do not apply softmax in the analysis path
- work in native embedding space; do not compute primary metrics from PCA or t-SNE coordinates
- write source file paths in Phase 3 JSON outputs as repo-relative paths, not machine-specific absolute paths

---

## Distance Metric Policy

The Phase 2 centroid heatmap used cosine distance. Phase 3 must not ignore that fact.

Phase 3 should therefore compute **both**:

- Euclidean distance in native embedding space
- cosine distance in native embedding space

Primary analysis metric:

- Euclidean distance

Reason:

- Phase 3 is trying to test whether raw pre-softmax logits track not only angular similarity but also native-space separation magnitude
- the embeddings and logits are both exported from the fine-tuned run without a normalization step that would make magnitude disposable

Secondary comparison metric:

- cosine distance

Reason:

- this maintains continuity with the published Phase 2 centroid heatmap
- it allows reviewers to see whether the apparent logit-geometry relationship depends on metric choice

Implementation requirement:

- `logit-geometry-summary.json` must explicitly record that Phase 2 used cosine for the centroid heatmap while Phase 3 uses Euclidean as primary and cosine as secondary
- if the qualitative conclusion differs materially between Euclidean and cosine, the findings note must state that clearly

---

## Required Label Handling

The most failure-prone part of Phase 3 is label handling.

The implementation should:

1. read `label_to_id` and `id_to_label` from source metadata where available
2. reconstruct evaluation labels using the same split logic used by the training run
3. verify that reconstructed evaluation-set size matches the saved artifact counts
4. verify that reconstructed per-class counts are plausible for the recorded split configuration and balanced CSV assumption
5. record the exact `id_to_label` mapping used in the analysis outputs
6. fail loudly if any label mismatch or count mismatch appears

The implementation must **not**:

- hardcode `anger, fear, happiness, love, sadness, surprise`
- silently substitute `joy` for `happiness`
- compare Qwen outputs against BGE outputs by assuming the label sets are identical

---

## Analytical Procedure

### 1. Reconstruct evaluation labels

Use the same dataset path, seed, and split configuration recorded in the source run metadata.

This should mirror the training script's held-out split logic so that every row in:

- `eval_embeddings.npy`
- `eval_logits.npy`

has a reproducible ground-truth label.

Validation requirement:

- after reconstruction, compare the class-count distribution against the expected held-out distribution implied by the balanced CSV workflow
- if any class count is implausible for the recorded split, halt with a diagnostic message
- if a stable dataset fingerprint can be computed cheaply from the source CSV metadata or contents, record it in the summary JSON as an additional provenance check

### 2. Load centroid vectors and compute native-space distances

For every evaluation example:

- compute its distance to every class centroid

Primary metric:

- Euclidean distance in the native embedding space

Secondary metric:

- cosine distance in the native embedding space

The summary JSON should state which metric is primary and which is secondary.

### 3. Compare logits against distances

For each evaluation example compute:

- raw logit vector
- centroid-distance vector
- predicted class by argmax logit
- nearest-centroid class by minimum distance
- true-class logit
- true-class centroid distance
- logit margin:
  - top-1 logit minus top-2 logit
- distance margin:
  - second-smallest centroid distance minus smallest centroid distance

### 4. Aggregate agreement metrics

Compute at least:

- fraction of examples where predicted class equals nearest-centroid class
- fraction of examples where true class is also the nearest centroid
- mean Spearman rank correlation between:
  - descending logits
  - ascending negative distances
- corresponding Spearman p-values
- mean correlation broken down by ground-truth class
- class-wise p-values
- per-class summary of true-class logit vs true-class distance
- Euclidean-versus-cosine comparison summary for the main agreement metrics

### 5. Visualize the relationship

Plots should prioritize interpretation over visual novelty.

At minimum:

- `true-class-logit-vs-distance.png`
  - scatter or hexbin of true-class logit against true-centroid distance
- `predicted-class-logit-vs-distance.png`
  - same view for the predicted class
- `distance-margin-vs-logit-margin.png`
  - whether cleaner geometric separation aligns with more decisive logits
- `rank-correlation-by-class.png`
  - class-wise rank agreement summary
- `nearest-centroid-confusion-heatmap.png`
  - confusion between nearest-centroid assignment and predicted class

Outlier requirement:

- the analysis should identify strong disagreement cases where geometry and logits diverge sharply
- the findings report should summarize how outliers were defined and how many were found
- the summary JSON should include an outlier count and threshold note

The plot styling should follow the existing multiclass/Qwen visual language where practical.

---

## Structural Recommendation

### Keep the analysis separate from training

Do **not** add Phase 3 logic to:

- `experiments/text_model/train_multiclass.py`

Reason:

- Phase 3 is analytical post-processing
- it should be rerunnable without retraining
- future runs should reuse the same analysis flow after bridge creation

### Parameterize by bridge run

The new analysis script should accept either:

- a bridge run directory
- or a bridge run id that resolves to one

Reason:

- this makes future results straightforward to incorporate
- a new fine-tuned result should only need:
  1. a new source run under `experiments/text_model/runs/`
  2. a lightweight bridge run under dataset-level `runs/`
  3. a rerun of the same Phase 2 and Phase 3 scripts against that bridge

That is the correct long-term pattern for this repo.

---

## Comparative Logic: What Is Allowed Now

Phase 3 may include only limited, conservative comparison language.

Allowed:

- comparing Qwen internal agreement metrics against Qwen accuracy and Qwen centroid structure
- stating that Phase 3 extends Phase 2 from descriptive plotting into within-run geometry/logit analysis
- noting where class-specific ambiguity appears inside the Qwen run

Not allowed:

- claiming Qwen geometry is superior to BGE geometry
- directly ranking Qwen and BGE classes against each other as if label semantics are already harmonized
- treating `joy` and `happiness` as equivalent without an explicit mapping note and a justification

If any BGE reference appears in the findings note, it should be framed only as context and should explicitly restate the label caveat.

---

## Validation Requirements

Before Phase 3 is considered complete, the implementation should verify:

1. every bridge-referenced source file exists
2. `eval_embeddings.npy` row count equals `eval_logits.npy` row count
3. reconstructed evaluation-label count matches those row counts
4. reconstructed per-class counts are plausible for the recorded split configuration
5. centroid dimensionality matches embedding dimensionality
6. `eval_logits.npy` column count equals `num_labels`
7. label ids in centroids and metadata agree
8. source `.npy` files are readable before the main analysis begins
9. output directories are created only under the dataset-level artifacts tree
10. no new large arrays are duplicated into `artifacts/`

If any of these checks fails, the script should stop with a clear error.

---

## Acceptance Criteria

Phase 3 should be considered successfully implemented when:

1. the repo contains a dedicated Phase 3 analysis script under:
   - `experiments/text/multiclass/dair-ai-emotion/src/`
2. the script can regenerate the Phase 3 outputs from the bridge run without retraining
3. all output paths are dataset-level and do not duplicate raw source arrays
4. the summary JSON records:
   - source run id
   - bridge run id
   - source files used
   - primary distance metric
   - secondary distance metric
   - label names actually used
   - the exact `id_to_label` mapping used
   - evaluation set size
   - generated output files
5. summary JSON source file paths are repo-relative, not machine-specific absolute paths
6. the findings note describes the results conservatively, documents what cannot yet be concluded, and states how outliers were handled
7. the implementation makes no silent `joy` / `happiness` substitution
8. the script halts with a diagnostic message if any validation check fails
9. the analysis flow is reusable for later fine-tuned runs via bridge-run parameterization

---

## Implementation Order

1. add the Phase 3 analysis script
2. add validation checks for bridge paths, label reconstruction, and artifact dimensions
3. generate Phase 3 metric JSON files
4. generate Phase 3 plots
5. write the Phase 3 findings note
6. review the outputs for any claim drift before commit/push

---

## Immediate Recommendation

Proceed to audit this plan before implementation.

The main thing the audit should test is whether this Phase 3 scope is narrow enough to be scientifically defensible while still being useful.

If the reviewers agree, the implementation should proceed as:

- within-run Qwen logit-geometry analysis first
- no stronger cross-stage comparison until label alignment is explicitly handled
