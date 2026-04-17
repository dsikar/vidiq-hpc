# Phase 1 Audit: Qwen Fine-Tuned Run Integration

**Reviewer:** claude  
**Date:** 2026-04-17  
**Plan:** `reports/qwen-finetune-integration-plan.md`  
**Reviews considered:** claude, gemini, codex

---

## Overall Verdict

**Not approved.** Two blocking issues must be resolved before this can be committed.

---

## What Was Implemented

Phase 1 only — correct. No Phase 2 or Phase 3 work was introduced.

Files created:
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/config.json` ✓
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/artifacts.json` ✓
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/run-metadata.json` ✓
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/progress.json` ✓
- `experiments/text/multiclass/dair-ai-emotion/reports/qwen-finetune-first-findings.md` ✓
- `experiments/text/multiclass/dair-ai-emotion/reports/multiclass-emotion-dataset-findings.md` (updated) ✓

---

## Blocking Issues

### 1. Key evaluation metric not recorded in any bridge file

The plan specifies (Section "Proposed Target Structure", item 1) that the bridge files should contain:

> key metrics: eval accuracy, dataset size, label schema

Of the four bridge files:

- `config.json` records `dataset_size: 4038` and the label schema, but **not** `eval_accuracy`
- `progress.json` records only status flags (`completed`, `finetuned-classifier-bridge`) with no numeric outcome
- `run-metadata.json` records no metrics
- `artifacts.json` records only file paths

The headline result (`accuracy: 0.9804`) is only accessible by following the artifact pointer to `experiments/text_model/runs/tmqb0010_17763/analysis/eval_metrics.json`. A reader inspecting the bridge run folder cannot determine what the run achieved without leaving the dataset-level tree — which directly violates acceptance criterion 1:

> a reader inspecting only `experiments/text/multiclass/dair-ai-emotion/` can tell that a fine-tuned Qwen stage exists

The existence is visible; the result is not.

**Required fix:** add `eval_accuracy` and `eval_dataset_size` to `progress.json` or to a `metrics/summary.json` file. The audit prompt explicitly lists `metrics/summary.json` as an expected file; the implementation substituted a sparse `progress.json` without covering the same ground.

### 2. `metrics/summary.json` absent, `progress.json` is a status flag only

The audit prompt lists `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/metrics/summary.json` as a file to inspect. It does not exist.

`progress.json` was created instead, but its content is:

```json
{
  "completed_at_utc": "2026-04-17T20:54:49Z",
  "source_run_id": "tmqb0010_17763",
  "status": "completed",
  "stage": "finetuned-classifier-bridge"
}
```

This records completion status, not a metrics summary. It does not replace `metrics/summary.json`. The plan listed both `progress.json` and `summary.json` as options; given that neither epoch-level metrics nor eval accuracy appear anywhere in the bridge, one of these files must be augmented to record the headline numbers.

---

## Non-Blocking Issues

### 3. Mixed path styles across bridge files

`config.json` and `artifacts.json` use directory-relative paths (e.g., `../../../../../text_model/runs/tmqb0010_17763`). `run-metadata.json` uses a repo-root-relative path (`experiments/text_model/runs/tmqb0010_17763`). Both styles are internally consistent, but mixing them across files in the same bridge folder is a maintenance hazard for any tooling that resolves paths relative to file location.

**Recommendation:** pick one convention for the bridge folder and apply it uniformly. Repo-root-relative is more human-readable; directory-relative is safer for repo moves. Either is acceptable; mixing is not.

### 4. Codex-recommended validation step not implemented

Codex recommended a simple check that every file referenced by the bridge metadata actually exists. This was listed as an additional Phase 1 requirement. It was not implemented. Not blocking for the structural review, but worth noting before Phase 2 begins.

---

## Verification Summary

| Check | Result |
|---|---|
| Phase 1 only implemented | ✓ |
| Correct dataset-root path used everywhere (`dair-ai-emotion/`) | ✓ |
| `joy` vs `happiness` mismatch documented | ✓ (config.json comparison_note, findings report, main findings) |
| Bridge uses lightweight metadata only | ✓ (no `.npy` duplication) |
| Artifact paths point correctly to source run | ✓ (relative depth verified: 5 levels up from bridge → `experiments/`) |
| Findings report accurately describes run and limitations | ✓ |
| Main findings report integrates fine-tuned stage | ✓ |
| Headline metric (`0.9804`) visible from bridge folder | ✗ |
| `metrics/summary.json` present | ✗ |
| Path style consistent across bridge files | ✗ |

---

## Final Recommendation

**Not approved.**

Fix the two blocking issues:

1. Add the eval accuracy (`0.9804`) and dataset size (`4038`) to the bridge — either by creating `metrics/summary.json` or by expanding `progress.json` to include them.
2. Resolve the path-style inconsistency across bridge files (pick repo-root-relative or directory-relative, apply uniformly).

Once those are addressed, Phase 1 is structurally sound and the work can be committed.
