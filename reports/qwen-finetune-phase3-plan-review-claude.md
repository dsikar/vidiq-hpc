# Review: Qwen Fine-Tuned Run Phase 3 Implementation Plan

**Reviewer:** claude  
**Date:** 2026-04-17  
**Plan reviewed:** `reports/qwen-finetune-phase3-implementation-plan.md`

---

## Overall Assessment

The Phase 3 analytical scope is appropriate. The core question — do raw logits track geometric proximity to class centroids — is well-posed, the procedure is specific, and the guard rails against premature cross-stage comparison are correctly placed. The implementation order is also sensible.

Two blocking issues need resolution before implementation begins. Three non-blocking issues are worth addressing in the implementation spec.

---

## Precondition Check

Phase 1 and Phase 2 are complete:

- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/` — bridge run with `metrics/summary.json` present ✓
- `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/` — all five plots and `projection-summary.json` present ✓

The structural preconditions listed in the plan are met.

---

## Should Phase 3 Proceed Now?

**Not yet.** Fix the two blocking issues first.

---

## Blocking Issues

### 1. Distance metric inconsistency with Phase 2

Phase 3 proposes Euclidean distance as the primary centroid-proximity metric:

> Primary metric: Euclidean distance in the native embedding space

Phase 2 already computed and published centroid distances for the same run using cosine distance. This is recorded in `artifacts/plots/qwen-finetune-10e/projection-summary.json`:

```json
"centroid_distance_metric": "cosine"
```

Using Euclidean in Phase 3 and cosine in Phase 2 for the same artifacts creates an inconsistent analytical picture within a single run. A reader comparing the Phase 2 centroid heatmap against the Phase 3 logit-distance scatter plots will be looking at different distance semantics without any warning.

This may be intentional — cosine captures angular separation, Euclidean captures magnitude differences too, and in an unnormalized embedding space like Qwen's the latter can be informative. But the plan does not state this justification. As written, it looks like an oversight.

**Required fix:** the plan must either:
- state that Phase 3 deliberately uses Euclidean while Phase 2 used cosine, explain why, and note it in the `logit-geometry-summary.json` output; or
- align both phases on one metric.

The choice itself is defensible either way. The silence is not.

### 2. Label reconstruction validation is underpowered

The plan's validation check #3 requires:

> reconstructed evaluation-label count matches those row counts

A shuffled or reordered source CSV would pass this check while assigning every embedding and logit vector to the wrong ground-truth label. The correlation analysis would then run to completion and produce plausible-looking but wrong numbers.

The training script uses `StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on the full labels array, which is deterministic given stable CSV row order. CSV row order is generally stable, but the plan provides no protection against the case where it is not.

**Required fix:** add a class-distribution sanity check to validation. After reconstructing evaluation labels, verify that each class's count in the reconstruction is consistent with the expected class distribution (derivable from the total dataset size, test_size=0.2, and balanced CSV assumption). A mismatch in any class count should halt the script with a diagnostic message. This does not require ground-truth labels from outside the run — it only requires that the reconstructed distribution is plausible for a balanced 6-class 80/20 split.

---

## Non-Blocking Issues

### 3. `projection-summary.json` uses absolute machine-specific paths

Phase 2's `projection-summary.json` records source file paths as absolute machine-specific paths:

```json
"centroids": "/home/daniel/git/vidiq/vidiq-hpc/experiments/text_model/runs/tmqb0010_17763/analysis/centroids.json"
```

The Phase 3 plan correctly says to resolve paths from the bridge `artifacts.json` relative to the bridge directory. That is the right approach. However, the plan should also note that Phase 3 outputs should use repo-relative paths (not absolute paths) to avoid repeating the Phase 2 portability issue. The `logit-geometry-summary.json` should record paths as `experiments/text_model/runs/tmqb0010_17763/...` not `/home/daniel/...`.

### 4. `eval_logits.npy` column count not in validation checks

Validation check #4 verifies that centroid dimensionality matches embedding dimensionality. There is no check that `eval_logits.npy` has exactly `num_labels` (6) columns. A logit array with the wrong number of classes would produce silently wrong argmax predictions and corrupt the entire agreement analysis. Add it to the validation list:

> 4b. `eval_logits.npy` column count equals num_labels

### 5. Output JSON schemas are underspecified

The plan names four output JSON files but does not define their schemas:

- `nearest-centroid-vs-prediction.json`
- `distance-rank-agreement.json`
- `per-class-logit-distance-correlations.json`
- `logit-geometry-summary.json`

The analytical procedure section describes what to compute but does not map those computations to specific files. An implementer will have to decide which metrics go where. For a post-processing script intended to be reusable across future runs, schema stability matters. The plan should define at least the top-level keys for each output file before implementation begins.

---

## Science Assessment

The analytical procedure is well-structured:

- using raw logits (not softmax-squashed probabilities) is correct
- Spearman rank correlation is the right tool for logit-vs-distance ranking
- the logit margin vs distance margin comparison is well-defined and interpretable
- the within-run scope constraint is appropriate — no cross-stage claims before label alignment is resolved
- parameterization by bridge run path is the correct long-term pattern

The five proposed plots are all useful and non-redundant. The `nearest-centroid-confusion-heatmap.png` in particular will be informative for understanding where geometry and logit decisions diverge.

---

## Acceptance Criteria Assessment

The existing seven criteria are adequate with one addition:

8. The script halts with a diagnostic message if any validation check fails — not just if files are missing, but also if label reconstruction class distribution is implausible.

---

## Summary

Phase 3 is the right next step and the scope is correctly constrained. Fix the two blocking issues — clarify the Euclidean/cosine metric choice relative to Phase 2, and strengthen the label reconstruction validation — then the plan is ready to implement.
