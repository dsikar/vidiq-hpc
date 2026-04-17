# Review: Qwen Fine-Tuned Run Phase 2 Implementation Plan

**Reviewer:** claude  
**Date:** 2026-04-17  
**Plan reviewed:** `reports/qwen-finetune-phase2-implementation-plan.md`

---

## Overall Assessment

The Phase 2 strategy is correct in scope and structure. Keeping the plotting as a separate post-processing script, writing outputs into the dataset-level artifacts tree, and not duplicating large arrays are all right calls. The plan is also well-bounded: it explicitly excludes scientific comparison claims and logit-geometry analysis.

However, there are two blocking issues that must be resolved before implementation, and two non-blocking issues worth addressing in the implementation spec.

---

## Should Phase 2 Proceed Now?

**No, not yet.** Two gating conditions are unmet.

---

## Blocking Issues

### 1. Phase 1 prerequisite is not met

The plan states in its opening section:

> Phase 1 is already complete.

This is incorrect. The Phase 1 audit (`reports/qwen-finetune-phase1-audit-claude.md`) issued a verdict of **Not approved** with two blocking issues:

- headline eval metric (`0.9804`) not recorded in any bridge file
- `metrics/summary.json` absent from the bridge run; `progress.json` is a status flag only, not a metrics summary

Phase 2 should not begin until Phase 1 is cleared. The Phase 2 plotting script is designed to read from the bridge run folder as its primary input. If the bridge is incomplete, the script will either fail or silently omit the headline result from its metadata output.

### 2. `centroids.json` schema not documented; required pre-processing missing from the plan

The plan lists the centroid heatmap source as:

> `experiments/text_model/runs/tmqb0010_17763/analysis/centroids.json`

The actual file contains one entry per class (integers 0–5), each carrying the full raw embedding vector in the native model dimension (~1536 floats). It does **not** contain a pairwise distance matrix.

In contrast, the existing `plot_bge_variants.py` reads a pre-computed `centroid_cosine_distance_matrix` (6×6) from `metrics/summary.json`:

```python
np.array(metrics_summary["geometry"]["centroid_cosine_distance_matrix"])
```

That key does not exist in the Qwen run artifacts. To produce the centroid heatmap, the Phase 2 script must:

1. load the six centroid vectors from `centroids.json`
2. compute pairwise cosine distances (scipy or manual)
3. map class integers to label names using `run_metadata.json` (the file that holds `label_to_id`)

None of this pre-processing is mentioned in the plan. An implementer following the plan as written will discover the schema mismatch only when the code fails. The plan must document the required steps before Phase 2 implementation begins.

---

## Non-Blocking Issues

### 3. Label names must be read from metadata, not hardcoded

The existing `plot_bge_variants.py` hardcodes:

```python
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
```

The new script will need `["anger", "fear", "happiness", "love", "sadness", "surprise"]` (the Qwen training order from `label_to_id`). These must not be hardcoded and must be derived at runtime from `run_metadata.json` or `config.json`. If a developer uses the BGE script as a template and copies the hardcoded list, the heatmap and scatter plot labels will be wrong.

**Recommendation:** add an explicit note to the Phase 2 plan that label names must come from the bridge config or source run metadata, not from a hardcoded constant.

### 4. Relative path resolution in `artifacts.json` requires explicit handling

The bridge `artifacts.json` uses directory-relative paths:

```
"../../../../../text_model/runs/tmqb0010_17763/analysis/eval_embeddings.npy"
```

These paths are relative to the `artifacts.json` file itself, not to the script's working directory. The Phase 2 plan says "the plotting script should accept a bridge run directory as input" but does not specify that path resolution must be done relative to the bridge directory. This is easy to get wrong. The plan should state this explicitly, or the implementation spec should note it.

---

## Missing Acceptance Criteria

The existing list (items 1–7) is reasonable, but add:

8. The `projection-summary.json` must include the exact label list used (with the `happiness`/`joy` note preserved) so that any downstream script can verify it is reading Qwen-stage outputs and not BGE-stage outputs.
9. The centroid heatmap must use label names, not class integers, on both axes.

---

## Summary

The Phase 2 plan is directionally sound. Fix the two blocking issues — clear Phase 1 first, and document the centroid pre-processing step — then Phase 2 is safe to implement. The structural recommendation (separate plotting script, stable output path, no artifact duplication) is correct and should be preserved exactly as written.
