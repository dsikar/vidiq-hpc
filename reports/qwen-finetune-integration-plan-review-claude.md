# Review: Qwen Fine-Tuned Run Integration Plan

**Reviewer:** claude  
**Date:** 2026-04-17  
**Plan reviewed:** `reports/qwen-finetune-integration-plan.md`

---

## Overall Assessment

The integration strategy is sound. The core idea — keep the HPC training artifacts where they are and add a lightweight reference layer inside the dataset experiment root — is the right tradeoff given the paper deadline and the existing repo layout. I agree with the approach.

---

## Issues Found

### 1. Path inconsistency in the plan itself (medium risk)

The plan alternates between two different paths for the dataset experiment root:

- `experiments/text/multiclass/dair-ai-emotion/` (correct — this is what exists)
- `experiments/text/multiclass/dair-ai/emotion/` (does not exist)

The incorrect two-level form appears in Sections "Proposed Target Structure" items 1–4 and "Proposed Implementation Order". Anyone implementing from the plan without checking the filesystem first will create a new disconnected tree under `dair-ai/emotion/` rather than writing into the existing `dair-ai-emotion/` root.

**Recommendation:** Fix all occurrences to `dair-ai-emotion` before Phase 1 begins.

### 2. Label schema mismatch between stages (low-to-medium risk)

The existing BGE findings report (`multiclass-emotion-dataset-findings.md`) lists six emotions as:

> sadness, joy, love, anger, fear, surprise

The Qwen run metadata (`run_metadata.json`) lists:

> anger, fear, **happiness**, love, sadness, surprise

`joy` in the BGE stage corresponds to `happiness` in the balanced CSV used for fine-tuning. These are not the same label string. Any report that compares the two stages, or any script that joins metrics by label name, will silently misalign on this class.

**Recommendation:** The fine-tuned findings report (Phase 1, item 2) must document this label name difference explicitly. The main findings report update (Phase 1, item 3) should note it in the new section. Phase 3 (geometry comparison across stages) cannot proceed without resolving or explicitly accounting for it.

### 3. 98% accuracy plausibility note missing from acceptance criteria

The plan acknowledges in the findings report outline that 98.0% accuracy "may be suspiciously high," but the acceptance criteria do not include any requirement to sanity-check this. The balanced CSV is a derived subset with equal class sizes, which makes the task easier than the original dair-ai/emotion distribution. A reader unfamiliar with this context will not know whether 98% is a genuine result or an artifact of the easier balanced setup.

**Recommendation:** Add to acceptance criteria: the findings report must state explicitly whether evaluation was performed on a held-out split from the same balanced CSV or from the original dataset distribution. If it is the former, the result is not directly comparable to BGE-stage metrics.

---

## Should Phase 1 Proceed Now?

Yes, with the path inconsistency fixed first. Phase 1 is low-risk structural work with no large-file movement and no changes to the training system. The path fix is a one-line correction in each affected section of the plan.

---

## Should Phase 2 and Phase 3 Be Delayed?

**Phase 2 (plotting):** Delay is reasonable. The plots are useful but not blocking. Proceed after Phase 1 is reviewed and accepted.

**Phase 3 (geometry comparison):** Delay is strongly recommended. The label schema mismatch (`joy` vs `happiness`) must be resolved before any cross-stage geometry comparison is attempted. Running Phase 3 with mismatched label names will produce misleading centroid-distance comparisons. This is the most scientifically sensitive phase and should not proceed until the label alignment issue is documented and handled.

---

## Missing Acceptance Criteria

The current list is structurally complete but lacks the following:

1. The bridge run folder's `config.json` must record the label schema actually used in training (not assumed from the dataset name).
2. The findings report must state which split was used for evaluation (balanced held-out vs original distribution).
3. The main findings update must explicitly name the `joy`/`happiness` label discrepancy between stages.

---

## Summary

Agree with the integration approach. Phase 1 is safe to proceed once the `dair-ai/emotion` vs `dair-ai-emotion` path inconsistency is corrected in the plan. Phase 3 should be held until the label schema mismatch is addressed.
