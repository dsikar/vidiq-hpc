# Recommendation: Qwen Fine-Tuned Run Integration Plan

**Author:** Codex  
**Date:** 2026-04-17  
**Plan reviewed:** `reports/qwen-finetune-integration-plan.md`  
**Peer reviews considered:**

- `reports/qwen-finetune-integration-plan-review-claude.md`
- `reports/qwen-finetune-integration-plan-review-gemini.md`

## Decision

**Conditional approval for Phase 1 only.**

The integration work should **not** begin exactly as written in the current plan. The plan is directionally correct, but two issues are load-bearing enough that they must be fixed before implementation:

1. path inconsistency in the plan
2. label-schema mismatch between the BGE stage and the Qwen fine-tuned stage

Once those are corrected in the plan, **Phase 1 may proceed**.

## What I Agree With

The core architecture is correct:

- keep `experiments/text_model/` as the execution / training layer
- expose the fine-tuned stage inside `experiments/text/multiclass/dair-ai-emotion/` as the interpretation / reporting layer
- avoid duplicating large `.npy` arrays and model artifacts
- use lightweight bridge metadata rather than moving or copying the raw run

This is the right tradeoff for the current repo state and the paper timeline.

## Blocking Issues

### 1. Fix the path inconsistency first

The plan mixes:

- `experiments/text/multiclass/dair-ai-emotion/`  
with
- `experiments/text/multiclass/dair-ai/emotion/`

The latter path does not match the established dataset root. If implementation starts from the current plan text, there is a real risk of creating a second disconnected directory tree.

**Required fix:** update all references in the plan to the existing dataset root:

- `experiments/text/multiclass/dair-ai-emotion/`

### 2. Handle the label mismatch explicitly

The BGE-stage reports use:

- `sadness, joy, love, anger, fear, surprise`

The Qwen run metadata uses:

- `anger, fear, happiness, love, sadness, surprise`

The scientifically important mismatch is:

- `joy` vs `happiness`

This is not a cosmetic issue. Any comparison across stages, especially centroid- or class-level geometry comparison, will be wrong or ambiguous unless the label mapping is made explicit.

**Required fix:** the plan must state that:

- Phase 1 documentation must explicitly record the label schema used by the Qwen run
- the bridge metadata must include the exact label mapping
- any cross-stage comparison must either map `joy -> happiness` explicitly or explain why they should not be treated as identical

## Phase Approval

### Phase 1

**Approved after the two blocking fixes above are made.**

Phase 1 is low risk and worthwhile:

- add a dataset-level bridge run folder
- add a dedicated fine-tuned findings report
- update the main multiclass findings report

This resolves the discoverability problem without changing the training system.

### Phase 2

**Not approved yet, but appropriate immediately after Phase 1 if the updated plan is accepted.**

The plotting work is structurally sensible, but it should be gated on the dataset-level bridge existing first. Otherwise the plots will be generated without a stable dataset-local destination and the repo will remain split.

### Phase 3

**Deferred until the label-schema issue is explicitly resolved.**

Phase 3 is the scientifically sensitive stage. It involves logit/geometry interpretation and potentially comparison with earlier BGE-stage results. That work should not begin until:

- the label mapping is documented
- the comparison semantics are explicit
- the findings report states what is and is not directly comparable

## Additional Requirements I Recommend Adding

Before Phase 1 implementation begins, the plan should also add these requirements:

1. Bridge files should use repository-relative paths where practical, not machine-specific absolute paths.
2. The bridge metadata should record:
   - source run id
   - exact training dataset path
   - exact label schema used in training
   - eval split type and size
3. The fine-tuned findings report should state clearly that the 98.0% accuracy comes from a held-out split of the balanced CSV workflow, not from the earlier BGE validation setup.
4. A simple validation step should be included:
   - verify that every file referenced by the bridge metadata exists

## Final Recommendation

Proceed as follows:

1. revise the plan text to fix the path inconsistency
2. revise the plan text to make the `joy` / `happiness` label issue explicit
3. add the extra Phase 1 requirements above
4. once that is done, proceed with **Phase 1 only**
5. hold **Phase 2** until Phase 1 lands cleanly
6. hold **Phase 3** until label alignment is formally resolved

## Short Form

**Go forward with Phase 1 after plan fixes. Do not proceed with Phase 2 or Phase 3 yet.**
