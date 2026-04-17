# Peer Review: Qwen Fine-Tuned Run Phase 3 Implementation Plan

**Reviewer:** Gemini
**Date:** 2026-04-17

## 1. Agreement with Proposed Phase 3 Scope

I **strongly agree** with the proposed Phase 3 scope. 

The focus on **within-run logit-geometry analysis** is scientifically prudent. By establishing whether the fine-tuned model's internal representations (embeddings) and its classification head (logits) are geometrically aligned *before* attempting cross-model comparisons with BGE, we ensure a solid baseline for what "successful fine-tuning" looks like in this project's context.

The explicit exclusion of new training runs and large artifact duplication aligns with the repository's established hygiene standards.

## 2. Scientific or Structural Risks

### Scientific Risks
*   **Label Reconstruction Fragility:** The plan relies on reconstructing evaluation labels using `StratifiedShuffleSplit` with the same seed and test size. While this is standard, any discrepancy in how the CSV is read (e.g., encoding, line endings, or row filtering) between the training script and the analysis script could lead to a silent misalignment of labels and embeddings. 
    *   *Recommendation:* The analysis script should include a checksum or a small sample validation of the reconstructed labels against any metadata available in the source run.
*   **Metric Choice (Euclidean vs. Cosine):** While Euclidean distance is proposed as primary, Qwen (and most LLMs) often operate in a space where cosine similarity is more semantically meaningful, especially after fine-tuning. 
    *   *Recommendation:* The plan should explicitly require computing *both* and noting if the logit-geometry agreement differs significantly between the two.

### Structural Risks
*   **Bridge Path Dependency:** Like Phase 2, this phase is heavily dependent on the relative pathing between the bridge run and the source run. 
    *   *Recommendation:* The validation step should explicitly check for the readability of the `.npy` files before beginning the expensive correlation computations.

## 3. Readiness for Implementation

**Yes, the plan is ready to implement.** The preconditions are met (Phase 1 and 2 are complete), and the goals are clearly defined.

## 4. Missing Safeguards or Acceptance Criteria

*   **Correlation Significance:** Aggregating Spearman rank correlations is good, but the plan should also require reporting the **p-value** or some measure of statistical significance for these correlations to avoid over-interpreting noise in smaller classes (e.g., `surprise`).
*   **Outlier Analysis:** The findings report should explicitly mention how "outliers" (examples where logit and geometry disagree sharply) are handled or if they are highlighted in the scatter plots.
*   **Label Mapping Documentation:** While the plan correctly avoids *merging* `joy` and `happiness`, it should explicitly require that the `logit-geometry-summary.json` contains the `id_to_label` mapping used during the analysis to ensure complete provenance.

## 5. Should any part of the comparison logic be delayed?

**No.** The "conservative comparison" approach (framing BGE as context only) is the right level of restraint for this stage. Delaying it further would make the findings note less useful for the broader project narrative.

## Conclusion

The Phase 3 plan is a significant step toward the "interpretation layer" goals of the project. It handles the sensitive label-matching issue with appropriate care and establishes a reusable analytical pipeline. **Phase 3 should proceed.**
