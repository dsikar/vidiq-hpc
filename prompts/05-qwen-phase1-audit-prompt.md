Audit the implemented Phase 1 work for the Qwen fine-tuned run integration.

Context:
You are reviewing whether the approved Phase 1 integration work was implemented correctly, and whether it matches the reviewed plan plus the constraints from prior audit feedback.

Files to review:
- `reports/qwen-finetune-integration-plan.md`
- `reports/qwen-finetune-integration-plan-review-claude.md`
- `reports/qwen-finetune-integration-plan-review-gemini.md`
- `reports/qwen-finetune-integration-plan-recommendation-codex.md`

Implemented files to inspect:
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/config.json`
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/artifacts.json`
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/run-metadata.json`
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/progress.json`
- `experiments/text/multiclass/dair-ai-emotion/runs/run-201-qwen3-1-7b-finetune-10e/metrics/summary.json`
- `experiments/text/multiclass/dair-ai-emotion/reports/qwen-finetune-first-findings.md`
- `experiments/text/multiclass/dair-ai-emotion/reports/multiclass-emotion-dataset-findings.md`

Source run to verify against:
- `experiments/text_model/runs/tmqb0010_17763/`

What to check:
1. Was only Phase 1 implemented?
2. Was the path inconsistency fixed everywhere relevant?
3. Was the `joy` vs `happiness` label mismatch documented clearly enough?
4. Does the bridge run use lightweight metadata only, without duplicating large artifacts?
5. Do the bridge paths correctly point to the source run artifacts?
6. Does the new findings report accurately describe the run and its limitations?
7. Does the updated main findings report integrate the fine-tuned stage appropriately?
8. Did the implementation introduce any structural problems, misleading claims, or inconsistencies with the approved plan?
9. Is the work ready to proceed to commit/push, or does it need correction first?

Output requirements:
- Be concrete and critical.
- If something is wrong, quote the file path and describe the exact issue.
- Distinguish clearly between:
  - blocking issues
  - non-blocking improvements
- End with one of:
  - `Approved`
  - `Approved with minor fixes`
  - `Not approved`

Naming convention:
Write your review as a separate markdown file in `reports/` using this exact pattern:

- `qwen-finetune-phase1-audit-<reviewer-name>.md`

Examples:
- `qwen-finetune-phase1-audit-claude.md`
- `qwen-finetune-phase1-audit-gemini.md`

Your report must include:
1. Overall verdict
2. Blocking issues
3. Non-blocking issues
4. Verification summary
5. Final recommendation
