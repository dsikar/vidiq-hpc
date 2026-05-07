# vidiq-hpc

This repository is a public-safe research checkout for emotion-geometry experiments across text embeddings and brain representations. Internal meeting notes, submission-planning material, agent review files, and local machine configuration have been removed from this tree.

Start with [`repo_context/PROJECT_OVERVIEW.md`](repo_context/PROJECT_OVERVIEW.md). It is the retained compass for the anonymised checkout and points to the main experiment directories and outputs.

## Main Areas

- `experiments/understanding_text_embeddings/`
  five-phase text embedding geometry workflow, retained reports, and report-generation scripts
- `experiments/brain_embedding_understanding/`
  cross-system comparison experiments and retained reports
- `scripts/`
  data-preparation and embedding-generation utilities
- `configs/`
  experiment configuration files
- `paper_submissions/`
  retained paper-facing artifacts that do not depend on internal planning context

## Useful Entry Points

- [`experiments/understanding_text_embeddings/plan.md`](experiments/understanding_text_embeddings/plan.md)
- [`experiments/understanding_text_embeddings/reports/phase1_summary.html`](experiments/understanding_text_embeddings/reports/phase1_summary.html)
- [`experiments/understanding_text_embeddings/reports/phase2_summary.html`](experiments/understanding_text_embeddings/reports/phase2_summary.html)
- [`experiments/understanding_text_embeddings/reports/phase3_summary.html`](experiments/understanding_text_embeddings/reports/phase3_summary.html)
- [`experiments/understanding_text_embeddings/reports/phase4_summary.html`](experiments/understanding_text_embeddings/reports/phase4_summary.html)
- [`experiments/understanding_text_embeddings/reports/phase5_summary.html`](experiments/understanding_text_embeddings/reports/phase5_summary.html)
- [`experiments/brain_embedding_understanding/adding_spatial_context/outputs/BRAIN_11D_SYSTEMS_REPORT.md`](experiments/brain_embedding_understanding/adding_spatial_context/outputs/BRAIN_11D_SYSTEMS_REPORT.md)
- [`experiments/brain_embedding_understanding/checking_context_retention_across_dimensions/reports/CONTEXT_RETENTION_FINAL_REPORT.md`](experiments/brain_embedding_understanding/checking_context_retention_across_dimensions/reports/CONTEXT_RETENTION_FINAL_REPORT.md)
- [`experiments/brain_embedding_understanding/checking_centroids/reports/CENTROID_RELATIONAL_FINAL_REPORT.md`](experiments/brain_embedding_understanding/checking_centroids/reports/CENTROID_RELATIONAL_FINAL_REPORT.md)
- [`experiments/brain_embedding_understanding/valence-arousal-dimensional_reduction/reports/VA_GEOMETRIC_CONVERGENCE_REPORT.md`](experiments/brain_embedding_understanding/valence-arousal-dimensional_reduction/reports/VA_GEOMETRIC_CONVERGENCE_REPORT.md)

## External Inputs

Some retained scripts operate on datasets that are intentionally not checked into the repository. Those scripts now resolve inputs through repo-relative defaults or environment variables instead of machine-specific absolute paths.

Common environment variables:

- `BALANCED_EMOTIONS_CSV`
- `BALANCED_EMOTIONS_OUTPUT_ROOT`
- `TEXT_VALIDATION_CSV`
- `RETENTION_METRICS_SOURCE`
- `BRAIN_48D_CSV`
- `BRAIN_11D_CSV`
- `BRAIN_GEOMETRY_DIR`
- `LLM_PAIRWISE_DIR`
- `GLOBAL_BEHAVIOR_OUTPUT_DIR`

If those inputs are not present locally, the script will fail with a standard missing-file error rather than embedding a user-specific path in code or generated metadata.

## Release Notes

- Removed tracked meeting transcripts and minutes.
- Removed tracked repo-context drafting material, paper versions, and submission checklists.
- Removed tracked agent-review and chat-export reports.
- Removed tracked local agent settings and submodule URL metadata.
- Replaced hardcoded absolute paths in retained scripts with repo-relative or env-driven resolution.
