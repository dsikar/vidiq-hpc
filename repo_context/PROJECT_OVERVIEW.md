# Project Compass

This file is the retained orientation document for the anonymised checkout.

It is intentionally narrower than the original internal compass. The goal here is to keep the experiment surface understandable without shipping private planning material, meeting logs, submission strategy notes, or machine-local configuration.

## Scope

Retained:

- experiment code under `experiments/`
- retained experiment reports and figures under the experiment folders
- utility scripts under `scripts/`
- configuration files under `configs/`
- limited paper-facing outputs that do not depend on internal repo context

Removed:

- meeting transcripts and minutes
- submission checklists and paper-drafting context
- agent-generated review material and chat-export reports
- local agent settings and tracked submodule metadata
- supporting private documents that were previously stored under `repo_context/project_context/`

## Repository Map

### Text Embedding Geometry

`experiments/understanding_text_embeddings/`

- `src/`
  analysis and report-generation scripts for the five retained phases
- `reports/phase1_summary.html`
  baseline topology and centroid-distance view
- `reports/phase2_summary.html`
  overlap and radial density analysis
- `reports/phase3_summary.html`
  signal-retention / directional erasure view
- `reports/phase4_summary.html`
  isolated-subspace analysis
- `reports/phase5_summary.html`
  geometry versus logit-consistency analysis
- `plan.md`
  compact description of the phase structure

### Brain / Cross-System Experiments

`experiments/brain_embedding_understanding/`

- `adding_spatial_context/`
  retained 11D system representation outputs and generator
- `checking_context_retention_across_dimensions/`
  retained context-retention comparison outputs
- `checking_density_geometry/`
  retained density-geometry outputs
- `checking_centroids/`
  retained centroid and relational-geometry outputs
- `checking_centroids_with_spatial_context_data/`
  retained 11D centroid comparison outputs
- `valence-arousal-dimensional_reduction/`
  retained affect-axis alignment outputs
- `global_behavior_comparison/`
  retained cross-system comparison artifacts and script

### Utilities

- `scripts/prepare_balanced6_dataset.py`
- `scripts/embed_balanced_6_emotions_raw.py`
- `configs/`

## Working Assumptions

- Some scripts depend on external datasets that are not stored in this repository.
- Those scripts now use repo-relative defaults or environment variables instead of machine-specific absolute paths.
- The retained report artifacts are intended to remain readable even if the external raw datasets are absent.

## Common External Inputs

- `BALANCED_EMOTIONS_CSV`
- `TEXT_VALIDATION_CSV`
- `RETENTION_METRICS_SOURCE`
- `BRAIN_48D_CSV`
- `BRAIN_11D_CSV`
- `BRAIN_GEOMETRY_DIR`
- `LLM_PAIRWISE_DIR`
- `GLOBAL_BEHAVIOR_OUTPUT_DIR`

## Anonymisation Notes

This checkout is meant to be materially safer to share than the original working tree, but the safest release process is still:

1. export from this cleaned tree rather than from the original repository history
2. exclude `.git/` and any unrelated local scratch files
3. rerun the standard path/name sweeps before packaging a final archive

## Recommended Reading Order

1. `README.md`
2. `experiments/understanding_text_embeddings/plan.md`
3. the retained phase summaries under `experiments/understanding_text_embeddings/reports/`
4. the retained final reports under `experiments/brain_embedding_understanding/**/reports/`
