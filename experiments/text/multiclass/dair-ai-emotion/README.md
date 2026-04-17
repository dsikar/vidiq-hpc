# dair-ai/emotion Multiclass Experiment

This experiment root owns all `dair-ai/emotion` multiclass work.

Subdirectories:

- `reports/` experiment reports and conclusions
- `prompts/` reusable prompt context
- `src/` experiment-specific code
- `configs/` run and pipeline configs
- `requirements.txt` Python dependencies for this experiment
- `data/` raw and processed dataset material
- `artifacts/` reusable embeddings, metrics, plots, and logs
- `runs/` concrete run folders for embedding / validation variants

Current entry points:

- `src/run_embedding_generation.py` for direct BGE embedding generation
- `src/run_bge_ablation.py` for multiclass embedding-variant evaluation
- `src/plot_bge_variants.py` for visualization-only projections
- `src/plot_qwen_finetune_run.py` for dataset-level plotting of the integrated Qwen fine-tuned run

Qwen fine-tuned stage:

- dataset-level bridge run:
  - `runs/run-201-qwen3-1-7b-finetune-10e/`
- dataset-level plot output:
  - `artifacts/plots/qwen-finetune-10e/`
