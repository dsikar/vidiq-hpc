# Qwen Fine-Tuned Logit-Geometry Findings

## Run Summary

- bridge run: `run-201-qwen3-1-7b-finetune-10e`
- source run: `experiments/text_model/runs/tmqb0010_17763/`
- output bundle: `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/qwen-finetune-10e/logit-geometry/`
- evaluation size: `4038`
- evaluation accuracy: `0.9804`

## Main Result

The within-run Qwen analysis shows that logit ordering and centroid proximity are strongly aligned on the held-out balanced evaluation split.

- mean per-example rank agreement (Euclidean): `0.7189`
- mean per-example rank agreement (Cosine): `0.7159`
- predicted class equals nearest centroid (Euclidean): `0.9941`
- predicted class equals nearest centroid (Cosine): `0.9926`
- true label is nearest centroid (Euclidean): `0.9789`
- true label is nearest centroid (Cosine): `0.9792`

## Interpretation

The Euclidean and cosine analyses tell the same broad story for this run: the fine-tuned classifier is not only accurate, its raw logits are geometrically coherent with the exported evaluation embeddings.

- global true-class logit vs distance correlation (Euclidean): `-0.5802`
- global true-class logit vs distance correlation (Cosine): `-0.5307`

That means higher confidence for the true class generally corresponds to smaller distance from the true class centroid, which is the central Phase 3 claim.

## Caveats

- Outliers were defined as `predicted class differs from nearest centroid or per-example Spearman rank agreement is non-positive under the primary metric`.
- This remains a within-run analysis only. It does not justify claiming that Qwen geometry is better than the earlier BGE stage.
- The `joy` versus `happiness` label mismatch remains unresolved for direct cross-stage quantitative comparison.
- These results come from the held-out balanced CSV split, not the earlier BGE validation workflow.

## Outputs

- `artifacts/metrics/qwen-finetune-10e/logit-geometry-summary.json`
- `artifacts/metrics/qwen-finetune-10e/per-class-logit-distance-correlations.json`
- `artifacts/metrics/qwen-finetune-10e/nearest-centroid-vs-prediction.json`
- `artifacts/metrics/qwen-finetune-10e/distance-rank-agreement.json`
- `artifacts/plots/qwen-finetune-10e/logit-geometry/true-class-logit-vs-distance.png`
- `artifacts/plots/qwen-finetune-10e/logit-geometry/predicted-class-logit-vs-distance.png`
- `artifacts/plots/qwen-finetune-10e/logit-geometry/distance-margin-vs-logit-margin.png`
- `artifacts/plots/qwen-finetune-10e/logit-geometry/rank-correlation-by-class.png`
- `artifacts/plots/qwen-finetune-10e/logit-geometry/nearest-centroid-confusion-heatmap.png`
