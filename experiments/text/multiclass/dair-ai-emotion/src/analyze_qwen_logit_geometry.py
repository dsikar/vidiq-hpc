from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.model_selection import StratifiedShuffleSplit

from io_utils import ensure_dir, read_json, slugify, write_json
from paths import ARTIFACTS_DIR, EXPERIMENT_ROOT, METRICS_DIR, PLOTS_DIR

os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS_DIR / "logs" / "mplconfig"))

import matplotlib.pyplot as plt


PALETTE = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#94d2bd"]
DEFAULT_BRIDGE_RUN = "run-201-qwen3-1-7b-finetune-10e"


def _resolve_bridge_artifact_path(bridge_run_dir: Path, rel_path: str) -> Path:
    return (bridge_run_dir / rel_path).resolve()


def _repo_root(experiment_root: Path) -> Path:
    return experiment_root.parents[3]


def _repo_relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def _label_names_from_metadata(source_run_metadata: dict) -> list[str]:
    label_to_id = source_run_metadata["dataset"]["label_to_id"]
    return [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]


def _default_output_slug(bridge_run_dir: Path) -> str:
    summary_path = bridge_run_dir / "metrics" / "summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        epochs = summary.get("training", {}).get("epochs_completed")
        if epochs is not None:
            return f"qwen-finetune-{int(epochs)}e"
    return slugify(bridge_run_dir.name)


def _resolve_csv_path(repo_root: Path, source_run_metadata: dict) -> Path:
    args_csv = source_run_metadata["args"].get("csv_path")
    if args_csv:
        csv_path = Path(args_csv)
        if csv_path.exists():
            return csv_path.resolve()
        candidate = (repo_root / csv_path).resolve()
        if candidate.exists():
            return candidate
    dataset_path = source_run_metadata["dataset"].get("source_path")
    if dataset_path:
        csv_path = Path(dataset_path)
        if csv_path.exists():
            return csv_path.resolve()
        candidate = (repo_root / Path("experiments/text/multiclass/dair-ai-emotion/data/raw") / csv_path.name).resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not resolve the balanced CSV path from source run metadata.")


def _csv_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_filtered_csv_rows(csv_path: Path, label_to_id: dict[str, int]) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    raw_labels: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")
        candidate_text_columns = [column for column in ("cleaned_text", "sentence", "text") if column in reader.fieldnames]
        if not candidate_text_columns:
            raise ValueError(f"CSV file {csv_path} must contain one of: cleaned_text, sentence, text")
        for row in reader:
            chosen_text = None
            for column in candidate_text_columns:
                value = str(row.get(column, "")).strip()
                if value:
                    chosen_text = value
                    break
            if not chosen_text:
                continue
            label = str(row.get("emotion", "")).strip()
            if not label:
                continue
            texts.append(chosen_text)
            raw_labels.append(label_to_id[label])
    return texts, np.array(raw_labels, dtype=np.int64)


def _reconstruct_eval_labels(repo_root: Path, source_run_metadata: dict) -> tuple[np.ndarray, np.ndarray, Path, dict[str, object]]:
    args = source_run_metadata["args"]
    dataset = source_run_metadata["dataset"]
    csv_path = _resolve_csv_path(repo_root, source_run_metadata)
    label_to_id = {str(label): int(idx) for label, idx in dataset["label_to_id"].items()}
    texts, labels = _load_filtered_csv_rows(csv_path, label_to_id)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(args["test_size"]),
        random_state=int(args["seed"]),
    )
    _, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    eval_labels = labels[test_idx]
    text_hash = hashlib.sha256()
    for idx in test_idx[:50]:
        text_hash.update(texts[int(idx)].encode("utf-8"))
        text_hash.update(b"\n")
    provenance = {
        "csv_path": str(csv_path),
        "csv_sha256": _csv_sha256(csv_path),
        "filtered_row_count": int(len(labels)),
        "full_label_counts": [int(x) for x in np.bincount(labels, minlength=len(label_to_id))],
        "eval_label_counts": [int(x) for x in np.bincount(eval_labels, minlength=len(label_to_id))],
        "eval_text_sample_sha256": text_hash.hexdigest(),
    }
    return eval_labels, test_idx, csv_path, provenance


def _load_centroid_vectors(centroid_path: Path, label_names: list[str]) -> tuple[np.ndarray, dict[int, list[float]]]:
    centroids_payload = read_json(centroid_path)
    vectors_by_class = {int(item["class"]): np.array(item["centroid"], dtype=np.float32) for item in centroids_payload}
    ordered_vectors = np.stack([vectors_by_class[idx] for idx in range(len(label_names))], axis=0)
    return ordered_vectors, {idx: vectors_by_class[idx].tolist() for idx in range(len(label_names))}


def _validate_inputs(
    eval_embeddings: np.ndarray,
    eval_logits: np.ndarray,
    eval_labels: np.ndarray,
    centroid_vectors: np.ndarray,
    num_labels: int,
    test_size: float,
) -> dict[str, object]:
    errors: list[str] = []
    if eval_embeddings.shape[0] != eval_logits.shape[0]:
        errors.append(
            f"Embedding rows {eval_embeddings.shape[0]} do not match logit rows {eval_logits.shape[0]}."
        )
    if eval_embeddings.shape[0] != len(eval_labels):
        errors.append(
            f"Reconstructed eval labels length {len(eval_labels)} does not match artifact rows {eval_embeddings.shape[0]}."
        )
    if eval_logits.shape[1] != num_labels:
        errors.append(f"Logit column count {eval_logits.shape[1]} does not match num_labels {num_labels}.")
    if centroid_vectors.shape[0] != num_labels:
        errors.append(f"Centroid count {centroid_vectors.shape[0]} does not match num_labels {num_labels}.")
    if centroid_vectors.shape[1] != eval_embeddings.shape[1]:
        errors.append(
            f"Centroid dimensionality {centroid_vectors.shape[1]} does not match embedding dimensionality {eval_embeddings.shape[1]}."
        )
    expected_counts = np.full(num_labels, int(round(len(eval_labels) / max(num_labels, 1))), dtype=np.int64)
    observed_counts = np.bincount(eval_labels, minlength=num_labels)
    if not np.array_equal(observed_counts, expected_counts):
        errors.append(
            "Reconstructed evaluation label counts are implausible for the balanced CSV held-out split: "
            f"observed={observed_counts.tolist()} expected={expected_counts.tolist()} at test_size={test_size}."
        )
    if errors:
        raise ValueError("Phase 3 validation failed:\n- " + "\n- ".join(errors))
    return {
        "expected_eval_label_counts": [int(x) for x in expected_counts],
        "observed_eval_label_counts": [int(x) for x in observed_counts],
    }


def _metric_distances(eval_embeddings: np.ndarray, centroid_vectors: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "euclidean": euclidean_distances(eval_embeddings, centroid_vectors),
        "cosine": cosine_distances(eval_embeddings, centroid_vectors),
    }


def _per_example_rank_stats(logits: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    corrs = np.zeros(logits.shape[0], dtype=np.float64)
    pvals = np.zeros(logits.shape[0], dtype=np.float64)
    for idx in range(logits.shape[0]):
        result = spearmanr(logits[idx], -distances[idx])
        corrs[idx] = 0.0 if np.isnan(result.statistic) else float(result.statistic)
        pvals[idx] = 1.0 if np.isnan(result.pvalue) else float(result.pvalue)
    return corrs, pvals


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = spearmanr(x, y)
    statistic = 0.0 if np.isnan(result.statistic) else float(result.statistic)
    pvalue = 1.0 if np.isnan(result.pvalue) else float(result.pvalue)
    return statistic, pvalue


def _outlier_mask(
    predicted_labels: np.ndarray,
    nearest_labels: np.ndarray,
    per_example_spearman: np.ndarray,
) -> np.ndarray:
    return (predicted_labels != nearest_labels) | (per_example_spearman <= 0.0)


def _label_mapping(source_run_metadata: dict) -> dict[str, object]:
    label_to_id = {str(label): int(idx) for label, idx in source_run_metadata["dataset"]["label_to_id"].items()}
    id_to_label = {str(idx): label for label, idx in label_to_id.items()}
    return {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "label_caveat": "The Qwen stage uses happiness while the earlier BGE materials use joy.",
    }


def _plot_dual_hexbin(
    x_primary: np.ndarray,
    y: np.ndarray,
    x_secondary: np.ndarray,
    y_secondary: np.ndarray,
    output_path: Path,
    title: str,
    xlab_primary: str,
    xlab_secondary: str,
    ylab: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, xvals, yvals, subtitle, xlabel in [
        (axes[0], x_primary, y, "Euclidean", xlab_primary),
        (axes[1], x_secondary, y_secondary, "Cosine", xlab_secondary),
    ]:
        hb = ax.hexbin(xvals, yvals, gridsize=35, cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=ax)
        ax.set_title(subtitle)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(ylab)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_rank_bars(
    label_names: list[str],
    primary_values: list[float],
    secondary_values: list[float],
    output_path: Path,
) -> None:
    positions = np.arange(len(label_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(positions - width / 2, primary_values, width=width, color="#355070", label="Euclidean")
    ax.bar(positions + width / 2, secondary_values, width=width, color="#b56576", label="Cosine")
    ax.set_xticks(positions)
    ax.set_xticklabels(label_names, rotation=25, ha="right")
    ax.set_ylabel("Mean Spearman r")
    ax.set_title("Qwen fine-tune logit-distance rank agreement by class")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_dual_heatmap(
    primary_matrix: np.ndarray,
    secondary_matrix: np.ndarray,
    label_names: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, matrix, subtitle in [
        (axes[0], primary_matrix, "Euclidean"),
        (axes[1], secondary_matrix, "Cosine"),
    ]:
        image = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0)
        fig.colorbar(image, ax=ax)
        ax.set_title(subtitle)
        ax.set_xticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha="right")
        ax.set_yticks(range(len(label_names)))
        ax.set_yticklabels(label_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Nearest-centroid class")
    fig.suptitle("Qwen fine-tune nearest-centroid vs predicted class")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_confusion(nearest_labels: np.ndarray, predicted_labels: np.ndarray, num_labels: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((num_labels, num_labels), dtype=np.int64)
    for nearest, predicted in zip(nearest_labels, predicted_labels):
        counts[int(nearest), int(predicted)] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(counts, dtype=np.float64)
    np.divide(counts, np.maximum(row_sums, 1), out=normalized, where=row_sums != 0)
    return counts, normalized


def _write_findings_report(
    report_path: Path,
    output_slug: str,
    summary: dict,
    rank_agreement: dict,
    nearest_centroid_summary: dict,
) -> None:
    evaluation = summary["evaluation"]
    agreement = summary["agreement_metrics"]
    distance_metrics = summary["distance_metrics"]
    outlier = summary["notes"]["outlier_policy"]
    lines = [
        "# Qwen Fine-Tuned Logit-Geometry Findings",
        "",
        "## Run Summary",
        "",
        f"- bridge run: `{summary['bridge_run']}`",
        f"- source run: `experiments/text_model/runs/{summary['source_run_id']}/`",
        f"- output bundle: `experiments/text/multiclass/dair-ai-emotion/artifacts/plots/{output_slug}/logit-geometry/`",
        f"- evaluation size: `{evaluation['dataset_size']}`",
        f"- evaluation accuracy: `{evaluation['accuracy']:.4f}`",
        "",
        "## Main Result",
        "",
        "The within-run Qwen analysis shows that logit ordering and centroid proximity are strongly aligned on the held-out balanced evaluation split.",
        "",
        f"- mean per-example rank agreement (Euclidean): `{rank_agreement['mean_spearman_r']['euclidean']:.4f}`",
        f"- mean per-example rank agreement (Cosine): `{rank_agreement['mean_spearman_r']['cosine']:.4f}`",
        f"- predicted class equals nearest centroid (Euclidean): `{nearest_centroid_summary['overall_match_rate']['euclidean']:.4f}`",
        f"- predicted class equals nearest centroid (Cosine): `{nearest_centroid_summary['overall_match_rate']['cosine']:.4f}`",
        f"- true label is nearest centroid (Euclidean): `{agreement['true_label_is_nearest_centroid_rate']['euclidean']:.4f}`",
        f"- true label is nearest centroid (Cosine): `{agreement['true_label_is_nearest_centroid_rate']['cosine']:.4f}`",
        "",
        "## Interpretation",
        "",
        "The Euclidean and cosine analyses tell the same broad story for this run: the fine-tuned classifier is not only accurate, its raw logits are geometrically coherent with the exported evaluation embeddings.",
        "",
        f"- global true-class logit vs distance correlation (Euclidean): `{distance_metrics['global_true_class_logit_distance_correlation']['euclidean']['spearman_r']:.4f}`",
        f"- global true-class logit vs distance correlation (Cosine): `{distance_metrics['global_true_class_logit_distance_correlation']['cosine']['spearman_r']:.4f}`",
        "",
        "That means higher confidence for the true class generally corresponds to smaller distance from the true class centroid, which is the central Phase 3 claim.",
        "",
        "## Caveats",
        "",
        f"- Outliers were defined as `{outlier}`.",
        "- This remains a within-run analysis only. It does not justify claiming that Qwen geometry is better than the earlier BGE stage.",
        "- The `joy` versus `happiness` label mismatch remains unresolved for direct cross-stage quantitative comparison.",
        "- These results come from the held-out balanced CSV split, not the earlier BGE validation workflow.",
        "",
        "## Outputs",
        "",
        f"- `artifacts/metrics/{output_slug}/logit-geometry-summary.json`",
        f"- `artifacts/metrics/{output_slug}/per-class-logit-distance-correlations.json`",
        f"- `artifacts/metrics/{output_slug}/nearest-centroid-vs-prediction.json`",
        f"- `artifacts/metrics/{output_slug}/distance-rank-agreement.json`",
        f"- `artifacts/plots/{output_slug}/logit-geometry/true-class-logit-vs-distance.png`",
        f"- `artifacts/plots/{output_slug}/logit-geometry/predicted-class-logit-vs-distance.png`",
        f"- `artifacts/plots/{output_slug}/logit-geometry/distance-margin-vs-logit-margin.png`",
        f"- `artifacts/plots/{output_slug}/logit-geometry/rank-correlation-by-class.png`",
        f"- `artifacts/plots/{output_slug}/logit-geometry/nearest-centroid-confusion-heatmap.png`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 logit-geometry analysis for the integrated Qwen fine-tuned run.")
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--bridge-run", default=DEFAULT_BRIDGE_RUN)
    parser.add_argument("--output-slug", default=None)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    repo_root = _repo_root(experiment_root)
    bridge_run_dir = (experiment_root / "runs" / args.bridge_run).resolve()
    output_slug = args.output_slug or _default_output_slug(bridge_run_dir)

    bridge_artifacts = read_json(bridge_run_dir / "artifacts.json")
    bridge_config = read_json(bridge_run_dir / "config.json")
    source_run_metadata = read_json(_resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["run_metadata"]))
    label_names = _label_names_from_metadata(source_run_metadata)
    label_mapping = _label_mapping(source_run_metadata)

    eval_embeddings_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["eval_embeddings"])
    eval_logits_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["eval_logits"])
    centroid_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["centroids"])
    eval_metrics_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["eval_metrics"])
    train_metrics_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["train_metrics"])

    eval_embeddings = np.load(eval_embeddings_path)
    eval_logits = np.load(eval_logits_path)
    centroid_vectors, _ = _load_centroid_vectors(centroid_path, label_names)
    eval_labels, _, csv_path, provenance = _reconstruct_eval_labels(repo_root, source_run_metadata)
    eval_metrics = read_json(eval_metrics_path)

    validation = _validate_inputs(
        eval_embeddings=eval_embeddings,
        eval_logits=eval_logits,
        eval_labels=eval_labels,
        centroid_vectors=centroid_vectors,
        num_labels=int(source_run_metadata["dataset"]["num_labels"]),
        test_size=float(source_run_metadata["args"]["test_size"]),
    )

    distances = _metric_distances(eval_embeddings, centroid_vectors)
    predicted_labels = np.argmax(eval_logits, axis=1)
    nearest_labels = {metric: np.argmin(matrix, axis=1) for metric, matrix in distances.items()}
    per_example_stats = {
        metric: _per_example_rank_stats(eval_logits, matrix) for metric, matrix in distances.items()
    }

    top2_logits = np.sort(eval_logits, axis=1)[:, -2:]
    logit_margin = top2_logits[:, 1] - top2_logits[:, 0]
    distance_margin = {}
    true_class_logits = eval_logits[np.arange(len(eval_labels)), eval_labels]
    true_class_distances = {}
    predicted_class_distances = {}
    outlier_masks = {}
    for metric, matrix in distances.items():
        sorted_distances = np.sort(matrix, axis=1)
        distance_margin[metric] = sorted_distances[:, 1] - sorted_distances[:, 0]
        true_class_distances[metric] = matrix[np.arange(len(eval_labels)), eval_labels]
        predicted_class_distances[metric] = matrix[np.arange(len(eval_labels)), predicted_labels]
        outlier_masks[metric] = _outlier_mask(predicted_labels, nearest_labels[metric], per_example_stats[metric][0])

    metrics_root = ensure_dir(METRICS_DIR / output_slug)
    plot_root = ensure_dir(PLOTS_DIR / output_slug / "logit-geometry")

    _plot_dual_hexbin(
        true_class_distances["euclidean"],
        true_class_logits,
        true_class_distances["cosine"],
        true_class_logits,
        plot_root / "true-class-logit-vs-distance.png",
        "Qwen fine-tune true-class logit vs centroid distance",
        "True-class Euclidean distance",
        "True-class cosine distance",
        "True-class logit",
    )
    _plot_dual_hexbin(
        predicted_class_distances["euclidean"],
        eval_logits[np.arange(len(predicted_labels)), predicted_labels],
        predicted_class_distances["cosine"],
        eval_logits[np.arange(len(predicted_labels)), predicted_labels],
        plot_root / "predicted-class-logit-vs-distance.png",
        "Qwen fine-tune predicted-class logit vs centroid distance",
        "Predicted-class Euclidean distance",
        "Predicted-class cosine distance",
        "Predicted-class logit",
    )
    _plot_dual_hexbin(
        distance_margin["euclidean"],
        logit_margin,
        distance_margin["cosine"],
        logit_margin,
        plot_root / "distance-margin-vs-logit-margin.png",
        "Qwen fine-tune distance margin vs logit margin",
        "Euclidean distance margin",
        "Cosine distance margin",
        "Logit margin",
    )

    per_class_rows = []
    per_class_mean_primary: list[float] = []
    per_class_mean_secondary: list[float] = []
    per_class_primary_p: list[float] = []
    per_class_secondary_p: list[float] = []
    per_class_match_primary = {}
    per_class_match_secondary = {}
    per_class_true_nearest_primary = {}
    per_class_true_nearest_secondary = {}
    for label_idx, label_name in enumerate(label_names):
        mask = eval_labels == label_idx
        per_class_corr_e, per_class_p_e = _safe_spearman(
            true_class_logits[mask], true_class_distances["euclidean"][mask]
        )
        per_class_corr_c, per_class_p_c = _safe_spearman(
            true_class_logits[mask], true_class_distances["cosine"][mask]
        )
        per_class_mean_primary.append(float(np.mean(per_example_stats["euclidean"][0][mask])))
        per_class_mean_secondary.append(float(np.mean(per_example_stats["cosine"][0][mask])))
        per_class_primary_p.append(float(np.mean(per_example_stats["euclidean"][1][mask])))
        per_class_secondary_p.append(float(np.mean(per_example_stats["cosine"][1][mask])))
        per_class_match_primary[label_name] = float(np.mean(predicted_labels[mask] == nearest_labels["euclidean"][mask]))
        per_class_match_secondary[label_name] = float(np.mean(predicted_labels[mask] == nearest_labels["cosine"][mask]))
        per_class_true_nearest_primary[label_name] = float(np.mean(nearest_labels["euclidean"][mask] == eval_labels[mask]))
        per_class_true_nearest_secondary[label_name] = float(np.mean(nearest_labels["cosine"][mask] == eval_labels[mask]))
        per_class_rows.append(
            {
                "label": label_name,
                "count": int(np.sum(mask)),
                "spearman_r": float(np.mean(per_example_stats["euclidean"][0][mask])),
                "spearman_p": float(np.mean(per_example_stats["euclidean"][1][mask])),
                "secondary_spearman_r": float(np.mean(per_example_stats["cosine"][0][mask])),
                "secondary_spearman_p": float(np.mean(per_example_stats["cosine"][1][mask])),
                "true_class_logit_distance_correlation": {
                    "euclidean": {"spearman_r": per_class_corr_e, "spearman_p": per_class_p_e},
                    "cosine": {"spearman_r": per_class_corr_c, "spearman_p": per_class_p_c},
                },
                "outlier_count": int(np.sum(outlier_masks["euclidean"][mask])),
                "outlier_rate": float(np.mean(outlier_masks["euclidean"][mask])),
                "prediction_matches_nearest_centroid": {
                    "euclidean": per_class_match_primary[label_name],
                    "cosine": per_class_match_secondary[label_name],
                },
                "true_label_is_nearest_centroid": {
                    "euclidean": per_class_true_nearest_primary[label_name],
                    "cosine": per_class_true_nearest_secondary[label_name],
                },
            }
        )

    _plot_rank_bars(label_names, per_class_mean_primary, per_class_mean_secondary, plot_root / "rank-correlation-by-class.png")

    counts_primary, normalized_primary = _build_confusion(nearest_labels["euclidean"], predicted_labels, len(label_names))
    counts_secondary, normalized_secondary = _build_confusion(nearest_labels["cosine"], predicted_labels, len(label_names))
    _plot_dual_heatmap(
        normalized_primary,
        normalized_secondary,
        label_names,
        plot_root / "nearest-centroid-confusion-heatmap.png",
    )

    global_true_corr_e, global_true_p_e = _safe_spearman(true_class_logits, true_class_distances["euclidean"])
    global_true_corr_c, global_true_p_c = _safe_spearman(true_class_logits, true_class_distances["cosine"])

    distance_rank_agreement = {
        "label_names": label_names,
        "primary_metric": "euclidean",
        "secondary_metric": "cosine",
        "mean_spearman_r": {
            "euclidean": float(np.mean(per_example_stats["euclidean"][0])),
            "cosine": float(np.mean(per_example_stats["cosine"][0])),
        },
        "mean_spearman_p": {
            "euclidean": float(np.mean(per_example_stats["euclidean"][1])),
            "cosine": float(np.mean(per_example_stats["cosine"][1])),
        },
        "per_class_mean_spearman_r": {
            label: value for label, value in zip(label_names, per_class_mean_primary)
        },
        "per_class_mean_spearman_p": {
            label: value for label, value in zip(label_names, per_class_primary_p)
        },
        "secondary_per_class_mean_spearman_r": {
            label: value for label, value in zip(label_names, per_class_mean_secondary)
        },
        "secondary_per_class_mean_spearman_p": {
            label: value for label, value in zip(label_names, per_class_secondary_p)
        },
        "margin_summary": {
            "euclidean_logit_margin_spearman": {
                "spearman_r": _safe_spearman(logit_margin, distance_margin["euclidean"])[0],
                "spearman_p": _safe_spearman(logit_margin, distance_margin["euclidean"])[1],
            },
            "cosine_logit_margin_spearman": {
                "spearman_r": _safe_spearman(logit_margin, distance_margin["cosine"])[0],
                "spearman_p": _safe_spearman(logit_margin, distance_margin["cosine"])[1],
            },
        },
    }

    nearest_centroid_vs_prediction = {
        "label_names": label_names,
        "primary_metric": "euclidean",
        "secondary_metric": "cosine",
        "counts_matrix": counts_primary.tolist(),
        "normalized_matrix": normalized_primary.tolist(),
        "secondary_counts_matrix": counts_secondary.tolist(),
        "secondary_normalized_matrix": normalized_secondary.tolist(),
        "overall_match_rate": {
            "euclidean": float(np.mean(predicted_labels == nearest_labels["euclidean"])),
            "cosine": float(np.mean(predicted_labels == nearest_labels["cosine"])),
        },
        "per_class_match_rate": per_class_match_primary,
        "secondary_per_class_match_rate": per_class_match_secondary,
        "notes": {
            "rows": "nearest centroid class",
            "columns": "predicted class",
        },
    }

    per_class_logit_distance = {
        "label_names": label_names,
        "primary_metric": "euclidean",
        "secondary_metric": "cosine",
        "per_class": per_class_rows,
        "global": {
            "true_class_logit_distance_correlation": {
                "euclidean": {"spearman_r": global_true_corr_e, "spearman_p": global_true_p_e},
                "cosine": {"spearman_r": global_true_corr_c, "spearman_p": global_true_p_c},
            },
            "outlier_count": int(np.sum(outlier_masks["euclidean"])),
            "outlier_rate": float(np.mean(outlier_masks["euclidean"])),
        },
    }

    source_files = {
        "bridge_artifacts": _repo_relative(bridge_run_dir / "artifacts.json", repo_root),
        "bridge_run_metadata": _repo_relative(bridge_run_dir / "run-metadata.json", repo_root),
        "source_run_metadata": _repo_relative(_resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["run_metadata"]), repo_root),
        "train_metrics": _repo_relative(train_metrics_path, repo_root),
        "eval_metrics": _repo_relative(eval_metrics_path, repo_root),
        "eval_embeddings": _repo_relative(eval_embeddings_path, repo_root),
        "eval_logits": _repo_relative(eval_logits_path, repo_root),
        "centroids": _repo_relative(centroid_path, repo_root),
        "csv_path": _repo_relative(csv_path, repo_root),
    }
    dataset_provenance = {
        **provenance,
        "csv_path": _repo_relative(csv_path, repo_root),
    }
    generated_files = {
        "metrics": [
            f"experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/{output_slug}/logit-geometry-summary.json",
            f"experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/{output_slug}/per-class-logit-distance-correlations.json",
            f"experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/{output_slug}/nearest-centroid-vs-prediction.json",
            f"experiments/text/multiclass/dair-ai-emotion/artifacts/metrics/{output_slug}/distance-rank-agreement.json",
        ],
        "plots": [
            f"experiments/text/multiclass/dair-ai-emotion/artifacts/plots/{output_slug}/logit-geometry/true-class-logit-vs-distance.png",
            f"experiments/text/multiclass/dair-ai-emotion/artifacts/plots/{output_slug}/logit-geometry/predicted-class-logit-vs-distance.png",
            f"experiments/text/multiclass/dair-ai-emotion/artifacts/plots/{output_slug}/logit-geometry/distance-margin-vs-logit-margin.png",
            f"experiments/text/multiclass/dair-ai-emotion/artifacts/plots/{output_slug}/logit-geometry/rank-correlation-by-class.png",
            f"experiments/text/multiclass/dair-ai-emotion/artifacts/plots/{output_slug}/logit-geometry/nearest-centroid-confusion-heatmap.png",
        ],
        "report": f"experiments/text/multiclass/dair-ai-emotion/reports/qwen-finetune-logit-geometry-findings.md",
    }

    logit_geometry_summary = {
        "bridge_run": args.bridge_run,
        "source_run_id": bridge_config["source_run_id"],
        "source_files": source_files,
        "label_mapping": label_mapping,
        "evaluation": {
            "dataset_size": int(eval_metrics["dataset_size"]),
            "accuracy": float(eval_metrics["accuracy"]),
            "embedding_dimensionality": int(eval_embeddings.shape[1]),
            "num_labels": int(eval_logits.shape[1]),
            "label_names": label_names,
            "per_class_counts": {label: int(count) for label, count in zip(label_names, validation["observed_eval_label_counts"])},
            "validation": validation,
            "dataset_provenance": dataset_provenance,
        },
        "distance_metrics": {
            "phase2_centroid_heatmap_metric": "cosine",
            "primary_metric": "euclidean",
            "secondary_metric": "cosine",
            "global_true_class_logit_distance_correlation": {
                "euclidean": {"spearman_r": global_true_corr_e, "spearman_p": global_true_p_e},
                "cosine": {"spearman_r": global_true_corr_c, "spearman_p": global_true_p_c},
            },
            "mean_margin_alignment": {
                "euclidean": distance_rank_agreement["margin_summary"]["euclidean_logit_margin_spearman"],
                "cosine": distance_rank_agreement["margin_summary"]["cosine_logit_margin_spearman"],
            },
        },
        "agreement_metrics": {
            "prediction_matches_nearest_centroid_rate": nearest_centroid_vs_prediction["overall_match_rate"],
            "true_label_is_nearest_centroid_rate": {
                "euclidean": float(np.mean(nearest_labels["euclidean"] == eval_labels)),
                "cosine": float(np.mean(nearest_labels["cosine"] == eval_labels)),
            },
            "mean_rank_agreement": distance_rank_agreement["mean_spearman_r"],
        },
        "significance": {
            "mean_rank_agreement_p_value": distance_rank_agreement["mean_spearman_p"],
            "examples_with_p_lt_0_05_fraction": {
                "euclidean": float(np.mean(per_example_stats["euclidean"][1] < 0.05)),
                "cosine": float(np.mean(per_example_stats["cosine"][1] < 0.05)),
            },
            "global_true_class_logit_distance_p_value": {
                "euclidean": global_true_p_e,
                "cosine": global_true_p_c,
            },
        },
        "generated_files": generated_files,
        "notes": {
            "phase2_metric_note": "Phase 2 used cosine distance for the centroid heatmap. Phase 3 uses Euclidean as primary and cosine as secondary.",
            "comparison_constraint": "This is a within-run Qwen analysis only. It does not justify direct superiority claims against the earlier BGE stage.",
            "outlier_policy": "predicted class differs from nearest centroid or per-example Spearman rank agreement is non-positive under the primary metric",
            "large_arrays_not_duplicated": True,
        },
    }

    write_json(metrics_root / "logit-geometry-summary.json", logit_geometry_summary)
    write_json(metrics_root / "per-class-logit-distance-correlations.json", per_class_logit_distance)
    write_json(metrics_root / "nearest-centroid-vs-prediction.json", nearest_centroid_vs_prediction)
    write_json(metrics_root / "distance-rank-agreement.json", distance_rank_agreement)

    report_path = experiment_root / "reports" / "qwen-finetune-logit-geometry-findings.md"
    _write_findings_report(
        report_path=report_path,
        output_slug=output_slug,
        summary=logit_geometry_summary,
        rank_agreement=distance_rank_agreement,
        nearest_centroid_summary=nearest_centroid_vs_prediction,
    )


if __name__ == "__main__":
    main()
