from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedShuffleSplit

from io_utils import ensure_dir, read_json, write_json
from paths import ARTIFACTS_DIR, PLOTS_DIR

os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS_DIR / "logs" / "mplconfig"))

import matplotlib.pyplot as plt


PALETTE = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#eaac8b", "#94d2bd"]
DEFAULT_BRIDGE_RUN = "run-201-qwen3-1-7b-finetune-10e"


def _resolve_bridge_artifact_path(bridge_run_dir: Path, rel_path: str) -> Path:
    return (bridge_run_dir / rel_path).resolve()


def _label_names_from_metadata(source_run_metadata: dict) -> list[str]:
    label_to_id = source_run_metadata["dataset"]["label_to_id"]
    return [label for label, _ in sorted(label_to_id.items(), key=lambda item: item[1])]


def _load_filtered_csv_labels(csv_path: Path, label_to_id: dict[str, int]) -> np.ndarray:
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
            raw_labels.append(label_to_id[label])
    return np.array(raw_labels, dtype=np.int64)


def _reconstruct_eval_labels(source_run_metadata: dict) -> np.ndarray:
    args = source_run_metadata["args"]
    dataset = source_run_metadata["dataset"]
    if args["csv_path"] is None:
        raise ValueError("Phase 2 plotting currently expects a source run created from csv_path.")
    csv_path = Path(args["csv_path"])
    label_to_id = {str(label): int(idx) for label, idx in dataset["label_to_id"].items()}
    labels = _load_filtered_csv_labels(csv_path, label_to_id)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(args["test_size"]),
        random_state=int(args["seed"]),
    )
    _, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return labels[test_idx]


def _pca_projection(vectors: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)
    return coords, pca


def _tsne_projection(vectors: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        return reducer.fit_transform(vectors), "tsne"
    except Exception:
        pca_coords, _ = _pca_projection(vectors)
        return pca_coords, "pca-fallback"


def _scatter_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    title: str,
    output_path: Path,
    xlab: str,
    ylab: str,
) -> None:
    plt.figure(figsize=(8, 6))
    for label, name, color in zip(range(len(label_names)), label_names, PALETTE):
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.65, c=color, label=name)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _line_plot(x: list[int], y: list[float], title: str, y_label: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2, markersize=5, color="#355070")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _centroid_heatmap(matrix: np.ndarray, label_names: list[str], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(label_names)), label_names, rotation=45, ha="right")
    plt.yticks(range(len(label_names)), label_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _load_centroid_matrix(centroid_path: Path, label_names: list[str]) -> np.ndarray:
    centroids_payload = read_json(centroid_path)
    vectors_by_class = {int(item["class"]): np.array(item["centroid"], dtype=np.float32) for item in centroids_payload}
    ordered_vectors = np.stack([vectors_by_class[idx] for idx in range(len(label_names))], axis=0)
    return cosine_distances(ordered_vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset-level plots for the Qwen fine-tuned multiclass run.")
    parser.add_argument("--experiment-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bridge-run", default=DEFAULT_BRIDGE_RUN)
    args = parser.parse_args()

    experiment_root = args.experiment_root.resolve()
    bridge_run_dir = (experiment_root / "runs" / args.bridge_run).resolve()
    bridge_artifacts = read_json(bridge_run_dir / "artifacts.json")
    bridge_config = read_json(bridge_run_dir / "config.json")
    source_run_metadata = read_json(_resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["run_metadata"]))
    label_names = _label_names_from_metadata(source_run_metadata)

    eval_embeddings_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["eval_embeddings"])
    centroid_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["centroids"])
    train_metrics_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["train_metrics"])
    eval_metrics_path = _resolve_bridge_artifact_path(bridge_run_dir, bridge_artifacts["analysis"]["eval_metrics"])

    eval_embeddings = np.load(eval_embeddings_path)
    eval_labels = _reconstruct_eval_labels(source_run_metadata)
    if len(eval_labels) != len(eval_embeddings):
        raise ValueError(
            f"Reconstructed eval labels length {len(eval_labels)} does not match embedding rows {len(eval_embeddings)}."
        )

    train_metrics = read_json(train_metrics_path)
    eval_metrics = read_json(eval_metrics_path)
    centroid_distance_matrix = _load_centroid_matrix(centroid_path, label_names)

    plot_root = ensure_dir(PLOTS_DIR / "qwen-finetune-10e")

    epochs = [int(item["epoch"]) for item in train_metrics]
    losses = [float(item["loss"]) for item in train_metrics]
    accuracies = [float(item["accuracy"]) for item in train_metrics]

    _line_plot(
        epochs,
        losses,
        "Qwen fine-tune train loss (balanced 10e)",
        "Loss",
        plot_root / "train-loss-vs-epoch.png",
    )
    _line_plot(
        epochs,
        accuracies,
        "Qwen fine-tune eval accuracy (balanced 10e)",
        "Accuracy",
        plot_root / "eval-accuracy-vs-epoch.png",
    )

    pca_coords, pca = _pca_projection(eval_embeddings)
    tsne_coords, tsne_method = _tsne_projection(eval_embeddings)

    _scatter_plot(
        pca_coords,
        eval_labels,
        label_names,
        "Qwen fine-tune eval embeddings PCA (balanced holdout)",
        plot_root / "eval-embeddings-pca-2d.png",
        "PC1",
        "PC2",
    )
    _scatter_plot(
        tsne_coords,
        eval_labels,
        label_names,
        f"Qwen fine-tune eval embeddings {tsne_method.upper()} (balanced holdout)",
        plot_root / "eval-embeddings-tsne-2d.png",
        f"{tsne_method.upper()}-1",
        f"{tsne_method.upper()}-2",
    )
    _centroid_heatmap(
        centroid_distance_matrix,
        label_names,
        "Qwen fine-tune centroid cosine distance",
        plot_root / "centroid-distance-heatmap.png",
    )

    summary = {
        "bridge_run": args.bridge_run,
        "source_run_id": bridge_config["source_run_id"],
        "source_files": {
            "train_metrics": str(train_metrics_path),
            "eval_embeddings": str(eval_embeddings_path),
            "centroids": str(centroid_path),
            "eval_metrics": str(eval_metrics_path),
        },
        "evaluation": {
            "dataset_size": int(eval_metrics["dataset_size"]),
            "accuracy": float(eval_metrics["accuracy"]),
            "embedding_dimensionality": int(eval_embeddings.shape[1]),
            "label_names": label_names,
            "label_caveat": "The Qwen stage uses happiness while the earlier BGE materials use joy.",
        },
        "plotting": {
            "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "nonlinear_projection_method": tsne_method,
            "centroid_distance_metric": "cosine",
            "generated_files": [
                "train-loss-vs-epoch.png",
                "eval-accuracy-vs-epoch.png",
                "eval-embeddings-pca-2d.png",
                "eval-embeddings-tsne-2d.png",
                "centroid-distance-heatmap.png",
            ],
        },
    }
    write_json(plot_root / "projection-summary.json", summary)


if __name__ == "__main__":
    main()
