from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from paths import ARTIFACTS_DIR, PLOTS_DIR

os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS_DIR / "logs" / "mplconfig"))

import matplotlib.pyplot as plt

from io_utils import ensure_dir, read_json, write_json


VARIANT_RUNS = {
    "raw": "run-101-bge-base-en-v1-5-raw-meanpool",
    "l2": "run-102-bge-base-en-v1-5-l2-meanpool",
    "centered_l2": "run-103-bge-base-en-v1-5-centered-l2-meanpool",
}


def _load_labels(experiment_root: Path) -> np.ndarray:
    labels_path = experiment_root / "data" / "processed" / "validation" / "labels.npy"
    return np.load(labels_path)


def _load_variant_paths(experiment_root: Path, run_name: str) -> tuple[np.ndarray, dict]:
    artifacts = read_json(experiment_root / "runs" / run_name / "artifacts.json")
    variant_key = artifacts["embedding_variant"]
    vectors = np.load(artifacts["validation_embeddings"][variant_key])
    return vectors, artifacts


def _pca_projection(vectors: np.ndarray) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)
    return coords, pca


def _nonlinear_projection(vectors: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        import umap

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=25,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(vectors), "umap"
    except Exception:
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        return reducer.fit_transform(vectors), "tsne"


def _sentiment_axis_projection(vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    pos = vectors[labels == 1]
    neg = vectors[labels == 0]
    axis = pos.mean(axis=0) - neg.mean(axis=0)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-12:
        return np.zeros(len(vectors), dtype=np.float32)
    unit_axis = axis / axis_norm
    return vectors @ unit_axis


def _scatter_plot(coords: np.ndarray, labels: np.ndarray, title: str, output_path: Path, xlab: str, ylab: str) -> None:
    plt.figure(figsize=(8, 6))
    for label, color, name in [(0, "#d1495b", "negative"), (1, "#00798c", "positive")]:
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.65, c=color, label=name)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _hist_plot(values: np.ndarray, labels: np.ndarray, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for label, color, name in [(0, "#d1495b", "negative"), (1, "#00798c", "positive")]:
        plt.hist(
            values[labels == label],
            bins=30,
            alpha=0.55,
            color=color,
            label=name,
            density=True,
        )
    plt.title(title)
    plt.xlabel("Projection value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualization-only plots for BGE SST-2 variants.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()

    experiment_root = args.experiment_root
    labels = _load_labels(experiment_root)
    plot_root = ensure_dir(PLOTS_DIR / "bge-variant-visuals")

    summary: dict[str, dict] = {}

    for variant_name, run_name in VARIANT_RUNS.items():
        vectors, artifacts = _load_variant_paths(experiment_root, run_name)
        variant_dir = ensure_dir(plot_root / variant_name)

        pca_coords, pca = _pca_projection(vectors)
        nonlinear_coords, nonlinear_method = _nonlinear_projection(vectors)
        axis_projection = _sentiment_axis_projection(vectors, labels)

        _scatter_plot(
            pca_coords,
            labels,
            f"SST-2 {variant_name} PCA (validation only)",
            variant_dir / "pca-2d.png",
            "PC1",
            "PC2",
        )
        _scatter_plot(
            nonlinear_coords,
            labels,
            f"SST-2 {variant_name} {nonlinear_method.upper()} (validation only)",
            variant_dir / f"{nonlinear_method}-2d.png",
            f"{nonlinear_method.upper()}-1",
            f"{nonlinear_method.upper()}-2",
        )
        _hist_plot(
            axis_projection,
            labels,
            f"SST-2 {variant_name} sentiment-axis projection",
            variant_dir / "sentiment-axis-hist.png",
        )

        summary[variant_name] = {
            "run_name": run_name,
            "embedding_variant": artifacts["embedding_variant"],
            "nonlinear_projection_method": nonlinear_method,
            "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
            "pca_class_centroids": {
                "negative": pca_coords[labels == 0].mean(axis=0).tolist(),
                "positive": pca_coords[labels == 1].mean(axis=0).tolist(),
            },
            "sentiment_axis_stats": {
                "negative_mean": float(axis_projection[labels == 0].mean()),
                "positive_mean": float(axis_projection[labels == 1].mean()),
                "negative_std": float(axis_projection[labels == 0].std()),
                "positive_std": float(axis_projection[labels == 1].std()),
            },
        }

    write_json(plot_root / "projection-summary.json", summary)


if __name__ == "__main__":
    main()
