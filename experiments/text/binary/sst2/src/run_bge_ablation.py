from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data import prepare_dataset
from embeddings import embed_texts, save_embedding_variants
from io_utils import ensure_dir, read_json, write_json
from metrics import (
    centroid_metrics,
    confusion_matrix_payload,
    knn_probe,
    logistic_probe,
    pca_metrics,
    _safe_silhouette,
)
from paths import RUNS_DIR, ensure_base_dirs


def _run_dir(run_id: str) -> Path:
    return ensure_dir(RUNS_DIR / run_id)


def _save_run_config(run_dir: Path, config_path: Path, config: dict) -> None:
    shutil.copy2(config_path, run_dir / "config.json")
    write_json(
        run_dir / "run-metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(config_path),
            "experiment_name": config["experiment_name"],
        },
    )


def _load_embedding_variant(path_str: str) -> np.ndarray:
    return np.load(path_str)


def _log_stage(run_dir: Path, stage: str, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {stage}: {message}"
    print(line, flush=True)
    log_path = run_dir / "logs" / "stage-log.txt"
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_progress(run_dir: Path, payload: dict) -> None:
    write_json(run_dir / "progress.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BGE embedding ablations for SST-2.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "bge-ablation-stage.json",
    )
    args = parser.parse_args()

    ensure_base_dirs()
    config = read_json(args.config)
    prepared = prepare_dataset(config)

    dataset_cfg = config["dataset"]
    train_split = dataset_cfg["train_split"]
    val_split = dataset_cfg["validation_split"]
    train_labels = np.array(prepared[train_split]["labels"], dtype=np.int64)
    val_labels = np.array(prepared[val_split]["labels"], dtype=np.int64)

    model_cfg = config["model"]
    embedding_cfg = config["embedding"]
    eval_cfg = config["evaluation"]

    print("[ablation] starting dataset preparation and BGE embedding extraction", flush=True)
    train_raw, train_meta = embed_texts(
        prepared[train_split]["texts"],
        model_cfg["name"],
        embedding_cfg["max_length"],
        embedding_cfg["batch_size"],
    )
    val_raw, val_meta = embed_texts(
        prepared[val_split]["texts"],
        model_cfg["name"],
        embedding_cfg["max_length"],
        embedding_cfg["batch_size"],
    )

    train_paths = save_embedding_variants(train_split, model_cfg["slug"], train_raw, train_meta)
    val_paths = save_embedding_variants(val_split, model_cfg["slug"], val_raw, val_meta)

    for idx, variant_cfg in enumerate(tqdm(config["variants"], desc="BGE ablation variants", unit="variant"), start=1):
        variant_name = variant_cfg["name"]
        embedding_key = variant_cfg["embedding_key"]
        run_id = f"run-{idx + 100:03d}-{model_cfg['slug']}-{variant_name}"
        run_dir = _run_dir(run_id)
        ensure_dir(run_dir / "metrics")
        ensure_dir(run_dir / "logs")
        _save_run_config(run_dir, args.config, config)
        _write_progress(
            run_dir,
            {
                "status": "started",
                "variant_name": variant_name,
                "embedding_key": embedding_key,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

        _log_stage(run_dir, "load", f"loading embedding variant '{embedding_key}'")
        train_x = _load_embedding_variant(train_paths[embedding_key])
        val_x = _load_embedding_variant(val_paths[embedding_key])
        _write_progress(
            run_dir,
            {
                "status": "embeddings_loaded",
                "variant_name": variant_name,
                "embedding_key": embedding_key,
            },
        )

        summary: dict[str, object] = {}

        _log_stage(run_dir, "metric-start", "running logistic regression")
        summary["logistic_regression"] = logistic_probe(
            train_x,
            train_labels,
            val_x,
            val_labels,
            eval_cfg["logistic_max_iter"],
        )
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished logistic regression")

        _log_stage(run_dir, "metric-start", "running kNN probes")
        summary["knn"] = knn_probe(
            train_x,
            train_labels,
            val_x,
            val_labels,
            eval_cfg["knn_k_values"],
        )
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished kNN probes")

        _log_stage(run_dir, "metric-start", "computing centroid and distance geometry metrics")
        summary["geometry"] = centroid_metrics(val_x, val_labels)
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished geometry metrics")

        _log_stage(run_dir, "metric-start", "computing silhouette score")
        summary["cluster"] = {"silhouette": _safe_silhouette(val_x, val_labels)}
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished silhouette score")

        _log_stage(run_dir, "metric-start", "computing PCA spectrum metrics")
        summary["pca"] = pca_metrics(val_x)
        write_json(run_dir / "metrics" / "summary.json", summary)
        _log_stage(run_dir, "metric-done", "finished PCA spectrum metrics")

        _log_stage(run_dir, "metric-start", "computing confusion matrix")
        confusion = confusion_matrix_payload(
            train_x,
            train_labels,
            val_x,
            val_labels,
            eval_cfg["logistic_max_iter"],
        )
        write_json(run_dir / "metrics" / "summary.json", summary)
        write_json(run_dir / "metrics" / "confusion-matrix.json", confusion)
        write_json(
            run_dir / "artifacts.json",
            {
                "train_split_used": train_split,
                "validation_split_used": val_split,
                "embedding_variant": embedding_key,
                "train_embeddings": train_paths,
                "validation_embeddings": val_paths,
            },
        )
        _write_progress(
            run_dir,
            {
                "status": "completed",
                "variant_name": variant_name,
                "embedding_key": embedding_key,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        _log_stage(run_dir, "complete", f"finished variant '{variant_name}'")


if __name__ == "__main__":
    main()
