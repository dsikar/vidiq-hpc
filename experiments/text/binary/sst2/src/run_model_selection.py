from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from data import prepare_dataset
from embeddings import embed_texts, l2_normalize, save_embedding_variants
from io_utils import ensure_dir, read_json, write_json
from metrics import centroid_metrics, knn_probe, logistic_probe, pca_metrics, _safe_silhouette
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


def _log_stage(run_dir: Path, stage: str, message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {stage}: {message}"
    print(line, flush=True)
    log_path = run_dir / "logs" / "stage-log.txt"
    ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SST-2 model selection stage.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "model-selection-first.json",
    )
    args = parser.parse_args()

    ensure_base_dirs()
    config = read_json(args.config)
    prepared = prepare_dataset(config)

    dataset_cfg = config["dataset"]
    train_split = dataset_cfg["train_split"]
    val_split = dataset_cfg["validation_split"]
    model_select_train_split = "train_model_selection" if "train_model_selection" in prepared else train_split
    train_texts = prepared[model_select_train_split]["texts"]
    train_labels = np.array(prepared[model_select_train_split]["labels"], dtype=np.int64)
    val_texts = prepared[val_split]["texts"]
    val_labels = np.array(prepared[val_split]["labels"], dtype=np.int64)

    embedding_cfg = config["embedding"]
    eval_cfg = config["evaluation"]

    for idx, model_cfg in enumerate(tqdm(config["models"], desc="Model selection", unit="model"), start=1):
        run_id = f"run-{idx:03d}-{model_cfg['slug']}-meanpool-l2-model-select"
        run_dir = _run_dir(run_id)
        ensure_dir(run_dir / "metrics")
        ensure_dir(run_dir / "logs")
        _save_run_config(run_dir, args.config, config)

        _log_stage(run_dir, "embed-start", f"embedding train split for {model_cfg['name']}")
        train_raw, train_meta = embed_texts(
            train_texts,
            model_cfg["name"],
            embedding_cfg["max_length"],
            embedding_cfg["batch_size"],
        )
        _log_stage(run_dir, "embed-done", "finished train embedding extraction")

        _log_stage(run_dir, "embed-start", f"embedding validation split for {model_cfg['name']}")
        val_raw, val_meta = embed_texts(
            val_texts,
            model_cfg["name"],
            embedding_cfg["max_length"],
            embedding_cfg["batch_size"],
        )
        _log_stage(run_dir, "embed-done", "finished validation embedding extraction")

        train_paths = save_embedding_variants(model_select_train_split, model_cfg["slug"], train_raw, train_meta)
        val_paths = save_embedding_variants(val_split, model_cfg["slug"], val_raw, val_meta)

        train_l2 = l2_normalize(train_raw)
        val_l2 = l2_normalize(val_raw)

        metrics: dict[str, object] = {}

        _log_stage(run_dir, "metric-start", "running logistic regression")
        metrics["logistic_regression"] = logistic_probe(
            train_l2,
            train_labels,
            val_l2,
            val_labels,
            eval_cfg["logistic_max_iter"],
        )
        write_json(run_dir / "metrics" / "summary.json", metrics)
        _log_stage(run_dir, "metric-done", "finished logistic regression")

        _log_stage(run_dir, "metric-start", "running kNN probes")
        metrics["knn"] = knn_probe(
            train_l2,
            train_labels,
            val_l2,
            val_labels,
            eval_cfg["knn_k_values"],
        )
        write_json(run_dir / "metrics" / "summary.json", metrics)
        _log_stage(run_dir, "metric-done", "finished kNN probes")

        _log_stage(run_dir, "metric-start", "computing geometry metrics")
        metrics["geometry"] = centroid_metrics(val_l2, val_labels)
        write_json(run_dir / "metrics" / "summary.json", metrics)
        _log_stage(run_dir, "metric-done", "finished geometry metrics")

        _log_stage(run_dir, "metric-start", "computing silhouette score")
        metrics["cluster"] = {"silhouette": _safe_silhouette(val_l2, val_labels)}
        write_json(run_dir / "metrics" / "summary.json", metrics)
        _log_stage(run_dir, "metric-done", "finished silhouette score")

        _log_stage(run_dir, "metric-start", "computing PCA spectrum metrics")
        metrics["pca"] = pca_metrics(val_l2)
        write_json(run_dir / "metrics" / "summary.json", metrics)
        _log_stage(run_dir, "metric-done", "finished PCA spectrum metrics")

        write_json(
            run_dir / "artifacts.json",
            {
                "train_split_used_for_model_selection": model_select_train_split,
                "train_embeddings": train_paths,
                "validation_embeddings": val_paths,
            },
        )


if __name__ == "__main__":
    main()
