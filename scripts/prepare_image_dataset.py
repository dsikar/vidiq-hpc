#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def _bootstrap_repo_src() -> None:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    repo_src_str = str(repo_src)
    if repo_src_str not in sys.path:
        sys.path.insert(0, repo_src_str)


_bootstrap_repo_src()


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOTS = {
    "emotion6": Path(
        os.environ.get(
            "VIDIQ_IMAGE_DATA_ROOT_EMOTION6",
            "/users/aczd097/archive/vidiq-hpc/data/image/emotion6",
        )
    ),
    "fi": Path(
        os.environ.get(
            "VIDIQ_IMAGE_DATA_ROOT_FI",
            "/users/aczd097/archive/vidiq-hpc/data/image/fi",
        )
    ),
}


def get_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or validate staged local image datasets before main image runs."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DEFAULT_DATA_ROOTS),
        help="Dataset to prepare or validate.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Explicit staged dataset root. Defaults to the Hyperion dataset root for the selected dataset.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional manually acquired source dataset root to copy from. Expected layout: metadata.csv + images/.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the staged dataset root; do not attempt to copy from --source-root.",
    )
    parser.add_argument(
        "--sample-checks",
        type=int,
        default=8,
        help="Number of dataset rows to sample through the repo loader for validation.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_missing(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        ensure_dir(destination.parent)
        shutil.copy2(source, destination)


def stage_from_source(dataset: str, source_root: Path, data_root: Path) -> None:
    metadata_source = source_root / "metadata.csv"
    images_source = source_root / "images"
    if not metadata_source.exists() or not images_source.exists():
        raise FileNotFoundError(
            f"{dataset} source root must contain metadata.csv and images/. "
            f"Missing under {source_root}."
        )

    ensure_dir(data_root)
    copy_if_missing(metadata_source, data_root / "metadata.csv")
    copy_if_missing(images_source, data_root / "images")


def validate_dataset(dataset: str, data_root: Path, sample_checks: int) -> dict:
    from image_experiments.datasets import build_image_dataset

    dataset_obj = build_image_dataset(
        dataset_name=dataset,
        data_root=data_root,
        transform=None,
        download=False,
        dataset_id=None,
        split="train",
        hf_cache_dir=None,
    )

    sample_count = min(max(sample_checks, 1), len(dataset_obj))
    for idx in range(sample_count):
        dataset_obj[idx]

    return {
        "dataset_name": dataset,
        "data_root": str(data_root),
        "dataset_size": len(dataset_obj),
        "sample_checks": sample_count,
        "label_to_id": getattr(dataset_obj, "label_to_id", {}),
        "class_names": getattr(dataset_obj, "class_names", []),
        "backend": getattr(dataset_obj, "backend", "unknown"),
    }


def write_manifest(dataset: str, data_root: Path, validation_summary: dict, source_root: Path | None) -> Path:
    manifest_path = data_root / "preparation_manifest.json"
    payload = {
        "dataset_name": dataset,
        "prepared_at": get_timestamp(),
        "repo_root": str(REPO_ROOT),
        "data_root": str(data_root),
        "source_root": str(source_root) if source_root is not None else None,
        "status": "ready",
        "validation": validation_summary,
        "notes": [
            "This manifest documents staged dataset readiness for the image pipeline.",
            "Raw staged datasets under Hyperion archive storage are not intended to be pushed to git.",
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


def main() -> None:
    args = parse_args()
    dataset = args.dataset
    data_root = args.data_root or DEFAULT_DATA_ROOTS[dataset]

    print("=" * 60)
    print(f"DATASET PREP: {dataset}")
    print(f"TARGET ROOT : {data_root}")
    print(f"SOURCE ROOT : {args.source_root if args.source_root else '(none)'}")
    print(f"MODE        : {'validate-only' if args.validate_only else 'prepare+validate'}")
    print("=" * 60)

    if not args.validate_only and args.source_root is not None:
        print(f"Staging {dataset} from {args.source_root} ...")
        stage_from_source(dataset, args.source_root, data_root)

    try:
        summary = validate_dataset(dataset, data_root, args.sample_checks)
    except Exception as exc:
        message = (
            f"{dataset} is not ready at {data_root}.\n"
            "This preparation step does not fabricate dataset availability.\n"
            "Provide a manually acquired/staged dataset root with metadata.csv and images/, "
            "or rerun with --source-root pointing at such a dataset.\n"
            f"Underlying validation error: {exc}"
        )
        raise SystemExit(message) from exc

    manifest_path = write_manifest(dataset, data_root, summary, args.source_root)
    print("VALIDATION: success")
    print(f"DATASET SIZE: {summary['dataset_size']}")
    print(f"MANIFEST: {manifest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
