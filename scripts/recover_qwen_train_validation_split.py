from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_METADATA_PATH = REPO_ROOT / "experiments" / "text_model" / "runs" / "tmqb0010_17763" / "run_metadata.json"
OUTPUT_DIR = REPO_ROOT / "experiments" / "text" / "multiclass" / "dair-ai-emotion" / "data" / "splits" / "qwen_tmqb0010_17763"


def _load_run_metadata() -> dict:
    with RUN_METADATA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_source_csv(run_metadata: dict) -> Path:
    csv_path = Path(run_metadata["args"]["csv_path"])
    if csv_path.is_absolute():
        return csv_path
    return (REPO_ROOT / csv_path).resolve()


def _load_filtered_rows(csv_path: Path) -> tuple[list[dict[str, str]], dict[str, int], str]:
    rows: list[dict[str, str]] = []
    raw_labels: list[str] = []
    chosen_text_column: str | None = None
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {csv_path} has no header row")
        if "emotion" not in reader.fieldnames:
            raise ValueError(f"CSV file {csv_path} must contain an 'emotion' column")
        candidate_text_columns = [column for column in ("cleaned_text", "sentence", "text") if column in reader.fieldnames]
        if not candidate_text_columns:
            raise ValueError(f"CSV file {csv_path} must contain one of: cleaned_text, sentence, text")
        for source_row_index, row in enumerate(reader):
            chosen_text: str | None = None
            for column in candidate_text_columns:
                value = str(row.get(column, "")).strip()
                if value:
                    chosen_text = value
                    chosen_text_column = column
                    break
            if not chosen_text:
                continue
            label = str(row.get("emotion", "")).strip()
            if not label:
                continue
            rows.append(
                {
                    "source_row_index": str(source_row_index),
                    "sentence": str(row.get("sentence", "")),
                    "emotion": label,
                    "cleaned_text": str(row.get("cleaned_text", "")),
                }
            )
            raw_labels.append(label)
    unique_labels = sorted(set(raw_labels))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    if chosen_text_column is None:
        raise ValueError("No usable text column found after filtering rows")
    return rows, label_to_id, chosen_text_column


def _write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["source_row_index", "sentence", "emotion", "cleaned_text"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    run_metadata = _load_run_metadata()
    csv_path = _resolve_source_csv(run_metadata)
    rows, label_to_id, text_column = _load_filtered_rows(csv_path)
    expected_label_to_id = {str(label): int(idx) for label, idx in run_metadata["dataset"]["label_to_id"].items()}
    if label_to_id != expected_label_to_id:
        raise ValueError(f"Recovered label mapping {label_to_id} does not match run metadata {expected_label_to_id}")

    labels = [label_to_id[row["emotion"]] for row in rows]
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(run_metadata["args"]["test_size"]),
        random_state=int(run_metadata["args"]["seed"]),
    )
    train_idx, validation_idx = next(splitter.split([[0]] * len(labels), labels))
    train_rows = [rows[int(idx)] for idx in train_idx]
    validation_rows = [rows[int(idx)] for idx in validation_idx]

    train_indices = {row["source_row_index"] for row in train_rows}
    validation_indices = {row["source_row_index"] for row in validation_rows}
    overlap = train_indices & validation_indices
    if overlap:
        raise ValueError(f"Overlap detected between train and validation splits: {sorted(overlap)[:5]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"
    metadata_path = OUTPUT_DIR / "split_metadata.json"
    _write_split_csv(train_path, train_rows)
    _write_split_csv(validation_path, validation_rows)

    train_counts = Counter(row["emotion"] for row in train_rows)
    validation_counts = Counter(row["emotion"] for row in validation_rows)
    filtered_counts = Counter(row["emotion"] for row in rows)

    metadata = {
        "source_run_id": run_metadata["run_name"],
        "source_run_metadata": str(RUN_METADATA_PATH.relative_to(REPO_ROOT)),
        "source_csv": str(csv_path.relative_to(REPO_ROOT)),
        "reconstruction_method": (
            "Rebuilt the historical Qwen split by applying the same CSV filtering logic as "
            "experiments/text_model/train_multiclass.py::load_texts_and_labels_from_csv and "
            "the same StratifiedShuffleSplit parameters recorded in tmqb0010_17763/run_metadata.json."
        ),
        "text_column": text_column,
        "label_column": run_metadata["dataset"]["label_column"],
        "label_to_id": label_to_id,
        "test_size": float(run_metadata["args"]["test_size"]),
        "seed": int(run_metadata["args"]["seed"]),
        "filtered_row_count": len(rows),
        "training_row_count": len(train_rows),
        "validation_row_count": len(validation_rows),
        "filtered_class_counts": dict(sorted(filtered_counts.items())),
        "train_class_counts": dict(sorted(train_counts.items())),
        "validation_class_counts": dict(sorted(validation_counts.items())),
        "intended_reuse": "Reuse this exact split for BGE and MPNet fine-tuning to maintain cross-model comparability with the historical Qwen run.",
        "validation_checks": {
            "train_plus_validation_equals_filtered": len(train_rows) + len(validation_rows) == len(rows),
            "no_overlap_between_splits": not overlap,
            "validation_row_count_matches_expected": len(validation_rows) == 4038,
        },
    }
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"Wrote {train_path.relative_to(REPO_ROOT)} ({len(train_rows)} rows)")
    print(f"Wrote {validation_path.relative_to(REPO_ROOT)} ({len(validation_rows)} rows)")
    print(f"Wrote {metadata_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
