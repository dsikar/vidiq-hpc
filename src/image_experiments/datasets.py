from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from image_experiments.io_utils import ensure_dir


class BaseStagedImageDataset(Dataset):
    IMAGE_PATH_COLUMNS = ("image_path", "filepath", "path", "image", "filename")
    LABEL_COLUMNS = ("emotion", "label_name", "label", "class", "category")

    def __init__(
        self,
        data_root: Path,
        class_names: list[str],
        dataset_name: str,
        transform=None,
        download: bool = False,
    ):
        self.data_root = data_root
        self.transform = transform
        self.dataset_name = dataset_name
        self.image_dir = data_root / "images"
        self.annotation_file = data_root / "metadata.csv"
        self.source_manifest = data_root / "source_manifest.json"
        self.backend = "local"
        self.class_names = class_names
        self.label_to_id = {label: idx for idx, label in enumerate(class_names)}
        self.samples: list[dict[str, str]] = []

        if self.annotation_file.exists():
            self._load_local_samples()
        elif download:
            self._download()
        else:
            raise FileNotFoundError(self._missing_data_message())

        if not self.samples:
            raise RuntimeError(
                f"No {self.dataset_name} samples were loaded from {self.annotation_file}."
            )

    def _load_local_samples(self) -> None:
        self.backend = "local"
        with self.annotation_file.open(mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.samples = list(reader)

    def _download(self) -> None:
        raise FileNotFoundError(self._missing_data_message())

    @staticmethod
    def _detach_rgb_image(image: Image.Image) -> Image.Image:
        converted = image.convert("RGB")
        detached = converted.copy()
        try:
            image.close()
        except Exception:
            pass
        try:
            converted.close()
        except Exception:
            pass
        return detached

    def _canonicalize_label(self, value: str) -> str:
        return value.strip().lower()

    def _resolve_image_path(self, row: dict[str, str]) -> Path:
        for column in self.IMAGE_PATH_COLUMNS:
            raw_value = row.get(column)
            if raw_value:
                candidate = Path(raw_value)
                if candidate.is_absolute():
                    return candidate
                return self.image_dir / candidate
        raise KeyError(
            f"{self.dataset_name} metadata must contain one of {self.IMAGE_PATH_COLUMNS}."
        )

    def _resolve_label(self, row: dict[str, str]) -> int:
        for column in self.LABEL_COLUMNS:
            raw_value = row.get(column)
            if raw_value is None or raw_value == "":
                continue
            normalized = raw_value.strip()
            if normalized.isdigit():
                label_id = int(normalized)
                if 0 <= label_id < len(self.class_names):
                    return label_id
                raise ValueError(
                    f"{self.dataset_name} label id {label_id} is outside the expected range "
                    f"[0, {len(self.class_names) - 1}]."
                )
            canonical = self._canonicalize_label(normalized)
            if canonical in self.label_to_id:
                return self.label_to_id[canonical]
            raise KeyError(
                f"Unsupported {self.dataset_name} label `{normalized}` in {self.annotation_file}. "
                f"Expected one of {sorted(self.label_to_id)} or a 0-based integer id."
            )
        raise KeyError(
            f"{self.dataset_name} metadata must contain one of {self.LABEL_COLUMNS}."
        )

    def _recommended_hyperion_storage(self) -> list[str]:
        return [f"- /users/aczd097/archive/vidiq-hpc/data/image/{self.dataset_name}"]

    def _missing_data_message(self, extra_note: str | None = None) -> str:
        lines = [
            f"{self.dataset_name} staged metadata was not found at {self.annotation_file}.",
            f"Expected staged image directory: {self.image_dir}",
            "This dataset currently expects a staged local dataset root with:",
            "- `metadata.csv`",
            "- an `images/` directory containing the referenced files",
            f"Configured data root: {self.data_root}",
            "Recommended Hyperion storage:",
            *self._recommended_hyperion_storage(),
            "Metadata should preferably use canonical label names rather than integer ids.",
        ]
        if extra_note is not None:
            lines.append(extra_note)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        row = self.samples[idx]
        img_path = self._resolve_image_path(row)
        if not img_path.exists():
            raise FileNotFoundError(
                f"{self.dataset_name} staged image is missing: {img_path}. "
                f"Check that `data_root` points at the correct staged dataset root."
            )
        with Image.open(img_path) as image_file:
            image = image_file.convert("RGB").copy()
        label = self._resolve_label(row)

        if self.transform:
            image = self.transform(image)

        return image, label


class EmoSetDataset(BaseStagedImageDataset):
    """
    EmoSet-118K Dataset implementation.
    Reference: https://vcc.tech/EmoSet
    """

    EMOTIONS = [
        "amusement",
        "anger",
        "awe",
        "contentment",
        "disgust",
        "excitement",
        "fear",
        "sadness",
    ]

    def __init__(
        self,
        data_root: Path,
        transform=None,
        download: bool = False,
        dataset_id: str = "Woleek/EmoSet-118K",
        split: str = "train",
        hf_cache_dir: Path | None = None,
    ):
        self.dataset_id = dataset_id
        self.split = split
        self.hf_cache_dir = hf_cache_dir
        self.hf_dataset = None
        super().__init__(
            data_root=data_root,
            class_names=self.EMOTIONS,
            dataset_name="emoset",
            transform=transform,
            download=download,
        )

    def _recommended_hyperion_storage(self) -> list[str]:
        return [
            "- /users/aczd097/archive/vidiq-hpc/data/image/emoset for staged dataset manifests or copied assets",
            "- /users/aczd097/sharedscratch/huggingface/datasets for Hugging Face cache data",
        ]

    def _load_hf_samples(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "The `datasets` package is required for Hugging Face-backed EmoSet loading."
            ) from exc

        cache_dir = str(self.hf_cache_dir) if self.hf_cache_dir is not None else None
        self.hf_dataset = load_dataset(self.dataset_id, split=self.split, cache_dir=cache_dir)
        self.backend = "huggingface"
        ensure_dir(self.data_root)
        with self.source_manifest.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_id": self.dataset_id,
                    "split": self.split,
                    "backend": self.backend,
                    "data_root": str(self.data_root),
                    "hf_cache_dir": cache_dir,
                    "note": "Images and metadata are provided by the Hugging Face dataset cache rather than repo-local files.",
                },
                f,
                indent=2,
            )

    def _download(self) -> None:
        ensure_dir(self.data_root)

        if self.annotation_file.exists():
            self._load_local_samples()
            return

        print("Attempting Hugging Face-backed EmoSet load...")
        try:
            self._load_hf_samples()
        except Exception as exc:
            raise FileNotFoundError(self._missing_data_message(str(exc))) from exc

    def __len__(self) -> int:
        if self.backend == "huggingface":
            return len(self.hf_dataset)
        return super().__len__()

    def __getitem__(self, idx: int):
        if self.backend == "huggingface":
            row = self.hf_dataset[idx]
            image = row["image"]
            if not isinstance(image, Image.Image):
                if isinstance(image, dict):
                    image_path = image.get("path")
                    if image_path:
                        with Image.open(image_path) as image_file:
                            image = image_file.convert("RGB").copy()
                    else:
                        raise RuntimeError(
                            "Hugging Face EmoSet row did not provide a usable image payload."
                        )
                else:
                    raise RuntimeError("Unsupported Hugging Face image payload type.")
            else:
                image = self._detach_rgb_image(image)

            if "label" in row and isinstance(row["label"], int):
                label = int(row["label"])
            else:
                label = self.label_to_id[str(row["emotion"]).strip().lower()]

            if self.transform:
                image = self.transform(image)

            return image, label

        return super().__getitem__(idx)


class Emotion6Dataset(BaseStagedImageDataset):
    CLASS_NAMES = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

    def __init__(self, data_root: Path, transform=None, download: bool = False):
        super().__init__(
            data_root=data_root,
            class_names=self.CLASS_NAMES,
            dataset_name="emotion6",
            transform=transform,
            download=download,
        )

    def _canonicalize_label(self, value: str) -> str:
        aliases = {
            "happy": "joy",
            "happiness": "joy",
        }
        canonical = value.strip().lower()
        return aliases.get(canonical, canonical)

    def _missing_data_message(self, extra_note: str | None = None) -> str:
        note = (
            "Expected canonical labels: anger, disgust, fear, joy, sadness, surprise. "
            "If your staged metadata uses an alternative naming scheme, normalize it in metadata.csv."
        )
        if extra_note:
            note = f"{note} {extra_note}"
        return super()._missing_data_message(note)


class FIDataset(BaseStagedImageDataset):
    CLASS_NAMES = [
        "amusement",
        "anger",
        "awe",
        "contentment",
        "disgust",
        "excitement",
        "fear",
        "sadness",
    ]

    def __init__(self, data_root: Path, transform=None, download: bool = False):
        super().__init__(
            data_root=data_root,
            class_names=self.CLASS_NAMES,
            dataset_name="fi",
            transform=transform,
            download=download,
        )

    def _canonicalize_label(self, value: str) -> str:
        aliases = {
            "pleasure": "amusement",
            "satisfaction": "contentment",
            "content": "contentment",
            "surprise": "awe",
        }
        canonical = value.strip().lower()
        return aliases.get(canonical, canonical)

    def _missing_data_message(self, extra_note: str | None = None) -> str:
        note = (
            "FI access is request-based in the survey reports, so this repo only supports staged local data. "
            "Expected canonical labels: amusement, anger, awe, contentment, disgust, excitement, fear, sadness."
        )
        if extra_note:
            note = f"{note} {extra_note}"
        return super()._missing_data_message(note)


class EmoVerseDataset(Dataset):
    """
    EmoVerse Dataset implementation with Background-Attribute-Subject (B-A-S) triplets.
    Reference: https://arxiv.org/html/2511.12554v1
    """

    def __init__(self, data_root: Path, transform=None, download: bool = False, mode: str = "full"):
        self.data_root = data_root
        self.transform = transform
        self.mode = mode

        if download:
            self._download()

        self.samples = []

    def _download(self):
        print("Downloading EmoVerse B-A-S data...")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.mode == "subject":
            pass
        elif self.mode == "background":
            pass

        if self.transform:
            image = self.transform(image)

        return image, sample["label"]


def build_image_dataset(
    dataset_name: str,
    data_root: Path,
    transform=None,
    download: bool = False,
    dataset_id: str | None = None,
    split: str = "train",
    hf_cache_dir: Path | None = None,
) -> Dataset:
    normalized = dataset_name.strip().lower()
    if normalized == "emoset":
        return EmoSetDataset(
            data_root=data_root,
            transform=transform,
            download=download,
            dataset_id=dataset_id or "Woleek/EmoSet-118K",
            split=split,
            hf_cache_dir=hf_cache_dir,
        )
    if normalized == "emotion6":
        return Emotion6Dataset(
            data_root=data_root,
            transform=transform,
            download=download,
        )
    if normalized == "fi":
        return FIDataset(
            data_root=data_root,
            transform=transform,
            download=download,
        )
    raise ValueError(
        f"Unsupported image dataset `{dataset_name}`. Expected one of: emoset, emotion6, fi."
    )
