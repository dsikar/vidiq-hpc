# 18 FI Archive Analytics Report

This report analyzes the FI dataset archive on Hyperion and identifies the exact downstream preparation work needed to turn it into the staged FI dataset layout expected by this repository.

Repo-relative report path:

- `reports/18-fi-archive-analytics-report.md`

Absolute report path on Hyperion after this file is committed and pulled there:

- `/users/aczd097/git/vidiq-hpc/reports/18-fi-archive-analytics-report.md`

Archive analyzed:

- `/users/aczd097/archive/vidiq-hpc/data/image/fi/emotion_dataset.tar`

## 1. Archive structure summary

The FI archive is structurally simple and clean.

- top-level directory name:
  - `emotion_dataset/`
- class directories present directly under that top-level directory:
  - `amusement`
  - `anger`
  - `awe`
  - `contentment`
  - `disgust`
  - `excitement`
  - `fear`
  - `sadness`

These class names match the repo’s expected canonical FI labels exactly.

The archive appears to contain:

- image files only
- no `.csv`, `.json`, or `.txt` metadata file
- no obvious non-image auxiliary files

So the archive is a class-folder image payload, not a repo-ready staged dataset.

## 2. Quantitative inventory

The following counts were obtained directly from the tar headers on Hyperion without full extraction.

- total files: `23185`
- total image files: `23185`
- non-image files: `0`
- file extensions observed:
  - `.jpg`: `23185`
- empty classes: none

Image count per class:

- `amusement`: `4923`
- `anger`: `1255`
- `awe`: `3133`
- `contentment`: `5356`
- `disgust`: `1657`
- `excitement`: `2914`
- `fear`: `1046`
- `sadness`: `2901`

Observations:

- the archive is strongly imbalanced across classes
- all files appear to be `.jpg`
- the label set is exactly the expected 8-class FI schema

## 3. Staging gap analysis

The repo currently expects the staged FI dataset root to look like:

```text
/users/aczd097/archive/vidiq-hpc/data/image/fi/
  metadata.csv
  images/
```

The archive currently provides:

- one tar file:
  - `/users/aczd097/archive/vidiq-hpc/data/image/fi/emotion_dataset.tar`
- an internal top-level image tree:
  - `emotion_dataset/<label>/<image>.jpg`

What can be reused directly:

- the class labels
- the image files themselves
- the existing class-folder organization

What is missing:

- `metadata.csv`
- the repo-expected extracted `images/` tree under the staged FI root

Critical point:

The current FI loader cannot consume the tar archive as-is.

Why:

- `src/image_experiments/datasets.py` expects:
  - `data_root / "metadata.csv"`
  - `data_root / "images"`
- the current preparation script also expects a source root shaped like:
  - `metadata.csv`
  - `images/`

So there is a concrete structural mismatch:

- archive shape: `emotion_dataset/<label>/<image>.jpg`
- repo staging shape: `metadata.csv` + `images/...`

## 4. Downstream preparation requirements

The archive analysis implies the following exact downstream preparation work:

1. extract the tar archive into a controlled staging area under:
   - `/users/aczd097/archive/vidiq-hpc/data/image/fi/`
2. convert the extracted class-folder payload into the repo’s expected staged surface
3. create `metadata.csv` from the directory structure
4. ensure image paths in `metadata.csv` are relative to:
   - `/users/aczd097/archive/vidiq-hpc/data/image/fi/images/`
5. preserve the canonical FI labels exactly as:
   - `amusement`
   - `anger`
   - `awe`
   - `contentment`
   - `disgust`
   - `excitement`
   - `fear`
   - `sadness`
6. validate that the staged dataset can be loaded by:
   - `scripts/prepare_image_dataset.py`
   - `src/image_experiments/datasets.py`

### Does `metadata.csv` need to be created?

Yes.

The archive contains no metadata file, so `metadata.csv` must be generated from the extracted directory tree.

### Do the images need to be flattened?

No, not necessarily.

The current loader can work with class subdirectories under `images/` if `metadata.csv` uses relative paths like:

- `amusement/amusement_0000.jpg`
- `anger/anger_0000.jpg`

That means the cleanest staged layout is probably:

```text
/users/aczd097/archive/vidiq-hpc/data/image/fi/
  metadata.csv
  images/
    amusement/
    anger/
    awe/
    contentment/
    disgust/
    excitement/
    fear/
    sadness/
```

So the directory tree can remain class-organized under `images/`.

### Can the extracted directory be consumed as-is?

Not directly.

Even after extraction, the archive’s natural root would be:

```text
emotion_dataset/
  amusement/
  anger/
  ...
```

That still does not satisfy the repo’s required staged layout until:

- the content is placed under `images/`
- `metadata.csv` is generated

## 5. Recommended processing plan

The next repo task should be:

- extend `scripts/prepare_image_dataset.py` to support tar-based FI staging from:
  - `/users/aczd097/archive/vidiq-hpc/data/image/fi/emotion_dataset.tar`

That extension should:

1. detect the FI tar archive under the staged FI root
2. extract it into a temporary working area or directly into `images/`
3. normalize the extracted directory structure so the final staged root becomes:
   - `metadata.csv`
   - `images/<class>/<image>.jpg`
4. generate `metadata.csv` from the extracted class-folder tree
5. run the existing validation path after generation
6. write `preparation_manifest.json`

That is the cleanest next engineering task because it removes the current manual gap between:

- “archive exists”
- “repo-ready staged FI dataset exists”

### Exact remote-side validation and run sequence after that change exists

Once tar-based FI staging support has been implemented locally, committed, and pulled to Hyperion, the exact remote-side validation sequence should be:

1. update the Hyperion checkout:

```bash
cd /users/aczd097/git/vidiq-hpc
git pull --rebase origin main
```

2. run FI preparation directly against the staged archive root:

```bash
python3 scripts/prepare_image_dataset.py --dataset fi
```

Expected result:

- detect `/users/aczd097/archive/vidiq-hpc/data/image/fi/emotion_dataset.tar`
- extract and normalize the archive into the staged FI layout
- generate `metadata.csv`
- create or populate `images/`
- validate the FI dataset through the existing loader
- write `preparation_manifest.json`

3. rerun the prep script in validation-only mode:

```bash
python3 scripts/prepare_image_dataset.py --dataset fi --validate-only
```

Expected result:

- confirm that the staged FI root is now complete and loadable without doing any additional extraction work

4. run the main FI job:

```bash
sbatch hpc/image_embedding_fi.slurm
```

If you prefer to keep preparation under SLURM rather than running the prep step directly in the shell, the corresponding sequence should be:

```bash
cd /users/aczd097/git/vidiq-hpc
git pull --rebase origin main
sbatch --export=DATASET=fi hpc/prepare_image_dataset.slurm
python3 scripts/prepare_image_dataset.py --dataset fi --validate-only
sbatch hpc/image_embedding_fi.slurm
```

That is the missing concrete recommendation for what should be run on Hyperion to validate `emotion_dataset.tar` and then proceed into the actual FI processing job.

## 6. Hyperion command notes

These are the exact command shapes used to inspect the archive on Hyperion without extracting it fully.

Lightweight header listing:

```bash
tar -tf /users/aczd097/archive/vidiq-hpc/data/image/fi/emotion_dataset.tar | head -n 50
```

Quantitative archive analysis:

```bash
python3 - <<'PY'
import json, tarfile
from pathlib import PurePosixPath
archive='/users/aczd097/archive/vidiq-hpc/data/image/fi/emotion_dataset.tar'
class_counts={}
non_image=[]
top_levels=set()
ext_counts={}
total_files=0
image_files=0
with tarfile.open(archive) as tf:
    for m in tf.getmembers():
        parts=PurePosixPath(m.name).parts
        if len(parts)>=2 and parts[0]=='emotion_dataset':
            top_levels.add(parts[1])
        if not m.isfile():
            continue
        total_files += 1
        suffix=PurePosixPath(m.name).suffix.lower()
        ext_counts[suffix]=ext_counts.get(suffix,0)+1
        if suffix in {'.jpg','.jpeg','.png','.bmp','.gif','.webp','.tif','.tiff'}:
            image_files += 1
            if len(parts)>=3 and parts[0]=='emotion_dataset':
                cls=parts[1]
                class_counts[cls]=class_counts.get(cls,0)+1
        else:
            non_image.append(m.name)
summary={
    'archive': archive,
    'top_level_entries': sorted(top_levels),
    'total_files': total_files,
    'image_files': image_files,
    'class_counts': dict(sorted(class_counts.items())),
    'empty_classes': [c for c in sorted(top_levels) if c not in class_counts],
    'extension_counts': dict(sorted(ext_counts.items())),
    'non_image_files': non_image[:50],
    'non_image_file_count': len(non_image),
}
print(json.dumps(summary, indent=2))
PY
```

## 7. End-to-end handoff workflow

This section gives the exact operational sequence for local development, Hyperion execution, and return to local analysis.

### 1. Commit the script locally

Files involved:

- the next FI staging script or prep-script changes
- any matching doc updates

Command shape:

```bash
git add <files>
git commit -m "Add FI tar-based staging support"
```

Expected output artifact:

- a local commit containing the FI preparation logic

### 2. Push to remote

Files involved:

- the local FI preparation commit

Command shape:

```bash
git push origin main
```

Expected output artifact:

- the FI prep changes available on the remote repo

### 3. Pull from HPC

Files involved:

- the newly pushed FI preparation commit

Command shape:

```bash
ssh hyperion
cd /users/aczd097/git/vidiq-hpc
git pull --rebase origin main
```

Expected output artifact:

- Hyperion checkout updated with the FI preparation code

### 4. Run on HPC

Files involved:

- the FI preparation script
- the FI archive:
  - `/users/aczd097/archive/vidiq-hpc/data/image/fi/emotion_dataset.tar`

Command shape:

```bash
sbatch --export=DATASET=fi hpc/prepare_image_dataset.slurm
```

Expected output artifact:

- a repo-ready staged FI root under:
  - `/users/aczd097/archive/vidiq-hpc/data/image/fi/`
- including:
  - `metadata.csv`
  - `images/`
  - `preparation_manifest.json`

### 5. Commit results on HPC

Files involved:

- only tracked repo artifacts produced by the HPC run
- not the raw staged dataset under archive storage

Command shape:

```bash
git status --short
git add <tracked repo artifacts>
git commit -m "Add FI archive analytics report"
```

Expected output artifact:

- an HPC-side commit containing the report or other tracked repo outputs

### 6. Push back to remote

Files involved:

- the HPC-side results commit

Command shape:

```bash
git push origin main
```

Expected output artifact:

- the HPC-generated report/results available on the remote repo

### 7. Pull locally

Files involved:

- the pushed HPC results commit

Command shape:

```bash
git pull --rebase origin main
```

Expected output artifact:

- the report available in the local checkout for further analysis and decision-making

## Recommended next preparation task

The exact next downstream preparation task should be:

- implement tar-based FI staging in `scripts/prepare_image_dataset.py`

Specifically:

- recognize `emotion_dataset.tar`
- extract it into the FI staged root
- map `emotion_dataset/<label>/<file>.jpg` to `images/<label>/<file>.jpg`
- generate `metadata.csv`
- validate the staged root with the existing FI loader
