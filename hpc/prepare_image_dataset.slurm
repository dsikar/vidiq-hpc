#!/bin/bash
#SBATCH -D /users/aczd097/git/vidiq-hpc
#SBATCH --job-name imgprep
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.sikar@city.ac.uk
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH -e outputs/%x_%j.e
#SBATCH -o outputs/%x_%j.o

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATASET="${DATASET:-emotion6}"
SOURCE_ROOT="${SOURCE_ROOT:-}"
SAMPLE_CHECKS="${SAMPLE_CHECKS:-8}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/users/aczd097/sharedscratch/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export VIDIQ_IMAGE_DATA_ROOT_EMOTION6="${VIDIQ_IMAGE_DATA_ROOT_EMOTION6:-/users/aczd097/archive/vidiq-hpc/data/image/emotion6}"
export VIDIQ_IMAGE_DATA_ROOT_FI="${VIDIQ_IMAGE_DATA_ROOT_FI:-/users/aczd097/archive/vidiq-hpc/data/image/fi}"
source /users/aczd097/sharedscratch/venvs/main/bin/activate
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

start=$(date +%s)
mkdir -p outputs
mkdir -p "${HF_DATASETS_CACHE}"
mkdir -p "${VIDIQ_IMAGE_DATA_ROOT_EMOTION6}"
mkdir -p "${VIDIQ_IMAGE_DATA_ROOT_FI}"

echo "Job started at $(date)"
echo "Dataset prep target: ${DATASET}"
echo "Source root: ${SOURCE_ROOT:-"(none)"}"
echo "Node: $(hostname)"
echo "HF_HOME: ${HF_HOME}"
echo "HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"

ARGS=(
  --dataset "${DATASET}"
  --sample-checks "${SAMPLE_CHECKS}"
)

if [[ -n "${SOURCE_ROOT}" ]]; then
  ARGS+=(--source-root "${SOURCE_ROOT}")
fi

python3 scripts/prepare_image_dataset.py "${ARGS[@]}"

end=$(date +%s)
diff=$((end-start))
echo "Completed at $(date) - $(( diff/60 ))m $(( diff%60 ))s"

tail -n 50 "outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.o" | \
    mail -s "${SLURM_JOB_NAME} (${SLURM_JOB_ID}) done" daniel.sikar@city.ac.uk
