#!/bin/bash
#SBATCH -D /users/aczd097/git/vidiq-hpc
#SBATCH --job-name qpreval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.sikar@city.ac.uk
#SBATCH --partition=preemptgpu # gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24GB
#SBATCH --time=02:23:59
#SBATCH --gres=gpu:a100_80g:1 #pu:1 # gpu:a100:1
#SBATCH -e outputs/%x_%j.e
#SBATCH -o outputs/%x_%j.o

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export HF_HOME="${HF_HOME:-/users/aczd097/sharedscratch/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
source /users/aczd097/sharedscratch/venvs/main/bin/activate

start=$(date +%s)
mkdir -p outputs
echo "Job started at $(date)"
echo "Mode: extract-only"
echo "Model ref: Qwen/Qwen3-1.7B"
echo "Processed split: validation"
echo "HF_HOME: ${HF_HOME}"

python experiments/text_model/train_multiclass.py \
    --mode extract-only \
    --data-root experiments/text/multiclass/dair-ai-emotion \
    --data-split validation \
    --run-root experiments/text_model/runs \
    --run-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}" \
    --batch-size 8 \
    --max-length 256 \
    --model-path Qwen/Qwen3-1.7B

end=$(date +%s)
diff=$((end-start))
echo "Completed at $(date) - $(( diff/60 ))m $(( diff%60 ))s"

tail -n 50 "outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.o" | \
    mail -s "${SLURM_JOB_NAME} (${SLURM_JOB_ID}) done" daniel.sikar@city.ac.uk
