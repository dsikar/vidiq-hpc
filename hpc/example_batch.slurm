#!/bin/bash
#SBATCH -D /users/aczd097
#SBATCH --job-name smoketest
#SBATCH --mail-type=ALL
#SBATCH --mail-user=daniel.sikar@city.ac.uk
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:a100:1
#SBATCH -e outputs/%x_%j.e
#SBATCH -o outputs/%x_%j.o

# Environment
# Environment
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /users/aczd097/sharedscratch/venvs/main/bin/activate

# Start
start=$(date +%s)
mkdir -p outputs
echo "Job started at $(date)"

# Smoke tests
echo "--- Python ---"
python --version

echo "--- PyTorch + CUDA ---"
python -c "
import torch
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
t = torch.randn(1000, 1000).cuda()
print('Tensor on GPU:', t.device)
"

echo "--- Transformers ---"
python -c "import transformers; print('Transformers:', transformers.__version__)"

echo "--- HuggingFace cache dir ---"
echo "HF_HOME: $HF_HOME"

# Duration + email
end=$(date +%s)
diff=$((end-start))
echo "Completed at $(date) — $(( diff/60 ))m $(( diff%60 ))s"

tail -n 50 "outputs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.o" | \
    mail -s "smoketest (${SLURM_JOB_ID}) done" daniel.sikar@city.ac.uk
