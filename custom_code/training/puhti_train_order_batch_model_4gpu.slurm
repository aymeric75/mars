#!/bin/bash
#SBATCH --job-name=order_model
#SBATCH --account=project_2012747
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:v100:2
#SBATCH --output=logs_batch_model/%x_%j.out
#SBATCH --error=logs_batch_model/%x_%j.err

set -euo pipefail
mkdir -p logs

module purge

PYTHON=/projappl/project_2012747/mars/mars_env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# fail fast if env is wrong
$PYTHON -c "import sys; print('python:', sys.executable)"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
$PYTHON -c "import lightning; print('lightning:', lightning.__version__)"

srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u train_Order_Batch_Model_lightning.py \
  --train_dir ../../data/order_batch_model \
  --pattern "*tokens_v2.zarr.zip" \
  --seq_len_batches 16 \
  --batch_size 8 \
  --precision bf16-mixed"

