#!/bin/bash
#SBATCH --job-name=order_model
#SBATCH --account=project_2012747
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --time=06:15:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mem=256G
#SBATCH --output=logs_order_model/%x_%j.out
#SBATCH --error=logs_order_model/%x_%j.err

set -euo pipefail

module purge

PYTHON=/projappl/project_2012747/mars/mars_env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK



SCRATCH=/scratch/project_2012747 

# Put *everything* write-heavy on SCRATCH
export RUN_ROOT="$SCRATCH/mars_runs/order_model/${SLURM_JOB_ID}"
export TORCH_HOME="$RUN_ROOT/torch_cache"
export HF_HOME="$RUN_ROOT/hf_cache"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export TMPDIR="$RUN_ROOT/tmp"
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "$RUN_ROOT" "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"


# fail fast if env is wrong
$PYTHON -c "import sys; print('python:', sys.executable)"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
$PYTHON -c "import lightning; print('lightning:', lightning.__version__)"

srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u train_Order_Model_lightning.py \
  --train_dir /scratch/project_2012747/mars_data/order_model/train/final \
  --val_dir /scratch/project_2012747/mars_data/order_model/val/final \
  --model_variant base \
  --K 1024 \
  --batch_size 8 \
  --lr 3e-4 \
  --max_steps 2000 \
  --num_workers 4 \
  --precision bf16-mixed"
