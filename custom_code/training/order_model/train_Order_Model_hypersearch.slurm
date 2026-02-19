#!/bin/bash
#SBATCH --job-name=order_model_grid
#SBATCH --account=project_2012747
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=256G
#SBATCH --array=0-5
#SBATCH --output=logs_order_model/%x_%A_%a.out
#SBATCH --error=logs_order_model/%x_%A_%a.err

set -euo pipefail
module purge

PYTHON=/projappl/project_2012747/mars/mars_env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH=/scratch/project_2012747

# Put write-heavy caches on scratch (per array task)
export RUN_ROOT="$SCRATCH/mars_runs/order_model/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
export TORCH_HOME="$RUN_ROOT/torch_cache"
export HF_HOME="$RUN_ROOT/hf_cache"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export TMPDIR="$RUN_ROOT/tmp"
export CUDA_LAUNCH_BLOCKING=1

mkdir -p "$RUN_ROOT" "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

# Shared place for all tensorboard logs across the whole grid:
export TB_ROOT="$SCRATCH/mars_runs/order_model/tb_grid/${SLURM_ARRAY_JOB_ID}"
mkdir -p "$TB_ROOT"

# ---- define your grid here ----
LRS=(1e-4 3e-4 1e-3)
BSS=(16 32)   # <= pick what you want

n_lr=${#LRS[@]}
n_bs=${#BSS[@]}
total=$((n_lr * n_bs))

tid=${SLURM_ARRAY_TASK_ID}
if [ "$tid" -ge "$total" ]; then
  echo "ERROR: task id $tid >= total combos $total"
  exit 1
fi

lr_idx=$(( tid / n_bs ))
bs_idx=$(( tid % n_bs ))

LR=${LRS[$lr_idx]}
BS=${BSS[$bs_idx]}

RUN_NAME="bs=${BS}_lr=${LR}"
echo "Running combo: ${RUN_NAME}"
echo "TensorBoard root: ${TB_ROOT}"

# fail fast if env is wrong
$PYTHON -c "import sys; print('python:', sys.executable)"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
$PYTHON -c "import lightning; print('lightning:', lightning.__version__)"

srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u train_Order_Model_lightning.py \
  --train_dir /scratch/project_2012747/mars_data/order_model/train/final \
  --val_dir /scratch/project_2012747/mars_data/order_model/val/final \
  --model_variant base \
  --K 1024 \
  --batch_size ${BS} \
  --lr ${LR} \
  --max_steps 4000 \
  --num_workers 4 \
  --precision bf16-mixed \
  --run_root ${TB_ROOT} \
  --run_name ${RUN_NAME}"
