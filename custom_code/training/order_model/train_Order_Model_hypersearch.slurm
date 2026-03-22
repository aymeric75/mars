#!/bin/bash
#SBATCH --job-name=order_model_grid
#SBATCH --account=project_2018243
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --output=logs_order_model/%x_%j.out
#SBATCH --error=logs_order_model/%x_%j.err

set -euo pipefail
module purge

PYTHON=/projappl/project_2012747/mars/mars_env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH=/scratch/project_2012747 

# Shared root for this whole job
export TB_ROOT="$SCRATCH/mars_runs/order_model/tb_seq/${SLURM_JOB_ID}"
mkdir -p "$TB_ROOT"

# Per-run heavy write/cache dir (recreated per run)
BASE_RUN_ROOT="$SCRATCH/mars_runs/order_model/tmp_${SLURM_JOB_ID}"

export CUDA_LAUNCH_BLOCKING=1

# Fail fast if env is wrong
$PYTHON -c "import sys; print('python:', sys.executable)"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
$PYTHON -c "import lightning; print('lightning:', lightning.__version__)"


echo "SLURM_JOB_ID=${SLURM_JOB_ID-}"
echo "SLURM_NTASKS=${SLURM_NTASKS-}"
echo "SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE-}"
echo "SLURM_GPUS=${SLURM_GPUS-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}"




# ---- Hyperparameter grid (edit freely) ----
LRS=(1e-4) # 3e-4 1e-3
BSS=(8)  #(8 16)

for LR in "${LRS[@]}"; do
  for BS in "${BSS[@]}"; do

    RUN_NAME="bs=${BS}_lr=${LR}"
    echo "======================================"
    echo "Starting run: ${RUN_NAME}"
    echo "======================================"

    # Isolated temp/cache directory per run
    RUN_ROOT="${BASE_RUN_ROOT}/${RUN_NAME}"
    export TORCH_HOME="$RUN_ROOT/torch_cache"
    export HF_HOME="$RUN_ROOT/hf_cache"
    export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
    export TMPDIR="$RUN_ROOT/tmp"

    mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

    srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u train_Order_Model_hypersearch.py \
      --train_dir /scratch/project_2012747/mars_data/order_model/train/raw_tmp \
      --val_dir /scratch/project_2012747/mars_data/order_model/val/raw_tmp \
      --model_variant base \
      --K 1024 \
      --batch_size ${BS} \
      --lr ${LR} \
      --max_steps 4000 \
      --num_workers 4 \
      --precision 32-true \
      --run_root ${TB_ROOT} \
      --run_name ${RUN_NAME}"

    echo "Finished run: ${RUN_NAME}"
    echo ""

  done
done

echo "All grid runs completed."
