#!/bin/bash
#SBATCH --job-name=ensemble_model_grid_single
#SBATCH --account=project_2017808
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --output=logs_ensemble_model/%x_%j.out
#SBATCH --error=logs_ensemble_model/%x_%j.err

set -euo pipefail
module purge

PYTHON=/projappl/project_2012747/mars/mars_env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH=/scratch/project_2012747

export TB_ROOT="$SCRATCH/mars_runs/ensemble_model/tb_seq_single/${SLURM_JOB_ID}"
mkdir -p "$TB_ROOT"

BASE_RUN_ROOT="$SCRATCH/mars_runs/ensemble_model/tmp_${SLURM_JOB_ID}"

$PYTHON -c "import sys; print('python:', sys.executable)"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
$PYTHON -c "import lightning; print('lightning:', lightning.__version__)"

mkdir -p logs_ensemble_model

# ---- Hyperparameter grid ----
LRS=(5e-4) #(1e-3 5e-4 2e-5) 
BSS=(8) #(4 8)

for LR in "${LRS[@]}"; do
  for BS in "${BSS[@]}"; do

    RUN_NAME="bs=${BS}_lr=${LR}"
    echo "======================================"
    echo "Starting run: ${RUN_NAME}"
    echo "======================================"

    RUN_ROOT="${BASE_RUN_ROOT}/${RUN_NAME}"
    export TORCH_HOME="$RUN_ROOT/torch_cache"
    export HF_HOME="$RUN_ROOT/hf_cache"
    export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
    export TMPDIR="$RUN_ROOT/tmp"

    mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

    srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u train_Ensemble_Model_lightning.py \
      --train_dir /scratch/project_2012747/mars_data/ensemble_model/train/final \
      --val_dir /scratch/project_2012747/mars_data/ensemble_model/val/final \
      --order_vocab_size 49152 \
      --batch_size ${BS} \
      --num_workers 5 \
      --lr ${LR} \
      --max_steps 10000 \
      --precision bf16-mixed \
      --default_root_dir ${TB_ROOT} \
      --run_name ${RUN_NAME}"

    echo "Finished run: ${RUN_NAME}"
    echo ""

  done
done

echo "All grid runs completed."