#!/bin/bash
#SBATCH --job-name=ensemble_model
#SBATCH --account=project_2012747
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --output=logs_ensemble_model/%x_%j.out
#SBATCH --error=logs_ensemble_model/%x_%j.err

set -euo pipefail
mkdir -p logs

module purge

PYTHON=/projappl/project_2012747/mars/mars_env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

$PYTHON -c "import sys; print('python:', sys.executable)"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
$PYTHON -c "import lightning; print('lightning:', lightning.__version__)"

srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u train_Ensemble_Model_lightning.py \
  --parquets_dir ../../data/ensemble/parquets \
  --next1s_dir ../../data/ensemble/next1s \
  --array_path arr_0 \
  --batch_size 64 \
  --order_vocab_size 49152 \
  --batch_vocab_size 8192 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 8 \
  --lr 3e-4 \
  --max_steps 20000 \
  --num_workers 4 \
  --precision bf16-mixed"
