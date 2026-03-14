#!/bin/bash
#SBATCH --job-name=mars_market_impact
#SBATCH --account=project_2018243
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --output=logs_tests/%x_%j.out
#SBATCH --error=logs_tests/%x_%j.err

set -euo pipefail
module purge

PYTHON=/projappl/project_2012747/mars/mars_env/bin/python

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH=/scratch/project_2012747
export RUN_ROOT="$SCRATCH/mars_runs/market_impact/${SLURM_JOB_ID}"
export TORCH_HOME="$RUN_ROOT/torch_cache"
export HF_HOME="$RUN_ROOT/hf_cache"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export TMPDIR="$RUN_ROOT/tmp"
export MARS_RAY_NAMESPACE="mars"

mkdir -p "$RUN_ROOT" "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" logs_tests


$PYTHON -c "import sys; print('python:', sys.executable)"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'n_gpu:', torch.cuda.device_count())"
$PYTHON -c "import ray; print('ray:', ray.__version__)"

srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u market_impact.py"