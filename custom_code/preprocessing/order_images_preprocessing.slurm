#!/bin/bash
#SBATCH --job-name=order_images_preprocessing
#SBATCH --account=project_2012747
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=logs_tests/%x_%j.out
#SBATCH --error=logs_tests/%x_%j.err

set -euo pipefail

module purge

PYTHON=/projappl/project_2012747/mars/mars_env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH=/scratch/project_2012747 

# Put *everything* write-heavy on SCRATCH
export RUN_ROOT="$SCRATCH/mars_runs/tests/${SLURM_JOB_ID}"
export TORCH_HOME="$RUN_ROOT/torch_cache"
export HF_HOME="$RUN_ROOT/hf_cache"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export TMPDIR="$RUN_ROOT/tmp"


mkdir -p "$RUN_ROOT" "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"


# fail fast if env is wrong
$PYTHON -c "import sys; print('python:', sys.executable)"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
$PYTHON -c "import lightning; print('lightning:', lightning.__version__)"


srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u order_images_preprocessing.py"


