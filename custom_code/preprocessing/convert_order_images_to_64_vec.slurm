#!/bin/bash
#SBATCH --job-name=convert_order_images_to_64_vec
#SBATCH --account=project_2012747
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs_tests/%x_%j.out
#SBATCH --error=logs_tests/%x_%j.err

set -euo pipefail


module purge



PYTHON=/projappl/project_2012747/mars/mars_env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


SCRATCH=/scratch/project_2012747
export RUN_ROOT="$SCRATCH/mars_runs/tests/${SLURM_JOB_ID}"
export TORCH_HOME="$RUN_ROOT/torch_cache"
export HF_HOME="$RUN_ROOT/hf_cache"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export TMPDIR="$RUN_ROOT/tmp"

mkdir -p "$RUN_ROOT" "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"


$PYTHON -c "import sys; print('python:', sys.executable)"
$PYTHON -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"
$PYTHON -c "import lightning; print('lightning:', lightning.__version__)"

srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u convert_order_images_to_64_vec.py"
