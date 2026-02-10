#!/bin/bash
#SBATCH --job-name=order_ensemble_model
#SBATCH --account=project_2012747
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8000
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:1
#SBATCH --output=logs_ensemble_model/%x_%j.out
#SBATCH --error=logs_ensemble_model/%x_%j.err

set -euo pipefail

module purge

PYTHON=/projappl/project_2012747/mars/mars_env/bin/python
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRATCH=/scratch/project_2012747

# Put *everything* write-heavy on SCRATCH
export RUN_ROOT="$SCRATCH/mars_runs/ensemble_model/${SLURM_JOB_ID}"
export TORCH_HOME="$RUN_ROOT/torch_cache"
export HF_HOME="$RUN_ROOT/hf_cache"
export XDG_CACHE_HOME="$RUN_ROOT/xdg_cache"
export TMPDIR="$RUN_ROOT/tmp"

mkdir -p "$RUN_ROOT" "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
mkdir -p logs_ensemble_model


srun --kill-on-bad-exit=1 bash -lc "time $PYTHON -u use_Ensemble.py"
