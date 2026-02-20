#!/bin/bash
#SBATCH --job-name=convert_order_images_to_64_vec
#SBATCH --account=project_2012747
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs_tests/%x_%j.out
#SBATCH --error=logs_tests/%x_%j.err


# Threads for parallel libraries
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional: your project path in PYTHONPATH
export PYTHONPATH=$(pwd)/..

# Load required modules
module purge
module load gcc/11.3.0
module load python-data/3.12-25.09
module load rocm/5.4.6                 # Only if using GPU

# Activate your project-specific Python environment
source /projappl/project_2012747/mars/mars_env_clean/bin/activate
PYTHON=/projappl/project_2012747/mars/mars_env_clean/bin/python

# Print info for debugging
echo "Started job on $(hostname) at $(date)"
echo "Python executable: $(which python)"
python --version
python -c "import numpy, torch, pandas, pyarrow, pytorch_lightning, omegaconf; print('All imports OK')"

# Run your Python script
srun $PYTHON /projappl/project_2012747/mars/MarS/custom_code/preprocessing/convert_order_images_to_64_vec.py
