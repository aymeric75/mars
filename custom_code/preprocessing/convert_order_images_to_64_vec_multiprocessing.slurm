#!/bin/bash
#SBATCH --job-name=convert_order_images_to_64_vec
#SBATCH --account=project_2017808
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs_tests/%x_%j.out
#SBATCH --error=logs_tests/%x_%j.err

# -------------------------
# Threads for parallel libraries
# -------------------------
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# -------------------------
# Optional: your project path in PYTHONPATH
# -------------------------
export PYTHONPATH=$(pwd)/..

# -------------------------
# Load required modules
# -------------------------
module purge
module load gcc/11.3.0
module load python-data/3.12-25.09
# module load rocm/5.4.6   # Only if needed for GPU

# -------------------------
# Activate your environment
# -------------------------
source /projappl/project_2012747/mars/mars_env_clean/bin/activate
PYTHON=/projappl/project_2012747/mars/mars_env_clean/bin/python

# -------------------------
# Print debug info
# -------------------------
echo "Started job on $(hostname) at $(date)"
echo "Python executable: $(which python)"
python --version
python -c "import numpy, torch, pandas, pyarrow, pytorch_lightning, omegaconf; print('All imports OK')"

# -------------------------
# List of symbols
# -------------------------
SYMBOLS=('AAPL'  'AMD' 'AMZN'  'ASML'  'AVGO'  'COST'  'GOOG_'  'GOOGL'  'INTC'  'META'  'MSFT'  'NFLX'  'NVDA'  'PLTR'  'TSLA')

#('AAPL'  'AMD' 'AMZN'  'ASML'  'AVGO'  'COST'  'GOOG_'  'GOOGL'  'INTC'  'META'  'MSFT'  'NFLX'  'NVDA'  'PLTR'  'TSLA') ['GOOGL', 'NVDA', 'PLTR', 'TSLA']



# -------------------------
# Loop over symbols and submit a job for each
# -------------------------
for SYMBOL in "${SYMBOLS[@]}"; do
    echo "Submitting job for symbol: $SYMBOL"

    # Optional: generate a per-symbol job name
    JOB_NAME="encode_${SYMBOL}"

    # Submit the job
    sbatch --job-name=$JOB_NAME --export=SYMBOL=$SYMBOL <<EOF
#!/bin/bash
#SBATCH --account=project_2017808
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --output=logs_tests/%x_%j.out
#SBATCH --error=logs_tests/%x_%j.err


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$(pwd)/..

module purge
module load gcc/11.3.0
module load python-data/3.12-25.09
# module load rocm/5.4.6

source /projappl/project_2012747/mars/mars_env_clean/bin/activate
PYTHON=/projappl/project_2012747/mars/mars_env_clean/bin/python

echo "Running symbol: $SYMBOL"
srun $PYTHON /projappl/project_2012747/mars/MarS/custom_code/preprocessing/convert_order_images_to_64_vec.py --symbol $SYMBOL
EOF

done