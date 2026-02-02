#!/bin/bash
#SBATCH --job-name=mars_order_ddp
#SBATCH --account=project_2012747
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

mkdir -p logs

PY=/projappl/project_2012747/mars/mars_env/bin/python
cd /projappl/project_2012747

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + ($SLURM_JOB_ID % 20000)))

export GLOO_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=ib0,bond0,eno1,eth0
export GLOO_SOCKET_IFNAME=ib0,bond0,eno1,eth0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN

srun $PY -m torch.distributed.run \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train_order_model_ddp.py
