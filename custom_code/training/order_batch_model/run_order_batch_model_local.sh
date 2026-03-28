#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/random/projects/MarS"
RUN_ROOT="$PROJECT_ROOT/mars_runs/order_batch_model"
LOG_ROOT="$PROJECT_ROOT/custom_code/training/order_batch_model/logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

TRAIN_DIR="$PROJECT_ROOT/data/train"
VAL_DIR="$PROJECT_ROOT/data/val"
VQ_CKPT_DIR="$PROJECT_ROOT/mars_runs/vqgan/tensorboard/bs=8_lr=1.5e-5"

# 

echo "$TRAIN_DIR"
echo "$RUN_ROOT"

LRS=(1e-4 3e-4)
BSS=(2 4 8)

for LR in "${LRS[@]}"; do
  for BS in "${BSS[@]}"; do
    RUN_NAME="bs=${BS}_lr=${LR}"
    OUT_FILE="$LOG_ROOT/${RUN_NAME}.out"
    ERR_FILE="$LOG_ROOT/${RUN_NAME}.err"

    echo "Starting $RUN_NAME"
    echo "stdout -> $OUT_FILE"
    echo "stderr -> $ERR_FILE"

    (
      cd "$SCRIPT_DIR"
      "$PYTHON" -u train_Order_Batch_Model_hypersearch.py \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR" \
        --pattern "*messages*.parquet" \
        --vq_ckpt_dir "$VQ_CKPT_DIR" \
        --batch_size "$BS" \
        --lr "$LR" \
        --cache_size 2 \
        --val_num_samples "$((20 * BS))" \
        --temporal_block_minutes 15 \
        --min_anchor_spacing_seconds 30 \
        --train_chunk_size 128 \
        --val_chunk_size 128 \
        --max_steps 10000 \
        --num_workers 0 \
        --precision bf16-mixed \
        --run_root "$RUN_ROOT" \
        --run_name "$RUN_NAME"
    ) >"$OUT_FILE" 2>"$ERR_FILE"

    echo "Finished $RUN_NAME"
  done
done
