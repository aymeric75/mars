#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT="/home/random/projects/MarS"
RUN_ROOT="$PROJECT_ROOT/mars_runs/order_model"
LOG_ROOT="$PROJECT_ROOT/custom_code/training/order_model/logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

TRAIN_DIR="$PROJECT_ROOT/data/train"
VAL_DIR="$PROJECT_ROOT/data/val"

echo $TRAIN_DIR
echo $RUN_ROOT


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

    python -u train_Order_Model_hypersearch.py \
      --train_dir "$TRAIN_DIR" \
      --val_dir "$VAL_DIR" \
      --cache_size 2 \
      --train_num_samples "$((20000 * BS))" \
      --train_chunk_size 2048 \
      --model_variant base \
      --K 1024 \
      --batch_size "$BS" \
      --lr "$LR" \
      --max_steps 20000 \
      --num_workers 2 \
      --precision bf16-mixed \
      --matmul_precision high \
      --no-deterministic \
      --run_root "$RUN_ROOT" \
      --run_name "$RUN_NAME" \
      >"$OUT_FILE" 2>"$ERR_FILE"

    echo "Finished $RUN_NAME"
  done
done
