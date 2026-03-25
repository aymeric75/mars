#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/random/projects/MarS"
RUN_ROOT="$PROJECT_ROOT/mars_runs/ensemble_model"
LOG_ROOT="$PROJECT_ROOT/custom_code/training/ensemble_model/logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

TRAIN_DIR="${TRAIN_DIR:-$PROJECT_ROOT/data/train}"
VAL_DIR="${VAL_DIR:-$PROJECT_ROOT/data/val}"
#ORDER_MODEL_CKPT="${ORDER_MODEL_CKPT:-}"
#VQ_CKPT_DIR="${VQ_CKPT_DIR:-}"
ORDER_MODEL_CKPT="/home/random/projects/MarS/mars_runs/order_model/tensorboard/bs=8_lr=1e-4/step=step=13920-val=val_loss=3.2903.ckpt"
VQ_CKPT_DIR="/home/random/projects/MarS/mars_runs/vqgan/2500steps/tensorboard/bs=8_lr=1e-5"

if [[ -z "$ORDER_MODEL_CKPT" ]]; then
  echo "Set ORDER_MODEL_CKPT to a trained Order Model .ckpt path" >&2
  exit 1
fi

if [[ -z "$VQ_CKPT_DIR" ]]; then
  echo "Set VQ_CKPT_DIR to a VQ checkpoint directory containing .ckpt files" >&2
  exit 1
fi

echo "$TRAIN_DIR"
echo "$RUN_ROOT"

LRS=(3e-5 5e-5)
BSS=(4 8)

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
      "$PYTHON" -u train_Ensemble_Model_hypersearch.py \
        --train_dir "$TRAIN_DIR" \
        --val_dir "$VAL_DIR" \
        --pattern "*messages*.parquet" \
        --order_model_ckpt "$ORDER_MODEL_CKPT" \
        --vq_ckpt_dir "$VQ_CKPT_DIR" \
        --batch_size "$BS" \
        --lr "$LR" \
        --cache_size 2 \
        --train_num_samples "$((1500 * BS))" \
        --train_chunk_size 128 \
        --val_num_samples 256 \
        --val_chunk_size 128 \
        --max_steps 5000 \
        --num_workers 0 \
        --precision bf16-mixed \
        --accumulate_grad_batches 4 \
        --gradient_clip_val 1.0 \
        --val_check_interval 200 \
        --limit_val_batches 64 \
        --run_root "$RUN_ROOT" \
        --run_name "$RUN_NAME"
    ) >"$OUT_FILE" 2>"$ERR_FILE"

    echo "Finished $RUN_NAME"
  done
done
