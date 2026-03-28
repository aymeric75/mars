#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/random/projects/MarS"
RUN_ROOT="$PROJECT_ROOT/mars_runs/heads"
LOG_ROOT="$PROJECT_ROOT/custom_code/training/heads/logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

TRAIN_DIR="${TRAIN_DIR:-$PROJECT_ROOT/data/train}"
VAL_DIR="${VAL_DIR:-$PROJECT_ROOT/data/val}"
SEED="${SEED:-42}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      if [[ $# -lt 2 ]]; then
        echo "Error: --seed requires a value" >&2
        exit 1
      fi
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    -s)
      if [[ $# -lt 2 ]]; then
        echo "Error: -s requires a value" >&2
        exit 1
      fi
      SEED="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--seed N]"
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      echo "Usage: $0 [--seed N]" >&2
      exit 1
      ;;
  esac
done

if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Error: --seed must be a non-negative integer" >&2
  exit 1
fi


HEAD_TYPE=probability # regression, multitask, probability
SCENARIO=order_model
TRADE_SIDE=long
TRAIN_SAMPLES_PER_EPOCH=512 #512
VAL_SAMPLES=256 #128
MAX_STEPS=5000
PNL_MARGIN=100

#HEAD_TYPE="${HEAD_TYPE:-regression}"
#SCENARIO="${SCENARIO:-order_model}"
#TRADE_SIDE="${TRADE_SIDE:-long}"
TRADE_QUANTITY="${TRADE_QUANTITY:-1.0}"
#TRAIN_SAMPLES_PER_EPOCH="${TRAIN_SAMPLES_PER_EPOCH:-3072}"
#VAL_SAMPLES="${VAL_SAMPLES:-1024}"

ORDER_MODEL_CKPT="${ORDER_MODEL_CKPT:-$PROJECT_ROOT/mars_runs/order_model/tensorboard/bs=8_lr=1e-4/step=step=13920-val=val_loss=3.2903.ckpt}"
ORDER_BATCH_CKPT="${ORDER_BATCH_CKPT:-$PROJECT_ROOT/mars_runs/order_batch_model/on_best_vqgan_0_031307/tensorboard/bs=4_lr=1e-4/step=step=0-val=val_loss=1.7026.ckpt}"
VQ_CKPT_DIR="${VQ_CKPT_DIR:-$PROJECT_ROOT/mars_runs/vqgan/2500steps/tensorboard/bs=8_lr=1e-5}"

echo "$TRAIN_DIR"
echo "$RUN_ROOT"

LRS=(1e-4 3e-4)
BSS=(8 16 32)
HIDDEN_DIMS=(128 256)

for LR in "${LRS[@]}"; do
  for BS in "${BSS[@]}"; do
    for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do
      RUN_NAME="head=${HEAD_TYPE}_scenario=${SCENARIO}_side=${TRADE_SIDE}_bs=${BS}_lr=${LR}_hd=${HIDDEN_DIM}_seed=${SEED}"
      OUT_FILE="$LOG_ROOT/${RUN_NAME}.out"
      ERR_FILE="$LOG_ROOT/${RUN_NAME}.err"

      echo "Starting $RUN_NAME"
      echo "stdout -> $OUT_FILE"
      echo "stderr -> $ERR_FILE"

      (
        cd "$SCRIPT_DIR"
        echo "RUN_NAME=$RUN_NAME"
        echo "SEED=$SEED"
        echo
        "$PYTHON" -u train_Heads_hypersearch.py \
          --train_dir "$TRAIN_DIR" \
          --val_dir "$VAL_DIR" \
          --pattern "*messages*.parquet" \
          --head_type "$HEAD_TYPE" \
          --scenario "$SCENARIO" \
          --trade_side "$TRADE_SIDE" \
          --trade_quantity "$TRADE_QUANTITY" \
          --pnl_margin "$PNL_MARGIN" \
          --order_model_ckpt "$ORDER_MODEL_CKPT" \
          --order_batch_ckpt "$ORDER_BATCH_CKPT" \
          --vq_ckpt_dir "$VQ_CKPT_DIR" \
          --hidden_dim "$HIDDEN_DIM" \
          --batch_size "$BS" \
          --lr "$LR" \
          --cache_size 2 \
          --train_samples_per_val "$TRAIN_SAMPLES_PER_EPOCH" \
          --train_chunk_size 32 \
          --val_num_samples "$VAL_SAMPLES" \
          --val_chunk_size 32 \
          --max_steps "$MAX_STEPS" \
          --seed "$SEED" \
          --num_workers 0 \
          --precision bf16-mixed \
          --accumulate_grad_batches 2 \
          --gradient_clip_val 1.0 \
          --run_root "$RUN_ROOT" \
          --run_name "$RUN_NAME"
      ) >"$OUT_FILE" 2>"$ERR_FILE"

      echo "Finished $RUN_NAME"
    done
  done
done
