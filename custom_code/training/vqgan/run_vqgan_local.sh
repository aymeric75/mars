#!/usr/bin/env bash
set -euo pipefail

PYTHON=python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

PROJECT_ROOT="/home/random/projects/MarS"
RUN_ROOT="$PROJECT_ROOT/mars_runs/vqgan"
LOG_ROOT="$PROJECT_ROOT/custom_code/training/vqgan/logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

TRAIN_DIR="$PROJECT_ROOT/data/train"
VAL_DIR="$PROJECT_ROOT/data/val"
CONVERTERS_JSON="$PROJECT_ROOT/custom_code/training/converters_portable.json"
VQ_CONFIG="$PROJECT_ROOT/third_party/latent_diffusion/models/first_stage_models/vq-f4/config.yaml"

# Point this to the pretrained VQ checkpoint you want to fine-tune from.
INIT_CKPT="$PROJECT_ROOT/third_party/latent_diffusion/models/first_stage_models/vq-f4/model.ckpt"

LRS=(1e-5 1.5e-5)
BSS=(2 4 8)

for LR in "${LRS[@]}"; do
  for BS in "${BSS[@]}"; do
    RUN_NAME="bs=${BS}_lr=${LR}"
    OUT_FILE="$LOG_ROOT/${RUN_NAME}.out"
    ERR_FILE="$LOG_ROOT/${RUN_NAME}.err"

    echo "Starting $RUN_NAME"
    echo "stdout -> $OUT_FILE"
    echo "stderr -> $ERR_FILE"

    ARGS=(
      -u "$PROJECT_ROOT/custom_code/training/vqgan/train_vqgan_hypersearch.py"
      --train_dir "$TRAIN_DIR"
      --val_dir "$VAL_DIR"
      --converter_json_path "$CONVERTERS_JSON"
      --vq_config "$VQ_CONFIG"
      --batch_size "$BS"
      --lr "$LR"
      --max_steps 15000
      --num_workers 2
      --precision 32
      --matmul_precision high
      --no-deterministic
      --run_root "$RUN_ROOT"
      --run_name "$RUN_NAME"
    )

    if [[ -n "$INIT_CKPT" ]]; then
      ARGS+=(--init_ckpt "$INIT_CKPT")
    fi

    "$PYTHON" "${ARGS[@]}" >"$OUT_FILE" 2>"$ERR_FILE"

    echo "Finished $RUN_NAME"
  done
done
