#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/random/projects/MarS"


DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data/test}"
ORDER_MODEL_CKPT="${ORDER_MODEL_CKPT:-$PROJECT_ROOT/mars_runs/order_model/tensorboard/bs=8_lr=1e-4/step=step=13920-val=val_loss=3.2903.ckpt}"
DEVICE="${DEVICE:-cpu}"
MAX_FILES=1
MAX_WINDOWS_PER_FILE="${MAX_WINDOWS_PER_FILE:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_ROWS_PER_SLICE="${MAX_ROWS_PER_SLICE:-4096}"

cd "$SCRIPT_DIR"

CMD=(
  python -u order_model_test.py
  --ckpt "$ORDER_MODEL_CKPT"
  --data_dir "$DATA_DIR"
  --device "$DEVICE"
  --max_files "$MAX_FILES"
  --batch_size "$BATCH_SIZE"
  --max_windows_per_file "$MAX_WINDOWS_PER_FILE"
  --max_rows_per_slice "$MAX_ROWS_PER_SLICE"
)

"${CMD[@]}" \
  "$@"
