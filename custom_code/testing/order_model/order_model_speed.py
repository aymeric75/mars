from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_code.preprocessing.order_model.messages_to_features_no_engine import (
    NUM_BINS_ORDER_INTERVAL,
    NUM_BINS_ORDER_VOLUME,
    NUM_BINS_PRICE_LEVEL,
    add_lob_volumes,
    bins_lob_volumes,
    create_mars_order_type_column,
    create_mid_price_column,
    create_price_change_to_open,
    create_seconds_since_open,
    create_slots_columns,
)
from custom_code.testing.utils import load_order_model
from market_simulation.models.utils import read_parquet_row_slice


MARKET_OPEN_NS = (9 * 60 * 60 + 30 * 60) * 1_000_000_000
MARKET_CLOSE_NS = (16 * 60 * 60) * 1_000_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark order-model preprocessing and inference speed.")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "order_model" / "tensorboard" / "bs=8_lr=1e-4" / "step=step=13920-val=val_loss=3.2903.ckpt",
    )
    p.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data" / "test")
    p.add_argument("--message_file", type=Path, default=None)
    p.add_argument("--snapshot_file", type=Path, default=None)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--start_offset", type=int, default=0, help="Offset inside market-hours rows.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def pick_files(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.message_file is not None:
        snapshot_file = args.snapshot_file or Path(str(args.message_file).replace("_messages", "_snapshots"))
        if not snapshot_file.exists():
            raise FileNotFoundError(f"Missing snapshot file: {snapshot_file}")
        return args.message_file, snapshot_file
    message_file = next(iter(sorted(args.data_dir.glob("*_messages.parquet"))), None)
    if message_file is None:
        raise FileNotFoundError(f"No *_messages.parquet files found in {args.data_dir}")
    snapshot_file = Path(str(message_file).replace("_messages", "_snapshots"))
    if not snapshot_file.exists():
        raise FileNotFoundError(f"Missing snapshot file: {snapshot_file}")
    return message_file, snapshot_file


def read_market_bounds(message_file: Path) -> tuple[int, int]:
    times = pd.read_parquet(message_file, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
    rows = np.flatnonzero((times >= MARKET_OPEN_NS) & (times <= MARKET_CLOSE_NS))
    if rows.size == 0:
        raise RuntimeError(f"No market-hours rows found in {message_file}")
    return int(rows[0]), int(rows[-1] + 1 - rows[0])


def load_raw_rows(
    message_file: Path, snapshot_file: Path, market_start: int, start_offset: int, need_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    include_prev = start_offset > 0
    raw_start = market_start + start_offset - int(include_prev)
    raw_rows = need_rows + int(include_prev)
    msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
    snap_cols = ["Ask_Price_1", "Bid_Price_1"] + [f"{side}_Size_{i}" for side in ("Ask", "Bid") for i in range(1, 6)]
    return (
        read_parquet_row_slice(message_file, columns=msg_cols, start_row=raw_start, num_rows=raw_rows),
        read_parquet_row_slice(snapshot_file, columns=snap_cols, start_row=raw_start, num_rows=raw_rows),
        include_prev,
    )


def make_batch(messages: pd.DataFrame, snapshots: pd.DataFrame, include_prev: bool, seq_len: int, batch_size: int) -> torch.Tensor:
    messages, snapshots = messages.copy(), snapshots.copy()
    create_mars_order_type_column(messages)
    features = messages[["Time", "Mars_type", "Price", "Size"]].copy()
    create_mid_price_column(features, snapshots)
    create_slots_columns(features)
    features["f0"] = (
        features["Mars_type"] * NUM_BINS_PRICE_LEVEL * NUM_BINS_ORDER_VOLUME * NUM_BINS_ORDER_INTERVAL
        + features["bin_price"] * NUM_BINS_ORDER_VOLUME * NUM_BINS_ORDER_INTERVAL
        + features["bin_vol"] * NUM_BINS_ORDER_INTERVAL
        + features["bin_interval"]
    )
    features = features.drop(columns=["bin_price", "bin_vol", "seconds_since_prev", "bin_interval"])
    features["vol_ratio_slot"] = 0
    features["trans_ratio_slot"] = 0
    create_price_change_to_open(features)
    features = features.drop(columns=["mid_price"])
    create_seconds_since_open(features)
    add_lob_volumes(features, snapshots)
    features = features.drop(columns=["Time", "Mars_type", "Price", "Size"]).fillna(0)
    bins_lob_volumes(features)
    array = features.to_numpy(dtype=np.int64, copy=True)[1:] if include_prev else features.to_numpy(dtype=np.int64, copy=True)
    if len(array) < seq_len + batch_size - 1:
        raise RuntimeError("Not enough rows to build the requested batch.")
    return torch.tensor(np.stack([array[i : i + seq_len] for i in range(batch_size)]), dtype=torch.long)


def bench(fn, repeat: int, sync) -> tuple[object, float]:
    out, times = None, []
    for _ in range(repeat):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t0)
    return out, 1e3 * float(np.mean(times))


def run_inference(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(x)


def main() -> None:
    args = parse_args()
    device = args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    if args.seq_len <= 1 or args.batch_size <= 0 or args.warmup < 0 or args.repeat <= 0 or args.start_offset < 0:
        raise ValueError("Invalid benchmark arguments.")

    message_file, snapshot_file = pick_files(args)
    market_start, market_rows = read_market_bounds(message_file)
    need_rows = args.seq_len + args.batch_size - 1
    if args.start_offset + need_rows > market_rows:
        raise ValueError(f"Need {need_rows} market rows from offset {args.start_offset}, but only {market_rows} are available.")

    messages, snapshots, include_prev = load_raw_rows(message_file, snapshot_file, market_start, args.start_offset, need_rows)
    model = load_order_model(str(args.ckpt), device=device, K=args.seq_len).eval()
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    for _ in range(args.warmup):
        x = make_batch(messages, snapshots, include_prev, args.seq_len, args.batch_size)
        run_inference(model, x.to(device))

    x_cpu, preprocess_ms = bench(
        lambda: make_batch(messages, snapshots, include_prev, args.seq_len, args.batch_size),
        args.repeat,
        lambda: None,
    )
    x = x_cpu.to(device)
    _, inference_ms = bench(lambda: run_inference(model, x), args.repeat, sync)

    print(f"file={message_file.name}")
    print(f"seq_len={args.seq_len} batch_size={args.batch_size} device={device}")
    print(f"preprocess_ms={preprocess_ms:.3f}")
    print(f"inference_ms={inference_ms:.3f}")
    print(f"total_ms={preprocess_ms + inference_ms:.3f}")


if __name__ == "__main__":
    main()
