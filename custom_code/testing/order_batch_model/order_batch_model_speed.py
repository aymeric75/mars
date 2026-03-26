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

from custom_code.preprocessing.order_batch_model.messages_to_order_images import (
    MARKET_CLOSE_NS,
    MARKET_OPEN_NS,
    ONE_MINUTE_NS,
    chunks_to_order_images,
    compute_valid_anchor_indices,
    create_mars_order_type_column,
    create_mid_price_column,
    create_slots_columns,
    from_messages_and_snapshots_to_features,
    retrieve_chunk_last_16min_from_df,
)
from custom_code.testing.order_batch_model.order_batch_model_test import (
    encode_images_to_tokens,
    load_vq_model,
)
from custom_code.testing.utils import load_order_batch_model
from market_simulation.models.utils import read_parquet_row_slice


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark order-batch preprocessing and inference speed.")
    p.add_argument(
        "--order_batch_ckpt",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "order_batch_model" / "tensorboard" / "bs=2_lr=1e-4" / "step=step=0-val=val_loss=1.8096.ckpt",
    )
    p.add_argument(
        "--vq_ckpt",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "vqgan" / "2500steps" / "tensorboard" / "bs=8_lr=1e-5" / "step=4606-val_rec_loss=0.038047.ckpt",
    )
    p.add_argument(
        "--vq_config",
        type=Path,
        default=REPO_ROOT / "third_party" / "latent_diffusion" / "models" / "first_stage_models" / "vq-f4" / "config.yaml",
    )
    p.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data" / "test")
    p.add_argument("--message_file", type=Path, default=None)
    p.add_argument("--snapshot_file", type=Path, default=None)
    p.add_argument("--anchor_choice", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
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


def load_raw_slice(message_file: Path, snapshot_file: Path, anchor_choice: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    times = pd.read_parquet(message_file, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
    market_rows = np.flatnonzero((times >= MARKET_OPEN_NS) & (times <= MARKET_CLOSE_NS))
    if market_rows.size == 0:
        raise RuntimeError(f"No market-hours rows found in {message_file}")

    market_start = int(market_rows[0])
    market_times = times[market_start : int(market_rows[-1] + 1)]
    valid = compute_valid_anchor_indices(market_times)
    if valid.size == 0:
        raise RuntimeError(f"No valid 16-minute anchors found in {message_file}")
    if not 0 <= anchor_choice < len(valid):
        raise IndexError(f"anchor_choice={anchor_choice} out of range for {len(valid)} valid anchors")

    anchor_idx = int(valid[anchor_choice])
    slice_start = int(np.searchsorted(market_times, market_times[anchor_idx] - 16 * ONE_MINUTE_NS, side="left"))
    raw_start = market_start + slice_start
    raw_rows = anchor_idx - slice_start + 1
    msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
    snap_cols = ["Ask_Price_1", "Bid_Price_1"]
    return (
        read_parquet_row_slice(message_file, columns=msg_cols, start_row=raw_start, num_rows=raw_rows),
        read_parquet_row_slice(snapshot_file, columns=snap_cols, start_row=raw_start, num_rows=raw_rows),
        anchor_idx - slice_start,
    )


def feature_from_last_raw_row(message_row: pd.DataFrame, snapshot_row: pd.DataFrame) -> pd.DataFrame:
    messages = message_row.copy()
    features = messages[["Time", "Price", "Size"]].copy()
    create_mars_order_type_column(messages)
    features["Mars_type"] = messages["Mars_type"].to_numpy(copy=False)
    create_mid_price_column(features, snapshot_row)
    create_slots_columns(features)
    return features[["Time", "Mars_type", "bin_price", "bin_vol"]].fillna(0).astype(
        {"Time": "int64", "Mars_type": "int32", "bin_price": "int32", "bin_vol": "int32"}
    )


def preprocess(history_features: pd.DataFrame, last_message: pd.DataFrame, last_snapshot: pd.DataFrame, vq_model: torch.nn.Module, device: str) -> tuple[torch.Tensor, int]:
    last_feature = feature_from_last_raw_row(last_message, last_snapshot)
    features = pd.concat([history_features, last_feature], ignore_index=True)
    features = features.loc[features["Time"] >= int(features["Time"].iat[-1] - 16 * ONE_MINUTE_NS)].reset_index(drop=True)
    context_images = chunks_to_order_images(retrieve_chunk_last_16min_from_df(features, len(features) - 1))
    context_tokens, _ = encode_images_to_tokens(vq_model, context_images, device=device)
    return context_tokens.reshape(-1), int(context_tokens.shape[1])


def bench(fn, repeat: int, sync) -> tuple[object, float]:
    out, times = None, []
    for _ in range(repeat):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t0)
    return out, 1e3 * float(np.mean(times))


def run_inference(model: torch.nn.Module, prefix_tokens: torch.Tensor, device: str) -> torch.Tensor:
    with torch.inference_mode():
        return model.top_next(prefix_tokens.unsqueeze(0).to(device=device, dtype=torch.long))


def main() -> None:
    args = parse_args()
    device = args.device if not args.device.startswith("cuda") or torch.cuda.is_available() else "cpu"
    if args.warmup < 0 or args.repeat <= 0:
        raise ValueError("Invalid benchmark arguments.")

    message_file, snapshot_file = pick_files(args)
    messages, snapshots, anchor_idx = load_raw_slice(message_file, snapshot_file, args.anchor_choice)
    history_features = from_messages_and_snapshots_to_features(messages.iloc[:-1], snapshots.iloc[:-1])
    last_message = messages.iloc[[-1]]
    last_snapshot = snapshots.iloc[[-1]]
    if anchor_idx != len(messages) - 1:
        raise RuntimeError("Expected the selected anchor to be the last row of the loaded raw slice.")
    order_batch_model = load_order_batch_model(str(args.order_batch_ckpt), device=device).to(device).eval()
    vq_model = load_vq_model(args.vq_ckpt, args.vq_config, device=device)
    sync = torch.cuda.synchronize if device.startswith("cuda") else (lambda: None)

    for _ in range(args.warmup):
        prefix_tokens, tokens_per_image = preprocess(history_features, last_message, last_snapshot, vq_model, device)
        run_inference(order_batch_model, prefix_tokens, device)

    (prefix_tokens, tokens_per_image), preprocess_ms = bench(
        lambda: preprocess(history_features, last_message, last_snapshot, vq_model, device),
        args.repeat,
        sync,
    )
    _, inference_ms = bench(
        lambda: run_inference(order_batch_model, prefix_tokens, device),
        args.repeat,
        sync,
    )

    print(f"file={message_file.name}")
    print(f"anchor_choice={args.anchor_choice} device={device}")
    print(f"prefix_tokens={prefix_tokens.numel()} tokens_per_image={tokens_per_image}")
    print(f"preprocess_ms={preprocess_ms:.3f}")
    print(f"one_token_inference_ms={inference_ms:.3f}")
    print(f"total_ms={preprocess_ms + inference_ms:.3f}")


if __name__ == "__main__":
    main()
