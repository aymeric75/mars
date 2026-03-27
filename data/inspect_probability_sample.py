from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from market_simulation.models.utils_heads import (
    OnlineReturnHeadDataset,
    buy_fill_price,
    sell_fill_price,
)
from market_simulation.models.utils_order_batch_model import VQRuntimeConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--sample_idx", type=int, required=True)
    parser.add_argument("--pattern", default="*messages*.parquet")
    parser.add_argument("--scenario", default="order_model", choices=["order_model", "order_batch", "both"])
    parser.add_argument("--trade_side", default="long", choices=["long", "short"])
    parser.add_argument("--trade_quantity", type=float, default=1.0)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--horizon_seconds", type=int, default=30)
    args = parser.parse_args()

    vq_runtime = None
    if args.scenario in {"order_batch", "both"}:
        vq_runtime = VQRuntimeConfig(ckpt_dir="", latent_diffusion_root="", taming_root="")

    message_files = sorted(str(path) for path in Path(args.data_dir).glob(args.pattern))

    dataset = OnlineReturnHeadDataset(
        message_files=message_files,
        seq_len=args.seq_len,
        scenario=args.scenario,
        horizon_seconds=args.horizon_seconds,
        cache_size=1,
        feature_chunk_size=128,
        sample_chunk_size=128,
        vq_runtime=vq_runtime,
    )

    file_idx, local_pos = dataset._locate_index(args.sample_idx)
    anchor_idx, future_idx = dataset._resolve_anchor_and_future_indices(file_idx, np.asarray([local_pos], dtype=np.int64))
    anchor_idx = int(anchor_idx[0])
    future_idx = int(future_idx[0])

    (
        current_mid,
        future_mid,
        target_return,
        ask_prices,
        bid_prices,
        ask_sizes,
        bid_sizes,
        future_ask_prices,
        future_bid_prices,
        future_ask_sizes,
        future_bid_sizes,
    ) = dataset._build_return_targets(
        file_idx,
        np.asarray([anchor_idx], dtype=np.int64),
        np.asarray([future_idx], dtype=np.int64),
    )

    current_mid = float(current_mid[0])
    future_mid = float(future_mid[0])
    target_return = float(target_return[0])
    ask_prices = ask_prices[0].tolist()
    bid_prices = bid_prices[0].tolist()
    ask_sizes = ask_sizes[0].tolist()
    bid_sizes = bid_sizes[0].tolist()
    future_ask_prices = future_ask_prices[0].tolist()
    future_bid_prices = future_bid_prices[0].tolist()
    future_ask_sizes = future_ask_sizes[0].tolist()
    future_bid_sizes = future_bid_sizes[0].tolist()

    if args.trade_side == "long":
        entry_fill = buy_fill_price(ask_prices=ask_prices, ask_sizes=ask_sizes, quantity=args.trade_quantity)
        exit_fill = sell_fill_price(bid_prices=future_bid_prices, bid_sizes=future_bid_sizes, quantity=args.trade_quantity)
        entry_cost = entry_fill - current_mid
        exit_cost = future_mid - exit_fill
        pnl = exit_fill - entry_fill
    else:
        entry_fill = sell_fill_price(bid_prices=bid_prices, bid_sizes=bid_sizes, quantity=args.trade_quantity)
        exit_fill = buy_fill_price(ask_prices=future_ask_prices, ask_sizes=future_ask_sizes, quantity=args.trade_quantity)
        entry_cost = current_mid - entry_fill
        exit_cost = exit_fill - future_mid
        pnl = entry_fill - exit_fill

    market_times = dataset._load_market_times(file_idx)
    anchor_time = int(market_times[anchor_idx])
    future_time = int(market_times[future_idx])
    raw_anchor_row = int(dataset.market_start_rows[file_idx] + anchor_idx)
    raw_future_row = int(dataset.market_start_rows[file_idx] + future_idx)

    print(f"message_file={dataset.message_files[file_idx]}")
    print(f"snapshot_file={dataset.snapshot_files[file_idx]}")
    print(f"sample_idx={args.sample_idx}")
    print(f"file_idx={file_idx}")
    print(f"local_pos={local_pos}")
    print(f"anchor_market_row={anchor_idx}")
    print(f"future_market_row={future_idx}")
    print(f"anchor_raw_row={raw_anchor_row}")
    print(f"future_raw_row={raw_future_row}")
    print(f"anchor_time={anchor_time}")
    print(f"future_time={future_time}")
    print(f"mid_price={current_mid}")
    print(f"future_mid_price={future_mid}")
    print(f"target_return={target_return}")
    print(f"entry_fill={entry_fill}")
    print(f"exit_fill={exit_fill}")
    print(f"entry_cost={entry_cost}")
    print(f"exit_cost={exit_cost}")
    print(f"full_cost={entry_cost + exit_cost}")
    print(f"pnl={pnl}")
    print(f"label={int(pnl > 0)}")


if __name__ == "__main__":
    main()
