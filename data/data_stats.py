from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from market_simulation.models.utils_heads import (
    OnlineReturnHeadDataset,
    buy_fill_price,
    sell_fill_price,
)


def compute_pnl(sample: dict, trade_side: str, quantity: float) -> float:
    if trade_side == "long":
        entry_fill = buy_fill_price(
            ask_prices=sample["ask_prices"].tolist(),
            ask_sizes=sample["ask_sizes"].tolist(),
            quantity=quantity,
        )
        exit_fill = sell_fill_price(
            bid_prices=sample["future_bid_prices"].tolist(),
            bid_sizes=sample["future_bid_sizes"].tolist(),
            quantity=quantity,
        )
        return float(exit_fill - entry_fill)

    entry_fill = sell_fill_price(
        bid_prices=sample["bid_prices"].tolist(),
        bid_sizes=sample["bid_sizes"].tolist(),
        quantity=quantity,
    )
    exit_fill = buy_fill_price(
        ask_prices=sample["future_ask_prices"].tolist(),
        ask_sizes=sample["future_ask_sizes"].tolist(),
        quantity=quantity,
    )
    return float(entry_fill - exit_fill)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--pattern", default="*messages*.parquet")
    parser.add_argument("--trade_side", default="long", choices=["long", "short"])
    parser.add_argument("--trade_quantity", type=float, default=1.0)
    parser.add_argument("--pnl_margin", type=float, default=0.0)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--horizon_seconds", type=int, default=30)
    args = parser.parse_args()

    message_files = sorted(Path(args.data_dir).glob(args.pattern))
    if args.max_files is not None:
        message_files = message_files[: args.max_files]
    if not message_files:
        raise FileNotFoundError(f"No files found in {args.data_dir} matching {args.pattern}")

    dataset = OnlineReturnHeadDataset(
        message_files=[str(path) for path in message_files],
        seq_len=args.seq_len,
        scenario="order_model",
        horizon_seconds=args.horizon_seconds,
        cache_size=1,
        feature_chunk_size=128,
        sample_chunk_size=128,
    )

    num_samples = min(int(args.max_samples), len(dataset))
    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.choice(len(dataset), size=num_samples, replace=False)
    pnls = np.empty(num_samples, dtype=np.float64)
    for row, idx in enumerate(sampled_indices.tolist()):
        pnls[row] = compute_pnl(dataset[idx], trade_side=args.trade_side, quantity=float(args.trade_quantity))

    keep_mask = np.abs(pnls) > float(args.pnl_margin)
    kept = pnls[keep_mask]
    positives = int(np.count_nonzero(kept > 0))
    negatives = int(np.count_nonzero(kept < 0))

    print(f"files={len(message_files)}")
    print(f"samples={num_samples}")
    print(f"pnl_margin={args.pnl_margin:g}")
    print(f"kept_samples={kept.size}")
    print(f"kept_fraction={keep_mask.mean():.6f}")
    print(f"positive={positives}")
    print(f"negative={negatives}")
    print(f"positive_rate={(positives / kept.size):.6f}" if kept.size else "positive_rate=nan")


if __name__ == "__main__":
    main()
