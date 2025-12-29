#!/usr/bin/env python3
"""
messages_only_to_features.py

Replay MarS messages parquet through the built-in Exchange (no snapshots parquet),
build BinConverters on-the-fly (2-pass), and write OrderState features to a memmap
so you don't OOM.

Usage (from the MarS repo root):
  python messages_only_to_features.py \
    --messages ../data/2025-10-10_messages_10.parquet \
    --out features.int32.memmap \
    --meta features_meta.parquet \
    --symbol SIM \
    --time-unit ns \
    --sample-every 20 \
    --keep-every 50 \
    --max-events 500000
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# --- make sure imports work when you run from repo root or elsewhere ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from mlib.core.limit_order import LimitOrder
from market_simulation.conf import C
from market_simulation.states.order_state import OrderState
from market_simulation.utils.bin_converter import BinConverter




SEQ_LEN = 1024
TOKEN_DIM = 15
NUM_BINS_PRICE_LEVEL = 32
NUM_BINS_PRED_ORDER_VOLUME = 32
NUM_BINS_ORDER_INTERVAL = 16
NUM_BINS_LOB_VOLUME = 32



@dataclass
class Converters:
    price_level: BinConverter
    order_volume: BinConverter
    pred_order_volume: BinConverter
    order_interval: BinConverter
    lob_volume: BinConverter


def unit_scale_to_seconds(time_unit: str) -> float:
    # messages_df["Time"] is assumed to be an integer offset; pick the right unit
    return {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[time_unit]


def make_exchange(symbol: str, date_str: str = "2025-10-10") -> Tuple[Exchange, pd.Timestamp]:
    market_open = pd.Timestamp(f"{date_str} 09:30:00")
    market_close = pd.Timestamp(f"{date_str} 15:00:00")
    cfg = create_exchange_config_without_call_auction(
        market_open=market_open,
        market_close=market_close,
        symbols=[symbol],
    )
    return Exchange(cfg), market_open


def row_to_order(
    r,
    *,
    symbol: str,
    base_time: pd.Timestamp,
    time_unit: str,
) -> Optional[LimitOrder]:
    """
    Message_Type mapping given by user:
      1 = new limit
      2 = cancel
      3 = delete
      4 = visible execution (skip)
      5 = hidden execution (skip)
      7 = tradinghalt (skip)

    We feed only 1/2/3 to Exchange; Exchange will generate executions internally when matching.
    """
    msg = int(r.Message_Type)

    if msg not in (1, 2, 3):
        return None

    # Convert time offset to Timestamp
    t = base_time + pd.to_timedelta(int(r.Time), unit=time_unit)

    direction = int(r.Direction) if not pd.isna(r.Direction) else 0
    side = "B" if direction == 1 else "S"

    order_id = int(r.Order) if not pd.isna(r.Order) else -1
    price = int(r.Price) if not pd.isna(r.Price) else 0
    size = int(r.Size) if not pd.isna(r.Size) else 0

    if msg == 1:
        # new limit order
        return LimitOrder(
            time=t,
            type=side,
            price=price,
            volume=size,
            symbol=symbol,
            agent_id=-1,
            order_id=order_id,
            cancel_type="",
            cancel_id=-1,
            tag="replay",
        )

    # cancel/delete -> MarS uses LimitOrder(type="C") with cancel_id & cancel_type and a cancel volume
    # Your feed encodes partial/full via Size; we pass Size through.
    return LimitOrder(
        time=t,
        type="C",
        price=price,
        volume=size,
        symbol=symbol,
        agent_id=-1,
        order_id=-1,
        cancel_type=side,
        cancel_id=order_id,
        tag="replay",
    )



def pass1_collect_samples(
    messages_df: pd.DataFrame,
    *,
    symbol: str,
    time_unit: str,
    sample_every: int,
    max_events: Optional[int],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Replay messages, but only *record* samples every `sample_every` rows to keep memory/time down.

    Returns:
      - price_minus_mid samples (signed)
      - lob volume samples (top-10 volumes from snapshot)
      - interval samples in seconds
      - order size samples (from new orders)
    """

    day = pd.to_datetime(messages_df["Time"].iloc[0], unit="ns").strftime("%Y-%m-%d")


    ex, base_time = make_exchange(symbol, day)
    scale = unit_scale_to_seconds(time_unit)

    price_minus_mid: List[float] = []
    lob_vols: List[float] = []
    intervals: List[float] = []
    sizes: List[float] = []

    prev_time: Optional[int] = None

    n = len(messages_df) if max_events is None else min(len(messages_df), max_events)
    for i, r in enumerate(messages_df.itertuples(index=False)):
        if i >= n:
            break

        order = row_to_order(r, symbol=symbol, base_time=base_time, time_unit=time_unit)
        if order is None:
            continue

        # always feed exchange to keep book consistent
        try:
            ex.submit_continuous_auction_order(order)
        except AssertionError:
            # feed/book mismatch happens in real feeds; skip this event
            continue

        # sizes for volume bins (only new orders are typically used)
        if order.type in ("B", "S"):
            sizes.append(float(order.volume))

        # record samples only every sample_every rows
        if (i % sample_every) != 0:
            continue

        # interval samples
        cur_time = int(r.Time)
        if prev_time is not None:
            dt_sec = (cur_time - prev_time) * scale
            if dt_sec > 0:
                intervals.append(float(dt_sec))
        prev_time = cur_time

        # snapshot from exchange-built orderbook
        snap = ex.get_lob(symbol).snapshot(level=10)

        # ---- SAFE mid price (less strict: only require best prices) ----
        if not snap.bid_prices or not snap.ask_prices:
            continue

        best_bid = snap.bid_prices[0]
        best_ask = snap.ask_prices[0]
        if best_bid is None or best_ask is None:
            continue

        mid = (best_bid + best_ask) / 2.0

        # signed price-minus-mid sample
        if order.type in ("B", "S") and order.price:
            price_minus_mid.append(float(order.price - mid))

        # collect top-10 volumes (positive only)
        if snap.ask_volumes:
            for v in snap.ask_volumes[:10]:
                if v and v > 0:
                    lob_vols.append(float(v))
        if snap.bid_volumes:
            for v in snap.bid_volumes[:10]:
                if v and v > 0:
                    lob_vols.append(float(v))

    # ensure we have non-empty intervals (required for converter)
    if not intervals:
        intervals = [1e-6]

    # fallback if lob snapshot volumes were never available
    if not lob_vols:
        lob_vols = sizes.copy() if sizes else [1.0]

    # fallback if price_minus_mid never collected (should be rare now)
    if not price_minus_mid:
        price_minus_mid = [0.0]

    return price_minus_mid, lob_vols, intervals, sizes




def build_converters_from_samples(price_minus_mid, lob_vols, intervals, sizes):
    # --- price_level fallback ---
    pm = [float(x) for x in price_minus_mid if x is not None and np.isfinite(x)]
    if len(pm) < 1000:
        # Fallback: build symmetric deltas around 0 using observed tick size
        # Use a conservative range: +/- 50 ticks of 100 units (edit if your tick differs)
        tick = 100.0
        pm = [i * tick for i in range(-50, 51)]  # 101 values

    price_level = BinConverter.create_from_values(pm, NUM_BINS_PRICE_LEVEL)

    ov = [float(x) for x in sizes if x is not None and x > 0]
    if not ov:
        ov = [1.0]
    order_volume = BinConverter.create_from_values(ov, NUM_BINS_PRED_ORDER_VOLUME)
    pred_order_volume = order_volume

    itv = [float(x) for x in intervals if x is not None and x > 0]
    if not itv:
        itv = [1e-6]
    order_interval = BinConverter.create_from_values(itv, NUM_BINS_ORDER_INTERVAL)

    lv = [float(x) for x in lob_vols if x is not None and x > 0]
    if not lv:
        lv = ov
    lob_volume = BinConverter.create_from_values(lv, NUM_BINS_LOB_VOLUME)

    return Converters(price_level, order_volume, pred_order_volume, order_interval, lob_volume)



def make_exchange_and_orderstate(symbol: str, conv: Converters) -> Tuple[Exchange, OrderState, pd.Timestamp]:
    ex, base_time = make_exchange(symbol)

    state = OrderState(
        num_max_orders=SEQ_LEN,
        num_bins_price_level=NUM_BINS_PRICE_LEVEL,
        num_bins_pred_order_volume=NUM_BINS_PRED_ORDER_VOLUME,
        num_bins_order_interval=NUM_BINS_ORDER_INTERVAL,
        converter=conv,
    )
    ex.register_state(state)
    return ex, state, base_time


def pad_state_vector(vec: np.ndarray, feat_dim: int) -> np.ndarray:
    if vec.size >= feat_dim:
        return vec[-feat_dim:].astype(np.int32)
    out = np.zeros(feat_dim, dtype=np.int32)
    out[-vec.size:] = vec.astype(np.int32)
    return out


def pass2_write_features(
    messages_df: pd.DataFrame,
    *,
    symbol: str,
    time_unit: str,
    keep_every: int,
    out_path: str,
    meta_path: Optional[str],
    conv: Converters,
    max_events: Optional[int],
) -> Tuple[str, int]:
    """
    Replay messages again, now with OrderState registered, and write fixed-size features to memmap.
    Also optionally writes a meta parquet mapping each written row to (original Step, Time, Message_Type, Order).
    """
    ex, order_state, base_time = make_exchange_and_orderstate(symbol, conv)


    seq_len = SEQ_LEN
    token_dim = TOKEN_DIM
    feat_dim = seq_len * token_dim

    n_total = len(messages_df) if max_events is None else min(len(messages_df), max_events)
    approx_keep = (n_total + keep_every - 1) // keep_every

    mm = np.memmap(out_path, mode="w+", dtype=np.int32, shape=(approx_keep, feat_dim))

    meta_rows = [] if meta_path else None

    w = 0
    for i, r in enumerate(messages_df.itertuples(index=False)):
        if i >= n_total:
            break

        order = row_to_order(r, symbol=symbol, base_time=base_time, time_unit=time_unit)
        if order is None:
            continue

        # Always feed to keep internal state correct
        try:
            ex.submit_continuous_auction_order(order)
        except AssertionError:
            continue

        # Only write every keep_every-th *input row index*
        if (i % keep_every) != 0:
            continue



        try:
            vec = order_state.to_vector()
        except AssertionError:
            # OrderState not ready yet (cur_order is None)
            continue


        print(vec)
        print(vec.shape)
        breakpoint()


        mm[w, :] = pad_state_vector(vec, feat_dim)


        breakpoint()


        if meta_rows is not None:
            meta_rows.append(
                {
                    "written_row": w,
                    "src_index": i,
                    "Step": int(r.Step) if hasattr(r, "Step") and not pd.isna(r.Step) else None,
                    "Time": int(r.Time),
                    "Message_Type": int(r.Message_Type),
                    "Order": int(r.Order) if not pd.isna(r.Order) else None,
                }
            )
        w += 1

    mm.flush()

    if meta_rows is not None:
        pd.DataFrame(meta_rows).to_parquet(meta_path, index=False)

    return out_path, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="Path to *_messages_*.parquet (messages only)")
    ap.add_argument("--out", required=True, help="Output memmap path (int32), e.g. features.int32.memmap")
    ap.add_argument("--meta", default=None, help="Optional output meta parquet with row mapping")
    ap.add_argument("--symbol", default="SIM", help="Exchange symbol name (any string)")
    ap.add_argument("--time-unit", default="ns", choices=["ns", "us", "ms", "s"], help="Unit of messages_df['Time']")
    ap.add_argument("--sample-every", type=int, default=20, help="Pass1: collect samples every N rows")
    ap.add_argument("--keep-every", type=int, default=50, help="Pass2: write features every N rows")
    ap.add_argument("--max-events", type=int, default=None, help="Process at most this many rows (debug)")
    ap.add_argument("--date", default="2025-10-10", help="Trading date used to build timestamps (YYYY-MM-DD)")
    args = ap.parse_args()

    # Load only required columns (saves RAM)
    messages_df = pd.read_parquet(args.messages, columns=["Time", "Step", "Message_Type", "Order", "Price", "Size", "Direction"])
    # Ensure expected dtypes
    # (no heavy to_numeric conversions; we keep ints where possible)
    # If your parquet has nullable ints, it's fine.

    # Pass 1: collect samples
    print("[Pass1] collecting samples ...")
    pmid, lobv, intervals, sizes = pass1_collect_samples(
        messages_df,
        symbol=args.symbol,
        time_unit=args.time_unit,
        sample_every=max(1, args.sample_every),
        max_events=args.max_events,
    )
    print(f"[Pass1] samples: price_minus_mid={len(pmid)}, lob_vols={len(lobv)}, intervals={len(intervals)}, sizes={len(sizes)}")
    print(f"[Pass1] samples: price_minus_mid={type(pmid)}, lob_vols={type(lobv)}, intervals={type(intervals)}, sizes={type(sizes)}")

    print(pmid[:50])
    print(lobv[:50])
    print(intervals[:50])
    print(sizes[:50])


    # Build converters
    print("[Pass1] building BinConverters ...")
    conv = build_converters_from_samples(pmid, lobv, intervals, sizes)

    print(conv)
    print(conv.price_level.bins)
    print(conv.order_volume.bins)
    print(conv.pred_order_volume.bins)
    print(conv.order_interval.bins)
    print(conv.lob_volume.bins)
    #breakpoint()

    # Pass 2: replay and write features
    print("[Pass2] replaying and writing features to memmap ...")
    out_path, written = pass2_write_features(
        messages_df,
        symbol=args.symbol,
        time_unit=args.time_unit,
        keep_every=max(1, args.keep_every),
        out_path=args.out,
        meta_path=args.meta,
        conv=conv,
        max_events=args.max_events,
    )

    print(f"[Done] wrote {written} rows to {out_path}")
    if args.meta:
        print(f"[Done] meta written to {args.meta}")
    print("Tip: load memmap with:")
    print(f"  mm = np.memmap('{out_path}', mode='r', dtype=np.int32, shape=({written}, {C.order_model.seq_len*C.order_model.token_dim}))")


if __name__ == "__main__":
    main()
