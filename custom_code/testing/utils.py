import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import random

from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Tuple

from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from mlib.core.limit_order import LimitOrder
from market_simulation.conf import C
from market_simulation.states.order_state import OrderState
from market_simulation.utils.bin_converter import BinConverter




SEQ_LEN = 1 # 1024
TOKEN_DIM = 15
NUM_BINS_PRICE_LEVEL = 32
NUM_BINS_ORDER_VOLUME = 32
NUM_BINS_ORDER_INTERVAL = 16
NUM_BINS_LOB_VOLUME = 32


@dataclass
class Converters:
    price_level: BinConverter
    order_volume: BinConverter
    pred_order_volume: BinConverter
    order_interval: BinConverter
    lob_volume: BinConverter





def make_exchange(symbol: str, date_str: str = "2025-10-10") -> Tuple[Exchange, pd.Timestamp]:
    market_open = pd.Timestamp(f"{date_str} 09:30:00")
    market_close = pd.Timestamp(f"{date_str} 15:00:00")
    cfg = create_exchange_config_without_call_auction(
        market_open=market_open,
        market_close=market_close,
        symbols=[symbol],
    )
    return Exchange(cfg), market_open




def make_exchange_and_orderstate(
    symbol: str,
    date_str: str,
    conv: Converters,
) -> Tuple[Exchange, OrderState, pd.Timestamp]:
    ex, base_time = make_exchange(symbol, date_str)

    state = OrderState(
        num_max_orders=SEQ_LEN,
        num_bins_price_level=NUM_BINS_PRICE_LEVEL,
        num_bins_pred_order_volume=NUM_BINS_ORDER_VOLUME,
        num_bins_order_interval=NUM_BINS_ORDER_INTERVAL,
        converter=conv,
    )
    ex.register_state(state)
    return ex, state, base_time





def row_to_order(
    r,
    *,
    symbol: str,
    base_time: pd.Timestamp,
    time_unit: str,
    ex: Optional[Exchange] = None,   # <-- NEW: allow price lookup from current book
) -> Optional[LimitOrder]:
    """
    Message_Type mapping given by user:
      1 = new limit
      2 = cancel
      3 = delete
      4 = visible execution  (apply as cancel/reduce on the resting order_id)
      5 = hidden execution   (skip for LOB depth coherence)
      7 = tradinghalt        (skip)
      12 = ExecutionCrossTrade (apply as cancel/reduce on the resting order_id)
    """
    msg = int(r.Message_Type)

    # Convert time offset to Timestamp*

    # print("time_unit")
    # print(time_unit)
    # breakpoint()


    t = pd.to_timedelta(int(r.Time), unit=time_unit)

    # Your convention: -1 = Bid, +1 = Ask
    direction = int(r.Direction) if not pd.isna(r.Direction) else 0
    side = "B" if direction == -1 else "S"

    order_id = int(r.Order) if not pd.isna(r.Order) else -1
    price = int(r.Price) if not pd.isna(r.Price) else 0
    size = int(r.Size) if not pd.isna(r.Size) else 0

    # Skip types we cannot/should not replay into MarS' visible LOB
    if msg in (7,):
        return None

    # Hidden execution: usually doesn't change displayed depth; skip to avoid corrupting visible LOB
    if msg == 5:
        return None

    # New limit order
    if msg == 1:
        return LimitOrder(
            time=t,
            type=side,          # "B" or "S"
            price=price,
            volume=size,
            symbol=symbol,
            agent_id=-1,
            order_id=order_id,
            cancel_type="",
            cancel_id=-1,
            tag="replay",
        )

    # Cancel/Delete (2/3): MarS models both as cancel messages reducing volume on an existing order id
    if msg in (2, 3):
        # If price is missing/0, try to infer it from the current orderbook
        if (price is None or price == 0) and ex is not None and order_id >= 0:
            try:
                price = ex.get_lob(symbol).get_price_of_order_id(order_id)
            except Exception:
                return None

        return LimitOrder(
            time=t,
            type="C",
            price=price,
            volume=size,
            symbol=symbol,
            agent_id=-1,
            order_id=-1,
            cancel_type=side,      # cancel a buy if side=="B", else cancel a sell
            cancel_id=order_id,
            tag="replay",
        )

    # Visible execution / cross trade: reduce resting visible order volume
    if msg in (4,):
        # We treat it as a cancel-on-id of 'size' shares.
        # If price missing/0, infer from current orderbook.
        if (price is None or price == 0) and ex is not None and order_id >= 0:
            try:
                price = ex.get_lob(symbol).get_price_of_order_id(order_id)
            except Exception:
                # If the order is already gone (fully executed earlier), skip quietly.
                return None

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
            tag="replay_exec",
        )


    # Unknown / unsupported
    return None





