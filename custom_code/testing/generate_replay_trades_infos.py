import sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from typing import Optional, Tuple, List
from pathlib import Path

from mlib.core.trade_info import TradeInfo
from mlib.core.lob_snapshot import LobSnapshot
from market_simulation.utils import pkl_utils

from utils import Converters, make_exchange_and_orderstate, row_to_order

from report_stylized_facts import get_minute_info

def build_replay_trade_infos(
    messages_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    symbol: str,
    day: str,
    conv: Converters,
    time_unit: str = "ns",
    max_events: Optional[int] = None,
    snapshot_level: int = 10,
) -> Tuple[List[TradeInfo], LobSnapshot]:
    """
    Replays historical messages through the MarS exchange to produce replay TradeInfo list
    + the start LOB snapshot (at MarketHours start, sys_code==22).

    Mirrors the MarketHours gating logic of pass2_write_features.
    """
    
    ex, order_state, _base_time = make_exchange_and_orderstate(symbol, day, conv)
    
    # Align meta System_Event_Code to each message row (same as pass2_write_features)
    meta = meta_df.sort_values("Time", kind="mergesort")
    mt = meta["Time"].to_numpy()
    mc = meta["System_Event_Code"].to_numpy()
    msg_t = messages_df["Time"].to_numpy()
    j = np.searchsorted(mt, msg_t)
    j = np.clip(j, 1, len(mt) - 1)
    left = j - 1
    right = j
    
    nearest = np.where((msg_t - mt[left]) <= (mt[right] - msg_t), left, right)
    msg_sys_code = mc[nearest]

    n = len(messages_df) if max_events is None else min(len(messages_df), max_events)
    
    counter = 0
    
    markethours = False
    start_lob: Optional[LobSnapshot] = None
    replay_trade_infos: List[TradeInfo] = []

    for i, r in enumerate(
        tqdm(messages_df.itertuples(index=False), total=n, desc="replay", unit="msg")
    ):
        


        if i == 0:
            order_state.open_time = r.Time

        order = row_to_order(r, symbol=symbol, base_time=None, time_unit=time_unit, ex=ex)
        
        if order is None:
            continue

        if order_state.open_trans_price is None and getattr(r, "Price", None) is not None:
            order_state.open_trans_price = r.Price

        trade_infos = ex.submit_continuous_auction_order(order)

        sys_code = int(msg_sys_code[i])

        if sys_code == 22 and not markethours:
            markethours = True
            start_lob = ex.get_lob(symbol).snapshot(level=snapshot_level)
        
        if sys_code == 23:
            break

        if not markethours:
            continue


        if not trade_infos:
            continue
        else:

            for trade_info in trade_infos:
                #if trade_info.transactions:
                replay_trade_infos.append(trade_info)

        
        
        if counter > n:
            break

        counter+=1

    return replay_trade_infos, start_lob




with open("/scratch/project_2012747/mars_data/order_model/train/intermediate/converters.pkl", "rb") as f:
    converters = pickle.load(f)
    
messages_df = pd.read_parquet("/scratch/project_2012747/mars_data/order_model/val/raw/AAPL_2025-11-28_messages.parquet")
meta_df = pd.read_parquet("/scratch/project_2012747/mars_data/order_model/val/raw/AAPL_2025-11-28_meta.parquet")

replay_trade_infos, start_lob = build_replay_trade_infos(
    messages_df,
    meta_df,
    symbol="AAPL",
    day="2025-11-28",
    conv=converters,
    max_events=None
)


minutes = get_minute_info(replay_trade_infos, start_lob)

pkl_utils.save_pkl_zstd(
    minutes,
    Path("my_minutes_replay.zstd")
)


print("start_lob")
print(start_lob)



