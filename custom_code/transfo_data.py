import numpy as np
import pandas as pd
import torch
import re

from market_simulation.utils.bin_converter import BinConverter
from mlib.core.limit_order import LimitOrder

from mlib.core.lob_snapshot import LobSnapshot

"""
Building the State:

In the paper:  emb(order_i) + linear_proj(LOBvolumes_i) + emb(LOBmid_price_i)


In the code [source: market_simulation/states/order_state.py]

emb(order_i) -> 4 components (type, price_slot, volume_slot, interval_slot)   (order_i means order_index)   =========   I

linear_proj(LOBvolumes_i) ->  lob_tokenizer(features[:, 5:15])                                              =========   II

emb(LOBmid_price_i)  ->   emb_chg_to_open(features[:, 3])                                                   =========   III

EXTRA TERM (not in the paper): emb_time_to_open(features[:, 4] // 5)                                        =========   IV

"""


NUM_MAX_ORDERS = 1024
DIM_ORDER = 15

NUM_BINS_PRICE_LEVEL = 32
NUM_BINS_VOLUME = 32
NUM_BINS_INTERVAL = 16


def map_order_type(row) -> int | None:
    """dsdsqdsqd"""
    # MarS uses 3 types: Ask, Bid, Cancel (paper) :contentReference[oaicite:1]{index=1}
    # You must decide how your CSV encodes cancel vs new/modify.

    # submission of a new limit order
    if int(row["Message_Type"]) == 1:
        # Buy Limit order
        if int(row["Direction"]) == 1:
            return "B"
        elif int(row["Direction"]) == -1:
            return "S"
        else:
            print("problem ")
            # breakpoint()
            return None

    # total deletion of a limit order
    if int(row["Message_Type"]) == 3:
        return "C"

    ####breakpoint()
    return None



def _levels(row, prefix):
    # collect (level, value) pairs like (1, 3642300.0)
    pairs = []
    for k, v in row.items():
        m = re.fullmatch(rf"{prefix}(\d+)", k)
        if m:
            lvl = int(m.group(1))
            pairs.append((lvl, v))
    # sort by level, keep only non-nan
    pairs.sort(key=lambda x: x[0])
    return [v for _, v in pairs if v == v]  # v==v filters out NaN

def bin_index(x, edges):
    """zadd"""
    # edges: ascending list of bin edges, returns 0..len(edges)-2
    return int(np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2))


def add_mid_and_price_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- best bid/ask (level 1) ---
    best_ask = pd.to_numeric(df["Ask_Price_1"], errors="coerce")
    best_bid = pd.to_numeric(df["Bid_Price_1"], errors="coerce")

    # mid price = (best ask + best bid) / 2  (only where both exist)
    df["mid_price"] = (best_ask + best_bid) / 2.0

    # order price (may be NaN for some message types)
    order_price = pd.to_numeric(df["Price"], errors="coerce")

    # price delta to mid (this is what MarS bins into price_slot)
    df["price_minus_mid"] = order_price - df["mid_price"]

    # optional: in "ticks" if you know your tick size (example tick=100 in your units)
    TICK_SIZE = 100.0
    df["price_minus_mid_ticks"] = df["price_minus_mid"] / TICK_SIZE

    return df

# # Example bins (you probably want to fit these from your data)
# VOL_EDGES = np.geomspace(1, 1e7, NUM_BINS_VOLUME + 1)  # size → 32 bins
# INT_EDGES = np.geomspace(1e-6, 1.0, NUM_BINS_INTERVAL + 1)  # seconds → 16 bins


def make_features_from_csv(df: pd.DataFrame) -> torch.Tensor:
    """loop over a df of limit orders and construct for each row a feature df for the Order Model"""


    print(df)
    breakpoint()

    # Time assumed nanoseconds since epoch (your sample looks like that)
    t = df["Time"].astype(np.int64).to_numpy()
    t0 = t[0]
    time_to_open_sec = ((t - t0) / 1e9).astype(np.int32)

    volume_bin_converter = BinConverter.create_from_values(df["Size"].values, NUM_BINS_VOLUME)
    VOL_EDGES = volume_bin_converter.bins

    # interval seconds between events (first one = 0)
    dt = np.diff(t, prepend=t[0]) / 1e9
    dt = np.maximum(dt, 0.0)
    interval_bin_converter = BinConverter.create_from_values(dt, NUM_BINS_INTERVAL)
    INT_EDGES = interval_bin_converter.bins

    # price minus mid bins

    price_minus_mid_bin_converter = BinConverter.create_from_values(df["price_minus_mid"].dropna().values, NUM_BINS_PRICE_LEVEL)
    PRICE_EDGES = price_minus_mid_bin_converter.bins

    rows = []
    prev_t = t[0]
    for i, row in df.iterrows():

        order_type = map_order_type(row)

        if order_type is None:
            continue

        if pd.isna(row["mid_price"]):
            continue


        print(row)
        time_ = pd.to_datetime(row["Time"], unit="ns")
        print(time_)


        limit_order = LimitOrder(
            time = pd.to_datetime(row["Time"], unit="ns"),
            type = order_type,
            price = int(row["Price"]),
            volume = int(row["Size"]),
            symbol = "APPL",
            agent_id =  99999,
            order_id = int(row["Order"]),
            cancel_type = "None" if order_type in ["B", "S"] else order_type,
            cancel_id = -1,
            tag = ""
        )
        print(_levels(row, "Ask_Price_"))

        # integer position of current row
        pos = df.index.get_loc(i)
        if pos == 0:
            prev_mid = 0
        else:
            prev_i = df.index[pos - 1]
            prev_row = df.loc[prev_i]
            prev_mid = prev_row["mid_price"]

        lobsnapshot = LobSnapshot(
            time = pd.to_datetime(row["Time"], unit="ns"),
            max_level = 10,
            last_price = prev_mid,
            ask_prices = _levels(row, "Ask_Price_"),
            ask_volumes = _levels(row, "Ask_Size_"),
            bid_prices = _levels(row, "Bid_Price_"),
            bid_volumes = _levels(row, "Bid_Size_")
        )

        print("lobsnapshot")
        print(lobsnapshot)

        breakpoint()


        mid_price = float(row["mid_price"]) if pd.notna(row["mid_price"]) else 0.0
        price_slot = bin_index(mid_price, PRICE_EDGES)

        # Volume slot: from Size (define your own bins or load MarS converters)
        size = float(row["Size"]) if pd.notna(row["Size"]) else 0.0
        volume_slot = bin_index(max(size, 1.0), VOL_EDGES)

        # Interval slot: from dt bins
        interval_slot = bin_index(max(dt[i], 1e-9), INT_EDGES)


        # Compose order_index exactly as MarS does
        order_index = (
            order_type * (NUM_BINS_PRICE_LEVEL * NUM_BINS_VOLUME * NUM_BINS_INTERVAL)
            + price_slot * (NUM_BINS_VOLUME * NUM_BINS_INTERVAL)
            + volume_slot * NUM_BINS_INTERVAL
            + interval_slot
        )

        print("order_index")
        print(order_index)

        breakpoint()



        # price_change_to_open needs open mid/price history -> placeholder
        price_change_to_open = 0

        # lob volumes need LOB snapshot -> placeholder 10 zeros
        lob_volumes = [0] * 10

        token15 = [
            int(order_index),
            int(volume_ratio_slot),
            int(trans_ratio_slot),
            int(price_change_to_open),
            int(time_to_open_sec[i]),
            *lob_volumes,
        ]
        rows.append(token15)

        if len(rows) >= NUM_MAX_ORDERS:
            break

    # left-pad or right-pad to NUM_MAX_ORDERS (MarS keeps a fixed-length window)
    while len(rows) < NUM_MAX_ORDERS:
        rows.append([0] * DIM_ORDER)

    arr = np.array(rows, dtype=np.int32)  # (1024, 15)
    flat = arr.reshape(1, NUM_MAX_ORDERS * DIM_ORDER)  # (1, 1024*15)
    return torch.from_numpy(flat).long()


messages_df = pd.read_parquet("../data/2025-10-10_messages_10.parquet")

lob_df = pd.read_parquet("../data/2025-10-10_snapshots_10.parquet")

merged = messages_df.join(lob_df.drop(columns=["Time","Step"], errors="ignore"), how="left")

merged = add_mid_and_price_delta(merged)

features = make_features_from_csv(merged)

print(features)
# # logits = model(features)


# inspect one by one
