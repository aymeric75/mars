import pandas as pd
import numpy as np
import json
import pyarrow.parquet as pq
from dataclasses import dataclass
from market_simulation.utils.bin_converter import BinConverter
from custom_code.preprocessing.order_model.messages_to_features import (
    build_converters_from_samples,
)


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

with open("converters_portable.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

price_minus_mid = []
for bin_item in obj["state"]["price_level"]["bin_values"]:
    price_minus_mid.extend(bin_item["data"])

sizes = []
for bin_item in obj["state"]["order_volume"]["bin_values"]:
    sizes.extend(bin_item["data"])

intervals = []
for bin_item in obj["state"]["order_interval"]["bin_values"]:
    intervals.extend(bin_item["data"])

lob_vols = []
for bin_item in obj["state"]["lob_volume"]["bin_values"]:
    lob_vols.extend(bin_item["data"])

converters = build_converters_from_samples(price_minus_mid, sizes, intervals, lob_vols)

#### prends le messages parquet ET LE SNAPSHOT

# construis:

#             Step            Time     f0  f1  f2   f3     f4  f5  f6  f7  f8  f9  f10  f11  f12  f13  f14
# 0          19935  34200000226319  10351   9   0  135  19799   0   1   0   1   0    0    0    0    1    0
# 1          19936  34200000627703  41132   4   9  135  19799   0   1   0   1   0    0    0    0    1    0
# 2          19937  34200002951713  32846   9   9  135  19799   0   1   0   1   0    0    0    0    1    0
# 3          19938  34200002958773  34887   9   9  135  19799   0   1   0   1   0    0    0    0    1    0




#  0(S), 1(B), 2(C)
def create_mars_order_type_column(messages):
    message_type = messages["Message_Type"]
    direction = messages["Direction"]
    messages["Mars_type"] = np.select(
        [
            message_type.isin([2, 3]),
            ((message_type == 1) & (direction == -1)) | ((message_type == 4) & (direction == 1)),
        ],
        [2, 1],
        default=0,
    )
def create_slots_columns(df):
    # cur_order.price - mid_price
    price_minus_mid = (df["Price"] - df["mid_price"]).to_numpy()
    df["bin_price"] = converters.price_level.get_bin_indices(price_minus_mid)

    size_values = df["Size"].to_numpy()
    df["bin_vol"] = converters.order_volume.get_bin_indices(size_values)

    df["seconds_since_prev"] = df["Time"].diff() / 1e9
    interval_values = df["seconds_since_prev"].to_numpy()
    df["bin_interval"] = converters.order_interval.get_bin_indices(interval_values)
    #df = df.drop(columns=["seconds_since_prev"])


def bins_lob_volumes(df):
    cols = [f"Ask_Size_{i}" for i in range(1, 6)] + [f"Bid_Size_{i}" for i in range(1, 6)]
    lob_values = df[cols].to_numpy()
    df[cols] = converters.lob_volume.get_bin_indices(lob_values)


def create_mid_price_column(df, snapshots):
    df["mid_price"] = (snapshots["Ask_Price_1"] + snapshots["Bid_Price_1"]) / 2


def create_price_change_to_open(df):
    df["price_change_to_open"] = (df["mid_price"] / df["Price"] - 1).fillna(0)
    df["price_change_to_open"] = np.clip(df["price_change_to_open"], -0.2, 0.2)


def create_seconds_since_open(df):
    open_time_ns = (9*3600 + 30*60) * 1_000_000_000
    df["seconds_since_open"] = (df["Time"] - open_time_ns) / 1e9


def add_lob_volumes(df, snapshots):
    cols = [f"Ask_Size_{i}" for i in range(1, 6)] + [f"Bid_Size_{i}" for i in range(1, 6)]
    df[cols] = snapshots[cols]

def _read_parquet_row_slice(parquet_path, columns, start_row=None, num_rows=None, batch_size=65536):
    if start_row is None:
        return pd.read_parquet(parquet_path, columns=columns)

    if num_rows is None or num_rows <= 0:
        return pd.DataFrame(columns=columns)

    stop_row = start_row + num_rows
    parquet_file = pq.ParquetFile(parquet_path)
    batches = []
    seen_rows = 0

    for batch in parquet_file.iter_batches(columns=columns, batch_size=batch_size):
        batch_len = len(batch)
        batch_start = seen_rows
        batch_stop = seen_rows + batch_len

        if batch_stop <= start_row:
            seen_rows = batch_stop
            continue

        if batch_start >= stop_row:
            break

        take_start = max(start_row, batch_start) - batch_start
        take_stop = min(stop_row, batch_stop) - batch_start
        batches.append(batch.slice(take_start, take_stop - take_start))
        seen_rows = batch_stop

    if not batches:
        return pd.DataFrame(columns=columns)

    return pd.concat([batch.to_pandas() for batch in batches], ignore_index=True)


def from_messages_to_features(message_file, snapshot_file, start_row=None, num_rows=None):


    msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
    snap_cols = [
        "Ask_Price_1", "Bid_Price_1",
        "Ask_Size_1", "Ask_Size_2", "Ask_Size_3", "Ask_Size_4", "Ask_Size_5",
        "Bid_Size_1", "Bid_Size_2", "Bid_Size_3", "Bid_Size_4", "Bid_Size_5",
    ]

    messages = _read_parquet_row_slice(message_file, columns=msg_cols, start_row=start_row, num_rows=num_rows)
    snapshots = _read_parquet_row_slice(snapshot_file, columns=snap_cols, start_row=start_row, num_rows=num_rows)

    start = (9*60*60 + 30*60) * 1_000_000_000   # 9:30 in ns
    end = (16*60*60) * 1_000_000_000            # 16:00 in ns
    time_mask = (messages["Time"] >= start) & (messages["Time"] <= end)

    messages = messages.loc[time_mask].copy()
    snapshots = snapshots.loc[time_mask].copy()

    create_mars_order_type_column(messages)
    features = messages[["Time", "Mars_type", "Price", "Size"]].copy()
    create_mid_price_column(features, snapshots)

    create_slots_columns(features)

    features["f0"] = (
        features["Mars_type"] * NUM_BINS_PRICE_LEVEL * NUM_BINS_ORDER_VOLUME * NUM_BINS_ORDER_INTERVAL
        + features["bin_price"] * NUM_BINS_ORDER_VOLUME * NUM_BINS_ORDER_INTERVAL
        + features["bin_vol"] * NUM_BINS_ORDER_INTERVAL
        + features["bin_interval"])


    features = features.drop(columns=["bin_price", "bin_vol", "seconds_since_prev", "bin_interval"])
    features["vol_ratio_slot"] = 0
    features["trans_ratio_slot"] = 0

    create_price_change_to_open(features)
    features = features.drop(columns=["mid_price"])
    create_seconds_since_open(features)

    add_lob_volumes(features, snapshots)

    features = features.drop(columns=["Time", "Mars_type", "Price", "Size"])
    bins_lob_volumes(features)

    features = features.fillna(0)

    #print(features)

    return features

# from_messages_to_features("../data/LOBSTER_AAPL_2025-10-29_messages_10.parquet", "../data/LOBSTER_AAPL_2025-10-29_snapshots_10.parquet")




# NUM_BINS_PRICE_LEVEL = 32
# NUM_BINS_ORDER_VOLUME = 32
# NUM_BINS_ORDER_INTERVAL = 16
# NUM_BINS_LOB_VOLUME = 32


# f0
# order_type * (self.num_bins_price_level * self.num_bins_pred_order_volume * self.num_bins_order_interval)
# + price_slot * (self.num_bins_pred_order_volume * self.num_bins_order_interval)
# + volume_slot * self.num_bins_order_interval
# + interval_slot


# une fois que tu as f0..


#
