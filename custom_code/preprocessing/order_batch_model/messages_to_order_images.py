import pandas as pd
import numpy as np
import json
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
    messages["Mars_type"] = 0
    # TYPE 2: cancel / delete
    messages.loc[messages["Message_Type"].isin([2, 3]), "Mars_type"] = 2

    # BUY PASSIVE ORDER
    messages.loc[
        ((messages["Message_Type"] == 1) & (messages["Direction"] == -1)),
        "Mars_type"
    ] = 1

    # SELL PASSIVE ORDER
    messages.loc[
        ((messages["Message_Type"] == 1) & (messages["Direction"] == 1)),
        "Mars_type"
    ] = 0
    # BUY AGRESSIVE ORDER
    messages.loc[
        ((messages["Message_Type"] == 4) & (messages["Direction"] == 1)),
        "Mars_type"
    ] = 1

    # SELL AGRESSIVE ORDER
    messages.loc[
        ((messages["Message_Type"] == 4) & (messages["Direction"] == -1)),
        "Mars_type"
    ] = 0


def create_slots_columns(df):
    """ """
    df["bin_price"] = (df["Price"] - df["mid_price"]).apply(converters.price_level.get_bin_index)
    df["bin_vol"] = df["Size"].apply(converters.order_volume.get_bin_index)


def bins_lob_volumes(df):
    cols = [f"Ask_Size_{i}" for i in range(1, 6)] + [f"Bid_Size_{i}" for i in range(1, 6)]
    for c in cols:
        df[c] = df[c].apply(converters.lob_volume.get_bin_index)


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

def from_messages_to_features(message_file, snapshot_file):
    """
    Input: a message parquet file and corresponding snapshot file

    Output: a features dataframe holding for each order:
        Time (in ns), MarS type (0: sell, 1: buy, 2: sell)
        bin price (0-31)
        bin vol (0-31)
    """

    msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
    snap_cols = [
        "Ask_Price_1", "Bid_Price_1",
        "Ask_Size_1", "Ask_Size_2", "Ask_Size_3", "Ask_Size_4", "Ask_Size_5",
        "Bid_Size_1", "Bid_Size_2", "Bid_Size_3", "Bid_Size_4", "Bid_Size_5",
    ]

    messages = pd.read_parquet(message_file, columns=msg_cols)
    snapshots = pd.read_parquet(snapshot_file, columns=snap_cols)

    #  0(S), 1(B), 2(C)
    create_mars_order_type_column(messages)
    features = messages[["Time", "Mars_type", "Price", "Size"]].copy()

    create_mid_price_column(features, snapshots)

    start = (9*60*60 + 30*60) * 1_000_000_000   # 9:30 in ns
    end = (16*60*60) * 1_000_000_000            # 16:00 in ns
    features = features[(features["Time"] >= start) & (features["Time"] <= end)]

    create_slots_columns(features)
    features = features.drop(columns=["Price", "Size", "mid_price"])

    features = features.fillna(0)

    features[["Mars_type", "bin_price", "bin_vol"]] = features[["Mars_type", "bin_price", "bin_vol"]].astype("int32")
    features["Time"] = features["Time"].astype("int64")

    return features


#print(from_messages_to_features("../data/LOBSTER_AAPL_2025-10-29_messages_10.parquet", "../data/LOBSTER_AAPL_2025-10-29_snapshots_10.parquet"))



def retrieve_chunk_last_16min(message_file, index):
    """
    test if present (index) - 16min exists
    """

    msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
    messages = pd.read_parquet(message_file, columns=msg_cols)

    t = messages['Time'].to_numpy()
    m = 60_000_000_000
    b = np.searchsorted(t, t[index] - np.arange(16, -1, -1)*m)

    enough_history = b[0] > 0

    if not enough_history:
        raise Exception("Not enough historical data available (16 minutes required)")

    chunks = [messages.iloc[b[k]:b[k+1]] for k in range(16)]

    return chunks


chunks = retrieve_chunk_last_16min("../data/LOBSTER_AAPL_2025-10-29_messages_10.parquet", 15000)


print(chunks)
