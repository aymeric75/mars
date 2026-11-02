import pandas as pd
import numpy as np
import json
import time

from dataclasses import dataclass
from pathlib import Path
from market_simulation.utils.bin_converter import BinConverter
from custom_code.preprocessing.order_model.messages_to_features_no_engine import (
    build_converters_from_samples,
)

NUM_BINS_PRICE_LEVEL = 32
NUM_BINS_ORDER_VOLUME = 32
NUM_BINS_ORDER_INTERVAL = 16
NUM_BINS_LOB_VOLUME = 32


# ------------------------
#   Load Converters
# ------------------------


@dataclass
class Converters:
    price_level: BinConverter
    order_volume: BinConverter
    pred_order_volume: BinConverter
    order_interval: BinConverter
    lob_volume: BinConverter


CONVERTERS_JSON = Path(__file__).resolve().parent / "training" / "converters_portable.json"

with CONVERTERS_JSON.open("r", encoding="utf-8") as f:
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


# vol, price, i , timestamp,
def return_list_past_indices_windows(message_df: pd.DataFrame, index):
    """from a given dataframe return a list of index ranges representing 16 past windows
    first element is the oldest
    """
    last_16_index = int(message_df["index_pastmin16"][index])
    step_size = (index - last_16_index) // 16

    indices = []
    first_index = last_16_index
    for i in range(16):
        last_index = first_index + step_size
        if i == 15:
            last_index = index
        window = [first_index, last_index]
        indices.append(window)
        first_index = last_index
    return indices


# ----------------------------
#   Adapt messages.parquet
# ----------------------------


df = pd.read_parquet("testing/data/AAPL_2025-10-29_messages.parquet")
minimal_df = df[["Time", "Message_Type", "Price", "Size", "Direction"]].copy()
minimal_df["price_level"] = minimal_df["Price"].apply(converters.price_level.get_bin_index)
minimal_df["order_volume"] = minimal_df["Size"].apply(converters.order_volume.get_bin_index)
conditions = [
    ((df["Message_Type"] == 1) & (df["Direction"] == 1)) | ((df["Message_Type"] == 4) & (df["Direction"] == -1)),
    ((df["Message_Type"] == 1) & (df["Direction"] == -1)) | ((df["Message_Type"] == 4) & (df["Direction"] == 1)),
    (df["Message_Type"].isin([2, 3])),
]
choices = [0, 1, 2]
minimal_df["matrix_index"] = np.select(conditions, choices)


# -----------------------------------------------------------
#   Create a "seconds" column and a last16min_index columns
# -----------------------------------------------------------
minimal_df["seconds_since_midnight"] = minimal_df["Time"] // 1_000_000_000
secs = minimal_df["seconds_since_midnight"].values
indices_pastmin16 = np.empty(len(secs), dtype=np.int64)
j = 0
for i in range(len(secs)):
    while secs[j] < secs[i] - 960:
        j += 1
    indices_pastmin16[i] = j

minimal_df["index_pastmin16"] = indices_pastmin16


# -----------------------------------------------------------
#   Drop unecessary columns (for creating the images)
# -----------------------------------------------------------
minimal_df.drop(columns=["Time", "Price", "Size", "Direction", "seconds_since_midnight"], inplace=True)


# -----------------------------------------------------------
#   create an image from minimal_df and an index
# -----------------------------------------------------------


start__ = time.perf_counter()

list_ranges = return_list_past_indices_windows(minimal_df, 5722753)

# on va créer 16 matrices,
for start, end in list_ranges:
    sub_df = minimal_df[(minimal_df.index >= start) & (minimal_df.index < end)]

    for mat_index in range(3):
        sub_channel_df = sub_df[sub_df["matrix_index"] == mat_index]

        pl = sub_channel_df["price_level"].to_numpy()
        ov = sub_channel_df["order_volume"].to_numpy()

        channel_matrix = np.bincount(pl * 32 + ov, minlength=32 * 32).reshape(32, 32)

        # print("channel_matrix")
        # print(channel_matrix)
        # print("channel_matrix")

        # exit()


end = time.perf_counter()
elapsed = end - start__
print(f"Elapsed time: {elapsed} seconds")
