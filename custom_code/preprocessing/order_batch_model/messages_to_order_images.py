import pandas as pd
import numpy as np
import json

from pathlib import Path
from dataclasses import dataclass
from market_simulation.utils.bin_converter import BinConverter
from custom_code.preprocessing.order_model.messages_to_features import (
    build_converters_from_samples,
)


NUM_BINS_PRICE_LEVEL = 32
NUM_BINS_ORDER_VOLUME = 32
NUM_BINS_ORDER_INTERVAL = 16
NUM_BINS_LOB_VOLUME = 32

ORDER_IMAGE_MAX_VALUE = 100
ONE_SECOND_NS = 1_000_000_000
ONE_MINUTE_NS = 60 * ONE_SECOND_NS

# Market hours (nanoseconds since midnight)
MARKET_OPEN_NS = 9 * 60 * 60 * 1_000_000_000 + 30 * 60 * 1_000_000_000  # 09:30
MARKET_CLOSE_NS = 16 * 60 * 60 * 1_000_000_000                          # 16:00

@dataclass
class Converters:
    price_level: BinConverter
    order_volume: BinConverter
    pred_order_volume: BinConverter
    order_interval: BinConverter
    lob_volume: BinConverter

with open("/projappl/project_2012747/mars/MarS/custom_code/preprocessing/order_batch_model/converters_portable.json", "r", encoding="utf-8") as f:
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






def create_mars_order_type_column(messages: pd.DataFrame) -> None:
    messages["Mars_type"] = 0
    messages.loc[messages["Message_Type"].isin([2, 3]), "Mars_type"] = 2
    messages.loc[
        (messages["Message_Type"] == 1) & (messages["Direction"] == -1),
        "Mars_type",
    ] = 1
    messages.loc[
        (messages["Message_Type"] == 1) & (messages["Direction"] == 1),
        "Mars_type",
    ] = 0
    messages.loc[
        (messages["Message_Type"] == 4) & (messages["Direction"] == 1),
        "Mars_type",
    ] = 1
    messages.loc[
        (messages["Message_Type"] == 4) & (messages["Direction"] == -1),
        "Mars_type",
    ] = 0


def create_mid_price_column(df: pd.DataFrame, snapshots: pd.DataFrame) -> None:
    df["mid_price"] = (snapshots["Ask_Price_1"] + snapshots["Bid_Price_1"]) / 2


def create_slots_columns(df: pd.DataFrame) -> None:
    df["bin_price"] = (df["Price"] - df["mid_price"]).apply(converters.price_level.get_bin_index)
    df["bin_vol"] = df["Size"].apply(converters.order_volume.get_bin_index)


def from_messages_and_snapshots_to_features(
    messages: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    messages = messages.copy()
    create_mars_order_type_column(messages)
    features = messages[["Time", "Mars_type", "Price", "Size"]].copy()
    create_mid_price_column(features, snapshots)
    features = features[(features["Time"] >= MARKET_OPEN_NS) & (features["Time"] <= MARKET_CLOSE_NS)]
    create_slots_columns(features)
    features = features.drop(columns=["Price", "Size", "mid_price"])
    features = features.fillna(0)
    features[["Mars_type", "bin_price", "bin_vol"]] = features[
        ["Mars_type", "bin_price", "bin_vol"]
    ].astype("int32")
    features["Time"] = features["Time"].astype("int64")
    return features.reset_index(drop=True)


def from_messages_to_features(message_file: str | Path, snapshot_file: str | Path) -> pd.DataFrame:
    msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
    snap_cols = ["Ask_Price_1", "Bid_Price_1"]
    messages = pd.read_parquet(message_file, columns=msg_cols)
    snapshots = pd.read_parquet(snapshot_file, columns=snap_cols)
    return from_messages_and_snapshots_to_features(messages, snapshots)


def retrieve_chunk_last_16min_from_df(features: pd.DataFrame, index: int) -> list[pd.DataFrame]:
    """
    Assumes `index` has already been validated upstream.
    Returns the last 16 one-minute chunks ending at features.iloc[index].
    """
    times = features["Time"].to_numpy(dtype=np.int64, copy=False)
    cut_points = np.searchsorted(
        times,
        times[index] - np.arange(16, -1, -1, dtype=np.int64) * ONE_MINUTE_NS,
        side="left",
    )
    return [features.iloc[cut_points[k]:cut_points[k + 1]] for k in range(16)]


def compute_valid_anchor_indices(times: np.ndarray) -> np.ndarray:
    """
    Return only the anchor indices for which the 16 one-minute chunks are valid.
    Uses the exact same boundary logic as retrieve_chunk_last_16min_from_df.
    """
    times = np.asarray(times, dtype=np.int64)
    if times.size == 0:
        return np.empty(0, dtype=np.int64)

    valid = []
    offsets = np.arange(16, -1, -1, dtype=np.int64) * ONE_MINUTE_NS

    for i in range(times.size):
        cut_points = np.searchsorted(times, times[i] - offsets, side="left")

        # Need the earliest boundary to be strictly inside the file history.
        # If cut_points[0] == 0, retrieval would rely on data before the file start.
        if cut_points[0] > 0:
            valid.append(i)

    return np.asarray(valid, dtype=np.int64)

def retrieve_chunk_last_16min(message_file: str | Path, index: int) -> list[pd.DataFrame]:
    msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
    messages = pd.read_parquet(message_file, columns=msg_cols)
    times = messages["Time"].to_numpy(dtype=np.int64, copy=False)
    cut_points = np.searchsorted(times, times[index] - np.arange(16, -1, -1) * ONE_MINUTE_NS)
    enough_history = cut_points[0] > 0
    if not enough_history:
        raise ValueError("Not enough historical data available (16 minutes required)")
    return [messages.iloc[cut_points[k] : cut_points[k + 1]] for k in range(16)]


def chunk_to_order_image(chunk: pd.DataFrame) -> np.ndarray:
    image = np.zeros((3, NUM_BINS_PRICE_LEVEL, NUM_BINS_ORDER_VOLUME), dtype=np.uint8)
    if chunk.empty:
        return image

    mars_type = np.clip(chunk["Mars_type"].to_numpy(dtype=np.int64, copy=False), 0, 2)
    bin_price = np.clip(
        chunk["bin_price"].to_numpy(dtype=np.int64, copy=False),
        0,
        NUM_BINS_PRICE_LEVEL - 1,
    )
    bin_vol = np.clip(
        chunk["bin_vol"].to_numpy(dtype=np.int64, copy=False),
        0,
        NUM_BINS_ORDER_VOLUME - 1,
    )

    np.add.at(image, (mars_type, bin_price, bin_vol), 1)
    np.clip(image, 0, ORDER_IMAGE_MAX_VALUE, out=image)
    return image


def chunks_to_order_images(chunks: list[pd.DataFrame]) -> np.ndarray:
    images = [chunk_to_order_image(chunk) for chunk in chunks]
    return np.stack(images, axis=0)


def chunks_to_order_images(chunks: list[pd.DataFrame]) -> np.ndarray:
    images = [chunk_to_order_image(chunk) for chunk in chunks]
    return np.stack(images, axis=0)
