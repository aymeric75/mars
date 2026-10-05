"""Draw the figures, assumes that the same list of stocks is given for any given day"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, re

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from itertools import islice
from tqdm import tqdm

from market_simulation.utils.bin_converter import BinConverter

from custom_code.preprocessing.order_model.messages_to_features_no_engine import (
    build_converters_from_samples,
)


NUM_BINS_PRICE_LEVEL = 32
NUM_BINS_ORDER_VOLUME = 32
NUM_BINS_ORDER_INTERVAL = 16
NUM_BINS_LOB_VOLUME = 32
CHUNK_SIZE = 50_000


@dataclass
class Converters:
    price_level: BinConverter
    order_volume: BinConverter
    pred_order_volume: BinConverter
    order_interval: BinConverter
    lob_volume: BinConverter


CONVERTERS_JSON = Path(__file__).resolve().parents[2] / "training" / "converters_portable.json"

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


# print(converters.price_level.bin_values)

# exit()


def serie_of_mars_type(message_file):
    """returns a pandas serie holding the order type
    i.e.
    0: passive sell
    1: passive buy
    2: cancel/delete
    3: agressive sell
    4: aggressive buy
    """

    messages = pd.read_parquet(message_file)
    # start_time = 34200000226319
    # end_time = 57599998528372
    # messages = df_[(df_["Time"] >= start_time) & (df_["Time"] <= end_time)]
    # en fct du Message_type puis de la direction, tu affecte soit 0(S), 1(B), 2(C)
    messages["Mars_type"] = 0
    # TYPE 2: cancel / delete
    messages.loc[messages["Message_Type"].isin([2, 3]), "Mars_type"] = 2
    # TYPE 1 : buy passive limit order
    messages.loc[((messages["Message_Type"] == 1) & (messages["Direction"] == -1)), "Mars_type"] = 1
    # TYPE 3 : sell agressive order
    messages.loc[((messages["Message_Type"] == 4) & (messages["Direction"] == -1)), "Mars_type"] = 3
    # TYPE 4 buy aggressive
    messages.loc[((messages["Message_Type"] == 4) & (messages["Direction"] == 1)), "Mars_type"] = 4

    return messages["Mars_type"]


data_folder = Path("jsons")


def rearrange_order_type(type_, price, index_, mars_type):
    """look at the comments"""
    # if cancel / delete
    if type_ == 2:
        return type_

    # # if price is 16, go take the
    # if price == 16:
    #     return mars_type

    # else:

    # if sell order
    if type_ == 0:
        if price > 16:
            return 0  # passive limit order
        else:
            return 3  # agressive
    # if buy order
    if type_ == 1:
        if price < 16:
            return 1  # passive limit order
        else:
            return 4  # agressive
    return


def decode_order_indices(order_indices, type_divisor, price_divisor, interval_mod):
    order_type = order_indices // type_divisor
    price = (order_indices % type_divisor) // price_divisor
    volume = (order_indices % price_divisor) // interval_mod
    interval = order_indices % interval_mod
    return order_type, price, volume, interval


def rearrange_order_type_batch(order_type, price, mars_type):
    out = order_type.copy()

    price_eq_mid = price == 16
    out[price_eq_mid] = mars_type[price_eq_mid]

    active_mask = ~price_eq_mid & (order_type != 2)
    sell_mask = active_mask & (order_type == 0)
    buy_mask = active_mask & (order_type == 1)

    out[sell_mask] = np.where(price[sell_mask] > 16, 0, 3)
    out[buy_mask] = np.where(price[buy_mask] < 16, 1, 4)
    return out


def update_counter_from_array(counter, values, minlength):
    bincounts = np.bincount(values, minlength=minlength)
    for idx, count in enumerate(bincounts):
        if count:
            counter[idx] += int(count)


def plot_feature_distribution(values_dico, feature_name, ax, gt_color="orange", xtick_pos=None, xtick_labels=None):
    gt_counts = Counter()
    pred_counts = Counter()

    for stock_dico in values_dico.values():
        for feature_counts in stock_dico.values():
            gt_counts.update(feature_counts[feature_name]["ground_truth"])
            pred_counts.update(feature_counts[feature_name]["predicted"])

    classes = sorted(set(gt_counts) | set(pred_counts))

    gt = [gt_counts.get(c, 0) for c in classes]
    pred = [pred_counts.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width / 2, gt, width, label="GT", color=gt_color)
    ax.bar(x + width / 2, pred, width, label="Pred", color="blue")

    corres_names = {
        "price": "price / mid price distance (in ticks)",
        "type": "Order Type",
        "interval": "time interval between two orders (in seconds)",
        "volume": "order volume (in shares)",
    }

    ax.set_title(corres_names[feature_name])
    if xtick_pos is not None and xtick_labels is not None:
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(classes)


days = []


values_dico = {}
stock_days = {}

data_path = Path("../../../data/test")

# iterate over all stock/date present in jsons, and gather data into the "values" dict
for file_gt in data_folder.glob("*order-indices-gt.json"):
    # retrieve data stock name and date and store them
    stock = file_gt.stem.split("_")[0]
    day = file_gt.stem.split("_")[1]

    message_path = data_path / Path(f"{stock}_{day}_messages.parquet")

    mars_types = serie_of_mars_type(message_path).to_numpy()

    if stock not in stock_days:
        stock_days[stock] = []
    stock_days[stock].append(day)

    if day not in days:
        days.append(day)

    # retrieve corresponding prediction file
    file_pred = Path(data_folder / file_gt.name.replace("-gt", "-pred"))

    # load ground truth indices
    with open(file_gt, "r") as f:
        indices_gt = json.load(f)

    # load predicted indices
    with open(file_pred, "r") as f:
        indices_pred = json.load(f)
    # print(len(indices_gt))
    # print(list(indices_gt.keys())[0])
    # exit()# 1988223

    if day not in values_dico:
        values_dico[day] = {}
    if stock not in values_dico[day]:
        values_dico[day][stock] = {
            "type": {"ground_truth": Counter(), "predicted": Counter()},
            "price": {"ground_truth": Counter(), "predicted": Counter()},
            "volume": {"ground_truth": Counter(), "predicted": Counter()},
            "interval": {"ground_truth": Counter(), "predicted": Counter()},
        }

    # order_type * (self.num_bins_price_level * self.num_bins_pred_order_volume * self.num_bins_order_interval)
    # + price_slot * (self.num_bins_pred_order_volume * self.num_bins_order_interval)
    # + volume_slot * self.num_bins_order_interval
    # + interval_slot

    type_divisor = NUM_BINS_PRICE_LEVEL * NUM_BINS_ORDER_VOLUME * NUM_BINS_ORDER_INTERVAL
    price_divisor = NUM_BINS_ORDER_VOLUME * NUM_BINS_ORDER_INTERVAL
    interval_mod = NUM_BINS_ORDER_INTERVAL
    feature_counts = values_dico[day][stock]
    pred_items = iter(indices_pred.items())
    num_chunks = (len(indices_pred) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for _ in tqdm(
        range(num_chunks),
        desc=f"{stock} {day}",
        unit="chunk",
    ):
        batch = list(islice(pred_items, CHUNK_SIZE))
        if not batch:
            break

        aligned_batch = [(int(kk), indices_gt[kk], vv) for kk, vv in batch if kk in indices_gt]
        if not aligned_batch:
            continue

        idx = np.fromiter((item[0] for item in aligned_batch), dtype=np.int64, count=len(aligned_batch))
        gt_indices = np.fromiter((item[1] for item in aligned_batch), dtype=np.int64, count=len(aligned_batch))
        pred_indices = np.fromiter((item[2] for item in aligned_batch), dtype=np.int64, count=len(aligned_batch))

        gt_order_type, gt_order_price, gt_order_volume, gt_order_interval = decode_order_indices(
            gt_indices, type_divisor, price_divisor, interval_mod
        )
        pred_order_type, pred_order_price, pred_order_volume, pred_order_interval = decode_order_indices(
            pred_indices, type_divisor, price_divisor, interval_mod
        )

        mars_type = mars_types[idx]
        gt_order_type = rearrange_order_type_batch(gt_order_type, gt_order_price, mars_type)
        pred_order_type = rearrange_order_type_batch(pred_order_type, pred_order_price, mars_type)

        update_counter_from_array(feature_counts["type"]["ground_truth"], gt_order_type, minlength=5)
        update_counter_from_array(feature_counts["type"]["predicted"], pred_order_type, minlength=5)
        update_counter_from_array(feature_counts["price"]["ground_truth"], gt_order_price, minlength=NUM_BINS_PRICE_LEVEL)
        update_counter_from_array(feature_counts["price"]["predicted"], pred_order_price, minlength=NUM_BINS_PRICE_LEVEL)
        update_counter_from_array(feature_counts["volume"]["ground_truth"], gt_order_volume, minlength=NUM_BINS_ORDER_VOLUME)
        update_counter_from_array(feature_counts["volume"]["predicted"], pred_order_volume, minlength=NUM_BINS_ORDER_VOLUME)
        update_counter_from_array(feature_counts["interval"]["ground_truth"], gt_order_interval, minlength=NUM_BINS_ORDER_INTERVAL)
        update_counter_from_array(feature_counts["interval"]["predicted"], pred_order_interval, minlength=NUM_BINS_ORDER_INTERVAL)


# print(stock_days)
# exit()Value distributions of price, volume, time difference, type and side for LOBERT model


# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# for MarS
fig.suptitle("Value distributions of price, volume, time difference, type", fontsize=14)

caption = "data gathered over: " + "; ".join(f"{stock} ({', '.join(days)})" for stock, days in stock_days.items())

fig.text(0.5, -0.01, caption, ha="center", fontsize=10)

features = ["type", "price", "interval", "volume"]

converter_map = {
    "price": converters.price_level,
    "interval": converters.order_interval,
    "volume": converters.order_volume,
}

for ax, feature in zip(axes.flat, features):
    if feature in converter_map:
        bins = converter_map[feature].bins
        mid = len(bins) // 2

        bin_first = bins[0]
        bin_mid = bins[mid]
        bin_last = bins[-1]
        if feature in ["price", "volume"]:
            bin_first = int(bin_first)
            bin_mid = int(bin_mid)
            bin_last = int(bin_last)
        plot_feature_distribution(
            values_dico,
            feature,
            ax,
            xtick_pos=[0, mid, len(bins) - 1],
            xtick_labels=[bin_first, bin_mid, bin_last],
        )
    else:
        plot_feature_distribution(values_dico, feature, ax)

axes[0, 0].legend()

plt.tight_layout()
plt.savefig("all_distributions.png", bbox_inches="tight")
plt.close()
