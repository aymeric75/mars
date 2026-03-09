""" Draw the figures, assumes that the same list of stocks is given for any given day """

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob, re

from dataclasses import dataclass
from pathlib import Path
from collections import Counter

from market_simulation.states.order_state import OrderState, PredOrderInfo
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

print(converters.price_level.bins)
print(converters.order_volume.bins)
print(converters.pred_order_volume.bins)
print(converters.order_interval.bins)

#exit()

data_folder = Path("jsons")





def rearrange_order_type(type_, price):
    """ look at the comments """
    # if cancel / delete
    if type_ == 2:
        return type_
    # if sell order
    if type_ == 0:
        if price > 16:
            return 0 # passive limit order
        else:
            return 3 # agressive
    # if buy order
    if type_ == 1:
        if price < 16:
            return 1 # passive limit order
        else:
            return 4 # agressive
    return



def plot_feature_distribution(values_dico, feature_name, ax, gt_color="orange", xtick_pos=None, xtick_labels=None):
    gt_values = []
    pred_values = []

    #
    for day, stock_dico in values_dico.items():
        for keys, spec_values in stock_dico.items():
            gt_values_tmp = [v[feature_name]["ground_truth"] for v in spec_values.values()]
            gt_values.extend(gt_values_tmp)
            pred_values_tmp = [v[feature_name]["predicted"] for v in spec_values.values()]
            pred_values.extend(pred_values_tmp)

    gt_counts = Counter(gt_values)
    pred_counts = Counter(pred_values)

    classes = sorted(set(gt_values) | set(pred_values))

    gt = [gt_counts.get(c, 0) for c in classes]
    pred = [pred_counts.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width/2, gt, width, label="GT", color=gt_color)
    ax.bar(x + width/2, pred, width, label="Pred")

    
    corres_names = {
        
        "price": "price / mid price distance (in ticks)",
        "type": "Order Type",
        "interval": "time interval between two orders (in seconds)",
        "volume": "order volume (in shares)"
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

# iterate over all stock/date present in jsons, and gather data into the "values" dict
for file_gt in data_folder.glob("*order-indices-gt.json"):

    # retrieve data stock name and date and store them
    stock = file_gt.stem.split("_")[0]
    day = file_gt.stem.split("_")[1]

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


    if day not in values_dico:
        values_dico[day] = {}
    if stock not in values_dico[day]:
        values_dico[day][stock] = {}



    # order_type * (self.num_bins_price_level * self.num_bins_pred_order_volume * self.num_bins_order_interval)
    # + price_slot * (self.num_bins_pred_order_volume * self.num_bins_order_interval)
    # + volume_slot * self.num_bins_order_interval
    # + interval_slot


    for kk, vv in indices_pred.items():

        if kk not in indices_gt:
            #print("PROBLEME !!!")
            #print(kk)
            #exit()
            continue


        gt_order_infos = OrderState.get_pred_order_info_static(indices_gt[kk], NUM_BINS_PRICE_LEVEL, NUM_BINS_ORDER_VOLUME, NUM_BINS_ORDER_INTERVAL)
        gt_order_type = PredOrderInfo.get_index_from_type(gt_order_infos.order_type)
        gt_order_price = gt_order_infos.price
        gt_order_volume = gt_order_infos.volume
        gt_order_interval = gt_order_infos.interval


        pred_order_infos = OrderState.get_pred_order_info_static(vv, NUM_BINS_PRICE_LEVEL, NUM_BINS_ORDER_VOLUME, NUM_BINS_ORDER_INTERVAL)
        pred_order_type = PredOrderInfo.get_index_from_type(pred_order_infos.order_type)
        pred_order_price = pred_order_infos.price
        pred_order_volume = pred_order_infos.volume
        pred_order_interval = pred_order_infos.interval


        gt_order_type = rearrange_order_type(gt_order_type, gt_order_price)
        pred_order_type = rearrange_order_type(pred_order_type, pred_order_price)


        kk = int(kk)

        values_dico[day][stock][kk] = {}

        values_dico[day][stock][kk]["type"] = {}
        values_dico[day][stock][kk]["type"]["ground_truth"] = gt_order_type
        values_dico[day][stock][kk]["type"]["predicted"] = pred_order_type


        #  types:
        # 0: new sell,
        # 1: new buy,
        # 2: cancel,
        # 3: agressive sell,
        # 4: agressive buy

        # if
        # type_, price, mid_price


        values_dico[day][stock][kk]["price"] = {}
        values_dico[day][stock][kk]["price"]["ground_truth"] = gt_order_price
        values_dico[day][stock][kk]["price"]["predicted"] = pred_order_price

        values_dico[day][stock][kk]["volume"] = {}
        values_dico[day][stock][kk]["volume"]["ground_truth"] = gt_order_volume
        values_dico[day][stock][kk]["volume"]["predicted"] = pred_order_volume

        values_dico[day][stock][kk]["interval"] = {}
        values_dico[day][stock][kk]["interval"]["ground_truth"] = gt_order_interval
        values_dico[day][stock][kk]["interval"]["predicted"] = pred_order_interval


# print(stock_days)
# exit()Value distributions of price, volume, time difference, type and side for LOBERT model


# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Value distributions of price, volume, time difference, type for MarS Model", fontsize=14)

caption = "data gathered over: " + "; ".join(
    f"{stock} ({', '.join(days)})" for stock, days in stock_days.items()
)

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

axes[0,0].legend()

plt.tight_layout()
plt.savefig("all_distributions.png", bbox_inches="tight")
plt.close()
