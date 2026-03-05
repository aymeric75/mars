
import json
from market_simulation.states.order_state import OrderState, PredOrderInfo
import pandas as pd
from pathlib import Path
import numpy as np


snapshots = pd.read_parquet(Path("data/LOBSTER_META_2025-10-01_snapshots_10.parquet"))

messages = pd.read_parquet(Path("data/LOBSTER_META_2025-10-01_messages_10.parquet"))


snapshots['mid_price'] = np.where(
    snapshots['Ask_Price_1'].notna() & snapshots['Bid_Price_1'].notna(),
    (snapshots['Ask_Price_1'] + snapshots['Bid_Price_1']) / 2,
    np.nan
)


# print(messages[(messages["Message_Type"] == 1) & (messages["Direction"] == 1)])

# exit()


# print(np.isnan(snapshots['mid_price'].unique()[0]))
# exit()

print("HEEEEEEEELOOOOOOOOOOOOOO")
with open("gt_indices.json", "r") as f:
    gt_indices = json.load(f)

with open("predicted_indices.json", "r") as f:
    predicted_indices = json.load(f)




def rearrange_order_type(type_, price):

    if type_ == 2:
        return type_

    # sell order
    if type_ == 0:

        if price > 16:
            return 0 # passive limit order
        else:
            return 3 # agressive

    # buy order
    if type_ == 1:

        if price < 16:
            return 1 # passive limit order
        else:
            return 4 # agressive

    return


TOKEN_DIM = 15
NUM_BINS_PRICE_LEVEL = 32
NUM_BINS_ORDER_VOLUME = 32
NUM_BINS_ORDER_INTERVAL = 16
NUM_BINS_LOB_VOLUME = 32



# order_type * (self.num_bins_price_level * self.num_bins_pred_order_volume * self.num_bins_order_interval)
# + price_slot * (self.num_bins_pred_order_volume * self.num_bins_order_interval)
# + volume_slot * self.num_bins_order_interval
# + interval_slot



values_dico = {}



for kk, vv in predicted_indices.items():

    if kk not in gt_indices:
        print("PROBLEME !!!")
        continue


    gt_order_infos = OrderState.get_pred_order_info_static(gt_indices[kk], NUM_BINS_PRICE_LEVEL, NUM_BINS_ORDER_VOLUME, NUM_BINS_ORDER_INTERVAL)
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



    values_dico[kk] = {}

    values_dico[kk]["type"] = {}
    values_dico[kk]["type"]["ground_truth"] = gt_order_type
    values_dico[kk]["type"]["predicted"] = pred_order_type


    #  types:
    # 0: new sell,
    # 1: new buy,
    # 2: cancel,
    # 3: agressive sell,
    # 4: agressive buy

    # if
    # type_, price, mid_price


    values_dico[kk]["price"] = {}
    values_dico[kk]["price"]["ground_truth"] = gt_order_price
    values_dico[kk]["price"]["predicted"] = pred_order_price

    values_dico[kk]["volume"] = {}
    values_dico[kk]["volume"]["ground_truth"] = gt_order_volume
    values_dico[kk]["volume"]["predicted"] = pred_order_volume

    values_dico[kk]["interval"] = {}
    values_dico[kk]["interval"]["ground_truth"] = gt_order_interval
    values_dico[kk]["interval"]["predicted"] = pred_order_interval






print(values_dico)

import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

data = values_dico



def plot_feature_distribution(values_dico, feature_name, filename, gt_color="orange"):

    # Extract values
    gt_values = [v[feature_name]["ground_truth"] for v in values_dico.values()]
    pred_values = [v[feature_name]["predicted"] for v in values_dico.values()]

    # Count occurrences
    gt_counts = Counter(gt_values)
    pred_counts = Counter(pred_values)

    # All possible classes/bins
    classes = sorted(set(gt_values) | set(pred_values))

    # Align counts
    gt = [gt_counts.get(c, 0) for c in classes]
    pred = [pred_counts.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    plt.figure()

    plt.bar(x - width/2, gt, width, label="Ground Truth", color=gt_color)
    plt.bar(x + width/2, pred, width, label="Predicted")

    plt.xlabel(feature_name.capitalize())
    plt.ylabel("Frequency")
    plt.title(f"Ground Truth vs Predicted Distribution ({feature_name})")

    plt.xticks(x, classes)
    plt.legend()

    plt.savefig(filename, bbox_inches="tight")
    plt.close()



def plot_feature_distribution(values_dico, feature_name, ax, gt_color="orange"):

    gt_values = [v[feature_name]["ground_truth"] for v in values_dico.values()]
    pred_values = [v[feature_name]["predicted"] for v in values_dico.values()]

    gt_counts = Counter(gt_values)
    pred_counts = Counter(pred_values)

    classes = sorted(set(gt_values) | set(pred_values))

    gt = [gt_counts.get(c, 0) for c in classes]
    pred = [pred_counts.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width/2, gt, width, label="GT", color=gt_color)
    ax.bar(x + width/2, pred, width, label="Pred")

    ax.set_title(feature_name)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)


# Create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

features = ["type", "price", "interval", "volume"]

for ax, feature in zip(axes.flat, features):
    plot_feature_distribution(values_dico, feature, ax)

axes[0,0].legend()

plt.tight_layout()
plt.savefig("all_distributions.png", bbox_inches="tight")
plt.close()



# plot_feature_distribution(values_dico, "type", "type_distribution.png")
# plot_feature_distribution(values_dico, "price", "price_distribution.png")
# plot_feature_distribution(values_dico, "interval", "interval_distribution.png")
# plot_feature_distribution(values_dico, "volume", "volume_distribution.png")
