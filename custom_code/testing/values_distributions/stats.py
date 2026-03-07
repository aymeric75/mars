""" Go through a day/stock, iterate usign the Market Engine and gather for each index
    ground truth and predicted order index
"""


import torch
import json
import pandas as pd
from pathlib import Path

messages = pd.read_parquet(Path("../data/LOBSTER_META_2025-10-01_messages_10.parquet"))
meta_df = pd.read_parquet(Path("../data/LOBSTER_META_2025-10-01_meta_10.parquet"))
snapshots = pd.read_parquet(Path("../data/LOBSTER_META_2025-10-01_snapshots_10.parquet"))


#-----------------------------------------
# Create an "Order State" and an exchange
#-----------------------------------------
import pickle
import numpy as np

from tqdm import tqdm

from dataclasses import dataclass

from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from mlib.core.limit_order import LimitOrder
from market_simulation.conf import C
from market_simulation.states.order_state import OrderState, PredOrderInfo
from market_simulation.utils.bin_converter import BinConverter

from ...preprocessing.order_model.messages_to_features import make_exchange_and_orderstate, make_exchange, row_to_order, build_converters_from_samples, pass2_write_features
exit()
from utils import load_order_model

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

# with open("converters.pkl", "rb") as f:
#     converters = pickle.load(f)



import json

with open("converters_portable.json", "r", encoding="utf-8") as f:
    obj = json.load(f)

#price_minus_mid = blob["state"]["price_level"]["bin_values"]["data"]

price_minus_mid = []
for bin_item in obj["state"]["price_level"]["bin_values"]:
    price_minus_mid.extend(bin_item["data"])

#sizes = blob["state"]["order_volume"]["bin_values"]["data"]
sizes = []
for bin_item in obj["state"]["order_volume"]["bin_values"]:
    sizes.extend(bin_item["data"])

#intervals = blob["state"]["order_interval"]["bin_values"]["data"]
intervals = []
for bin_item in obj["state"]["order_interval"]["bin_values"]:
    intervals.extend(bin_item["data"])



#lob_vols = blob["state"]["lob_volume"]["bin_values"]["data"]
lob_vols = []
for bin_item in obj["state"]["lob_volume"]["bin_values"]:
    lob_vols.extend(bin_item["data"])


converters = build_converters_from_samples(price_minus_mid, sizes, intervals, lob_vols)



# print("converters")
# print(converters.price_level.bins)
# #de indice 0 à 16 inclus, c'est prix négatif p rapport au milieu, donc BID side
# exit()



symbol = "META"
ex, order_state, base_time = make_exchange_and_orderstate(symbol, "2025-10-01", converters)
n = len(messages)

print(type(base_time))
print(base_time)

#n = 3000


#-----------------------------------------
# Loop over the 3 files, at the same time
#-----------------------------------------

count_buys = 0
count_sells = 0
count_cancels = 0
count_visible_exec = 0
total_transactions = 0

device="cpu"
# Load the Order Model
order_model = load_order_model(
    ckpt_path="step=step=3360-val=val_loss=3.7445.ckpt",
    device=device
)


order_model = order_model.to(device).eval()

tmp_last_10204_feats = np.empty((0, 15))

batch_list = []
batch_i_s = []

#   (Batch, 1024, 15)
#
#       s

counter_added_ = 0

pred_from_prev = None

gt_indices = {}
gt_type4 = {}
predicted_indices = {}



# We use the meta info to obtain when we are in MarketHours (between code 22 and 23)
meta = meta_df.sort_values("Time", kind="mergesort")
mt = meta["Time"].to_numpy()
mc = meta["System_Event_Code"].to_numpy()
msg_t = messages["Time"].to_numpy()
j = np.searchsorted(mt, msg_t)
j = np.clip(j, 1, len(mt) - 1)
left = j - 1
right = j
nearest = np.where((msg_t - mt[left]) <= (mt[right] - msg_t), left, right)
msg_sys_code = mc[nearest]   # aligned with messages_df rows
del meta, mt, mc, msg_t, j, left, right, nearest

markethours = False
# i AND THEN :   order_index,


for i, r in enumerate(tqdm(messages.itertuples(index=False),
                            total=n,
                            desc="pass",
                            unit="msg")):


    msg = messages.iloc[i]
    mta = meta_df.iloc[i]
    snap = snapshots.iloc[i]


    t = msg["Time"]  # same as mta["Time"] and snap["Time"]

    order = row_to_order(msg, symbol=symbol, time_unit="ns", ex=ex)

    if order is None:
        continue


    # print(type(order_state.open_time))
    # print(order_state.open_time)

    # if i > 20:
    #     break


    try:
        trade_infos = ex.submit_continuous_auction_order(order)
    except AssertionError:
        raise




    sys_code = msg_sys_code[i]

    if sys_code == 22:
        markethours = True
        order_state.open_time = pd.Timedelta(hours=9, minutes=30)

    if sys_code == 23:
        markethours = False
        break
    if not markethours:
        continue

    if trade_infos:
        for trade_info in trade_infos:
            if trade_info.transactions:
                for trans in trade_info.transactions:
                    if trans.type == "B":
                        count_buys+=1
                    if trans.type == "S":
                        count_sells+=1
                    if trans.type == "C":
                        count_cancels+=1
                    total_transactions += 1

    # the feature vector
    feat = order_state.recent_orders[-1].to_vector()

    if i == 101401:

        print("ON EST ICI ")
        exit()

    gt_indices[i] = feat[0]

    gt_type4[i] = True if order.tag == "type_4" else False

    tmp_last_10204_feats = np.vstack((tmp_last_10204_feats, feat))

    if tmp_last_10204_feats.shape[0] > 1024:
        tmp_last_10204_feats = tmp_last_10204_feats[-1024:]


    if tmp_last_10204_feats.shape == (1024, 15):

        if i % 300 == 0 and i > 1024:

            batch_list.append(tmp_last_10204_feats)
            batch_i_s.append(i+1)



    # last seq added is   the NEWEST one,  so the first element of the list is THE OLDEST
    # predicted type 4 ?

    #


    if len(batch_list) ==  15:

        batch_array = np.stack(batch_list, axis=0)
        X = torch.from_numpy(batch_array).to(device=device, dtype=torch.long)
        base_logits = order_model(X)
        logits_next = base_logits[:, -1, :]          # (49152,)
        probs_next  = torch.softmax(logits_next, 1) # (49152,)
        pred_id = torch.argmax(probs_next, dim=1) #.item()

        print("pred_idpred_idpred_id")
        print(pred_id)
        # tensor([32768,  8720,  8726, 16384, 16384, 12896, 16384, 16384, 39020, 24576,
        #         24279, 16395, 43625,  8704, 39020])

        for k, j in enumerate(batch_i_s):
            predicted_indices[j] = pred_id[k]

        batch_list = []
        batch_i_s = []


    if i > 1696659:
        print(snap)


        # quand c'est type 4, alors trans.type doit correspondre au type donné DANS LE MESSAGE (r.Message_Type)
        #print("CONCLUSION: NEAR END SNAPS SHOULD CORRESPOND TO GROUND TRUTH!!")


# 101401

import json
import numpy as np


filtered_gt = {k: gt_indices[k] for k in predicted_indices if k in gt_indices}

json.dump(filtered_gt, open("gt_indices.json", "w"), default=lambda x: x.item())



json.dump(predicted_indices, open("predicted_indices.json", "w"), default=lambda x: x.item())


# with open("gt_indices.json", "w") as f:
#     json.dump(gt_indices, f)

# with open("predicted_indices.json", "w") as f:
#     json.dump(predicted_indices, f)

# 157178
# print(gt_indices)
# print(predicted_indices)


# # 64065
# perc_buys = round((count_buys / total_transactions) * 100)
# perc_sells = round((count_sells / total_transactions) * 100)
# perc_cancels = round((count_cancels / total_transactions) * 100)

# print("count buys ", count_buys)
# print("count sells ", count_sells)
# print("count cancels ", count_cancels)
# print("count_visible_exec ", count_visible_exec)
# print("count total ", total_transactions)
# print()
# print("perc_buys ", perc_buys)
# print("perc_sells ", perc_sells)
# print("perc_cancels ", perc_cancels)

#-----------------------------------------------------------------------------------------------------
# Next, COUNT THE NUMBER OF TRADES, AND ALSO, SEE whatever "trade" is being made, or in a large sense,
# what are all the scenarios in which the Order Book is updated
#-----------------------------------------------------------------------------------------------------
