""" Go through a day/stock, iterate usign the Market Engine and gather for each index
    ground truth and predicted order index
"""

import torch
import json
import pandas as pd
import pickle
import numpy as np

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from mlib.core.limit_order import LimitOrder
from market_simulation.conf import C
from market_simulation.states.order_state import OrderState, PredOrderInfo
from market_simulation.utils.bin_converter import BinConverter
from custom_code.preprocessing.order_model.messages_to_features import (
    make_exchange_and_orderstate,
    make_exchange, row_to_order,
    build_converters_from_samples,
    pass2_write_features
)




from custom_code.testing.utils import load_order_model, load_ensemble_model, load_order_batch_model

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



data_dir = Path("/scratch/project_2012747/mars_data/order_model/test/raw")

#for message_file in data_dir.glob("*_messages.parquet"):


def process_file(message_file):


    device="cpu"
    # Load the Order Model
    order_model = load_order_model(
        ckpt_path="step=step=3360-val=val_loss=3.7445.ckpt",
        device=device
    )
    order_model = order_model.to(device).eval()

    stock = message_file.stem.split("_")[0]
    day = message_file.stem.split("_")[1]

    if Path(f"jsons/{stock}_{day}_order-indices-gt.json").exists():
        print(f"existence for {stock} {day}")
        return
    print(f"processing {stock} {day}")

    meta_file = Path(data_dir / message_file.name.replace("_messages", "_meta"))
    snapshots_file = Path(data_dir / message_file.name.replace("_messages", "_snapshots"))


    messages = pd.read_parquet(message_file)
    meta_df = pd.read_parquet(meta_file)
    snapshots = pd.read_parquet(snapshots_file)


    ex, order_state, base_time = make_exchange_and_orderstate(stock, day, converters)
    n = len(messages)

    #-----------------------------------------
    # Loop over the 3 files, at the same time
    #-----------------------------------------

    tmp_last_10204_feats = np.empty((0, 15))

    batch_list = []
    batch_i_s = []


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
                                unit="msg",
                                miniters=10000)):


        msg = messages.iloc[i]
        mta = meta_df.iloc[i]
        snap = snapshots.iloc[i]


        t = msg["Time"]  # same as mta["Time"] and snap["Time"]

        order = row_to_order(msg, symbol=stock, time_unit="ns", ex=ex)

        if order is None:
            continue


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



        # the feature vector
        feat = order_state.recent_orders[-1].to_vector()


        gt_indices[i] = feat[0]

        gt_type4[i] = True if order.tag == "type_4" else False

        tmp_last_10204_feats = np.vstack((tmp_last_10204_feats, feat))

        if tmp_last_10204_feats.shape[0] > 1024:
            tmp_last_10204_feats = tmp_last_10204_feats[-1024:]


        if tmp_last_10204_feats.shape == (1024, 15):

            if i % 300 == 0 and i > 1024:

                batch_list.append(tmp_last_10204_feats)
                batch_i_s.append(i+1)


        if len(batch_list) ==  15:

            batch_array = np.stack(batch_list, axis=0)
            X = torch.from_numpy(batch_array).to(device=device, dtype=torch.long)
            base_logits = order_model(X)
            logits_next = base_logits[:, -1, :]          # (49152,)
            probs_next  = torch.softmax(logits_next, 1) # (49152,)
            pred_id = torch.argmax(probs_next, dim=1) #.item()

            for k, j in enumerate(batch_i_s):
                predicted_indices[j] = pred_id[k]

            batch_list = []
            batch_i_s = []


        # if i > 1696659:
        #     print(snap)$

    filtered_gt = {k: gt_indices[k] for k in predicted_indices if k in gt_indices}
    json.dump(filtered_gt, open(f"jsons/{stock}_{day}_order-indices-gt.json", "w"), default=lambda x: x.item())
    json.dump(predicted_indices, open(f"jsons/{stock}_{day}_order-indices-pred.json", "w"), default=lambda x: x.item())



if __name__ == "__main__":

    files = list(data_dir.glob("*_messages.parquet"))

    # '/scratch/project_2012747/mars_data/order_model/test/raw/AMZN_2025-12-09_messages.parquet

    # Filtering the list
    tickers = {"NFLX", "NVDA", "TSLA"}
    start = datetime.fromisoformat("2025-12-09")
    end = datetime.fromisoformat("2025-12-11")

    filtered = []
    for f in files:
        ticker, date_str, *_ = f.stem.split("_")
        date = datetime.fromisoformat(date_str)

        if ticker in tickers and start <= date <= end:
            filtered.append(f)


    print("filtered")
    print(filtered)

    with ProcessPoolExecutor() as ex:
        list(ex.map(process_file, filtered))
