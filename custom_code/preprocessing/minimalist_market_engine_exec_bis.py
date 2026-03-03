""" from one *messages.parquet and corresponding *meta.parquet and *lobsnapshot, call the Market Engine in order to check the LOB snapshot validity

    (optionnaly aims to answer the questions:
        what percentage of transactions are done (vs percentage of orders)
        how "orders" are matched with each others (or more generally what are the different scenarios that update the LOB, and how)
    )

"""


#---------------------------
# call the different files
#---------------------------

import pandas as pd
from pathlib import Path

messages = pd.read_parquet(Path("data/LOBSTER_META_2025-10-01_messages_10.parquet"))
meta = pd.read_parquet(Path("data/LOBSTER_META_2025-10-01_meta_10.parquet"))
snapshots = pd.read_parquet(Path("data/LOBSTER_META_2025-10-01_snapshots_10.parquet"))


print(messages[messages["Message_Type"] == 4])
#print(messages.head(11))
#exit()

#-----------------------------------------
# Create an "Order State" and an exchange
#-----------------------------------------
import pickle
from tqdm import tqdm

from dataclasses import dataclass

from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from mlib.core.limit_order import LimitOrder
from market_simulation.conf import C
from market_simulation.states.order_state import OrderState
from market_simulation.utils.bin_converter import BinConverter

from messages_to_features import make_exchange_and_orderstate, make_exchange, row_to_order, build_converters_from_samples, pass2_write_features

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

print("OIOOOO")

symbol = "META"
ex, order_state, base_time = make_exchange_and_orderstate(symbol, "2025-10-01", converters)
n = len(messages)


#-----------------------------------------
# Loop over the 3 files, at the same time
#-----------------------------------------

count_buys = 0
count_sells = 0
count_cancels = 0
count_visible_exec = 0
total_transactions = 0




pass2_write_features(
    messages,
    meta,
    symbol = "META",
    time_unit = "ns",
    conv = converters,
    out_path="some_folder.parquet"
)

exit()





for i, r in enumerate(tqdm(messages.itertuples(index=False),
                            total=n,
                            desc="pass",
                            unit="msg")):


    msg = messages.iloc[i]
    mta = meta.iloc[i]
    snap = snapshots.iloc[i]


    t = msg["Time"]  # same as mta["Time"] and snap["Time"]

    order = row_to_order(msg, symbol=symbol, time_unit="ns", ex=ex)

    if order is None:
        continue

    # if i > 12:
    #     break



    #if r.Messate_type

    try:
        trade_infos = ex.submit_continuous_auction_order(order)
    except AssertionError:
        raise


    ########### replay_exec_

    if trade_infos:

        for trade_info in trade_infos:

            if trade_info.transactions:

                for trans in trade_info.transactions:


                    # if r.Message_Type == 4:
                    #     print("on y esst")
                    #     print(r)
                    #     print("trans.type")
                    #     print(trans.type)
                    #     print(trade_info.order.type)
                    #     exit()



                    # if float(trans.price) > 0:
                    #     print(">>>>>>>> 0")
                    #     print(trans)
                    #     print(trans.price)
                    #     breakpoint()

                    if trans.type == "B":
                        count_buys+=1
                    if trans.type == "S":
                        count_sells+=1
                    if trans.type == "C":
                        count_cancels+=1
                        # if order.tag == "replay_exec":
                        #     # print("IN REPLAYX ")
                        #     # print(trans.price)
                        #     # breakpoint()
                        #     count_visible_exec+=1
                        # else:
                        #     print("NOT IN REPLAY")
                        #     print(trans.price)
                        #     #breakpoint()
                    total_transactions += 1

                    #break

    if i > 1696659:
        print(snap)


        # quand c'est type 4, alors trans.type doit correspondre au type donné DANS LE MESSAGE (r.Message_Type)

        #

        #print("CONCLUSION: NEAR END SNAPS SHOULD CORRESPOND TO GROUND TRUTH!!")


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
