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

from messages_to_features import make_exchange_and_orderstate, make_exchange, row_to_order

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

with open("converters.pkl", "rb") as f:
    converters = pickle.load(f)


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

    try:
        trade_infos = ex.submit_continuous_auction_order(order)
    except AssertionError:
        raise


    ########### replay_exec_

    if trade_infos:

        for trade_info in trade_infos:

            if trade_info.transactions:

                for trans in trade_info.transactions:

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
                        if order.tag == "replay_exec":
                            # print("IN REPLAYX ")
                            # print(trans.price)
                            # breakpoint()
                            count_visible_exec+=1
                        else:
                            print("NOT IN REPLAY")
                            print(trans.price)
                            breakpoint()
                    total_transactions += 1

                    #break

    if i > 1696659:
        print(snap)


        print("CONCLUSION: NEAR END SNAPS SHOULD CORRESPOND TO GROUND TRUTH!!")


# 64065
perc_buys = round((count_buys / total_transactions) * 100)
perc_sells = round((count_sells / total_transactions) * 100)
perc_cancels = round((count_cancels / total_transactions) * 100)

print("count buys ", count_buys)
print("count sells ", count_sells)
print("count cancels ", count_cancels)
print("count_visible_exec ", count_visible_exec)
print("count total ", total_transactions)
print()
print("perc_buys ", perc_buys)
print("perc_sells ", perc_sells)
print("perc_cancels ", perc_cancels)

#-----------------------------------------------------------------------------------------------------
# Next, COUNT THE NUMBER OF TRADES, AND ALSO, SEE whatever "trade" is being made, or in a large sense,
# what are all the scenarios in which the Order Book is updated
#-----------------------------------------------------------------------------------------------------
