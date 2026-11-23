"""from one *messages.parquet and corresponding *meta.parquet and *lobsnapshot, call the Market Engine in order to check the LOB snapshot validity

(optionnaly aims to answer the questions:
    what percentage of transactions are done (vs percentage of orders)
    how "orders" are matched with each others (or more generally what are the different scenarios that update the LOB, and how)
)

"""


# ---------------------------
# call the different files
# ---------------------------

import pandas as pd
from pathlib import Path

messages = pd.read_parquet(Path("data/LOBSTER_META_2025-10-01_messages_10.parquet"))
meta = pd.read_parquet(Path("data/LOBSTER_META_2025-10-01_meta_10.parquet"))
snapshots = pd.read_parquet(Path("data/LOBSTER_META_2025-10-01_snapshots_10.parquet"))


# -----------------------------------------
# Create an "Order State" and an exchange
# -----------------------------------------
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

SEQ_LEN = 1  # 1024
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


# -----------------------------------------
# Loop over the 3 files, at the same time
# -----------------------------------------

for i, r in enumerate(tqdm(messages.itertuples(index=False), total=n, desc="pass1: bins", unit="msg")):
    msg = messages.iloc[i]
    mta = meta.iloc[i]
    snap = snapshots.iloc[i]

    t = msg["Time"]  # same as mta["Time"] and snap["Time"]

    order = row_to_order(msg, symbol=symbol, time_unit="ns", ex=ex)

    if order is None:
        continue

    try:
        ex.submit_continuous_auction_order(order)
    except AssertionError:
        raise

    if i > 1696659:
        print(snap)

        print("CONCLUSION: NEAR END SNAPS SHOULD CORRESPOND TO GROUND TRUTH!!")


# -----------------------------------------------------------------------------------------------------
# Next, COUNT THE NUMBER OF TRADES, AND ALSO, SEE whatever "trade" is being made, or in a large sense,
# what are all the scenarios in which the Order Book is updated
# -----------------------------------------------------------------------------------------------------
