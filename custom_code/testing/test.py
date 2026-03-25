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
from custom_code.preprocessing.order_model.messages_to_features_no_engine import (
    build_converters_from_samples,
)
from custom_code.testing.values_distributions.messages_to_features import (
    make_exchange_and_orderstate,
    make_exchange, row_to_order,
    pass2_write_features
)


device = "cpu"
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


CONVERTERS_JSON = Path(__file__).resolve().parents[1] / "training" / "converters_portable.json"

with CONVERTERS_JSON.open("r", encoding="utf-8") as f:
    obj = json.load(f)

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


print(converters.price_level.bin_values)
