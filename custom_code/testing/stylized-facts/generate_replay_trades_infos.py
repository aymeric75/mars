import sys
import re
import json
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from typing import Optional, Tuple, List
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import defaultdict

from mlib.core.trade_info import TradeInfo
from mlib.core.lob_snapshot import LobSnapshot
from market_simulation.utils import pkl_utils
from market_simulation.utils.bin_converter import BinConverter
from utils import Converters, make_exchange_and_orderstate, row_to_order

from report_stylized_facts import get_minute_info





SEQ_LEN = 1 # 1024
TOKEN_DIM = 15
NUM_BINS_PRICE_LEVEL = 32
NUM_BINS_ORDER_VOLUME = 32
NUM_BINS_ORDER_INTERVAL = 16
NUM_BINS_LOB_VOLUME = 32








def build_converters_from_samples(price_minus_mid, sizes, intervals, lob_vols):
    """
    Create BinConverters
    """

    pm = [float(x) for x in price_minus_mid if x is not None and np.isfinite(x)]
    price_level = BinConverter.create_from_values(pm, NUM_BINS_PRICE_LEVEL)

    ov = [float(x) for x in sizes if x is not None and x > 0]
    order_volume = BinConverter.create_from_values(ov, NUM_BINS_ORDER_VOLUME)

    itv = [float(x) for x in intervals if x is not None and x > 0]
    order_interval = BinConverter.create_from_values(itv, NUM_BINS_ORDER_INTERVAL)

    lv = [float(x) for x in lob_vols if x is not None and x > 0]
    lob_volume = BinConverter.create_from_values(lv, NUM_BINS_LOB_VOLUME)

    return Converters(price_level, order_volume, order_volume, order_interval, lob_volume)







def build_replay_trade_infos(
    messages_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    symbol: str,
    day: str,
    conv: Converters,
    time_unit: str = "ns",
    max_events: Optional[int] = None,
    snapshot_level: int = 10,
) -> Tuple[List[TradeInfo], LobSnapshot]:
    """
    Replays historical messages through the MarS exchange to produce replay TradeInfo list
    + the start LOB snapshot (at MarketHours start, sys_code==22).

    Mirrors the MarketHours gating logic of pass2_write_features.
    """

    ex, order_state, _base_time = make_exchange_and_orderstate(symbol, day, conv)


    # Align meta System_Event_Code to each message row (same as pass2_write_features)
    meta = meta_df.sort_values("Time", kind="mergesort")
    mt = meta["Time"].to_numpy()
    mc = meta["System_Event_Code"].to_numpy()
    msg_t = messages_df["Time"].to_numpy()
    j = np.searchsorted(mt, msg_t)
    j = np.clip(j, 1, len(mt) - 1)
    left = j - 1
    right = j

    nearest = np.where((msg_t - mt[left]) <= (mt[right] - msg_t), left, right)
    msg_sys_code = mc[nearest]

    n = len(messages_df) if max_events is None else min(len(messages_df), max_events)

    counter = 0

    count_buys = 0
    count_sells = 0
    count_cancels = 0
    count_visible_exec = 0
    total_transactions = 0



    markethours = False
    start_lob: Optional[LobSnapshot] = None
    replay_trade_infos: List[TradeInfo] = []

    time_zero_seq = None
    time_zero_min = None


    list_of_lists = []
    tmp_trade_infos = []

    trade_infos_list = []

    for i, r in enumerate(
        tqdm(messages_df.itertuples(index=False), total=n, desc="replay", unit="msg", miniters=10000)
    ):


        if i == 0:
            order_state.open_time = r.Time

        order = row_to_order(r, symbol=symbol, time_unit=time_unit, ex=ex)



        if order is None:
            continue

        if order_state.open_trans_price is None and getattr(r, "Price", None) is not None:
            order_state.open_trans_price = r.Price

        trade_infos = ex.submit_continuous_auction_order(order)

        sys_code = int(msg_sys_code[i])

        if sys_code == 22 and not markethours:
            markethours = True
            start_lob = ex.get_lob(symbol).snapshot(level=snapshot_level)
            start_lob = start_lob._replace(time=_base_time)

        if sys_code == 23:
            break

        if not markethours:
            continue



        # quand tu détecte un "B" (ou un "S")

        # 6e+10
        # 60000000000
        # 14400039340610
        # 1.56e+12
        # 1560000000000

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

                    if order.tag == "replay_exec":


                        if not tmp_trade_infos:
                            tmp_trade_infos.append(trade_info)
                            time_zero_seq = order.time
                            time_last_added_order = order.time
                        else:

                            dist_from_last_order = (order.time - time_last_added_order).value
                            dist_from_begin_seq = (order.time - time_zero_seq).value

                            # si on est encore dans les 26 minutes
                            if dist_from_begin_seq < 1560000000000:
                                #print("HHHEREEEEEEE")
                                # 60000000000
                                # si le dernier ordre ajouté était il y a moins de 60 seconde
                                if dist_from_last_order < 60000000000:
                                    #print("ET DONC ")
                                    tmp_trade_infos.append(trade_info)
                                    time_last_added_order = order.time
                                else:
                                    # print("BEFORE WE DELETE ")
                                    # print(tmp_trade_infos)
                                    # breakpoint()
                                    tmp_trade_infos = [] # sinon on reset la liste (car on aura une liste avec des ordres distants de + d'une minute)

                            # si on a dépassé les 26 minutes
                            else:
                                # print("AND GHERE ..???")
                                # print(len(tmp_trade_infos))
                                # breakpoint()
                                if tmp_trade_infos:
                                    assert len(tmp_trade_infos) > 25
                                    list_of_lists.append(tmp_trade_infos)
                                    tmp_trade_infos = []


        if len(list_of_lists) > 10:
            break



        if counter > n:
            break

        counter+=1

    return list_of_lists, start_lob




"""

with open("../preprocessing/converters.pkl", "rb") as f:
    converters = pickle.load(f)



# Directory containing the parquet files
data_dir = Path("/scratch/project_2012747/mars_data/experiments/stylized_facts")

# Regex pattern to extract stock, date, type, and optional suffix
pattern = re.compile(
    r"LOBSTER_(?P<stock>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<type>messages|meta)_\d+\.parquet"
)

# Dictionary to collect pairs
pairs = defaultdict(dict)

# Iterate over complete pairs only
for (stock, date), files in pairs.items():
    if "messages" in files and "meta" in files:
        messages_df = files["messages"]
        meta_df = files["meta"]



        list_of_replay_trade_infos_lists, start_lob = build_replay_trade_infos(
            messages_df,
            meta_df,
            symbol=str(stock),
            day=str(date),
            conv=converters,
            max_events=None
        )


        for i, list_ in enumerate(list_of_replay_trade_infos_lists):

            pkl_utils.save_pkl_zstd(
                [(list_, start_lob), (list_, start_lob)],
                Path(f"trade_infos/tradeInfos__replay_{stock}_{date}_{i}.zstd")
            )




"""



CONVERTERS = None


def _init_worker(converters_json_path: str):

    global CONVERTERS




    with open(converters_json_path, "r", encoding="utf-8") as f:
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


    CONVERTERS = build_converters_from_samples(price_minus_mid, sizes, intervals, lob_vols)










def _process_pair(args):
    stock, date, msg_path, meta_path, out_dir = args

    messages_df = pd.read_parquet(msg_path)
    meta_df = pd.read_parquet(meta_path)

    lists, start_lob = build_replay_trade_infos(
        messages_df,
        meta_df,
        symbol=stock,
        day=date,
        conv=CONVERTERS,
        max_events=None,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, lst in enumerate(lists):
        pkl_utils.save_pkl_zstd(
            [(lst, start_lob), (lst, start_lob)],
            out_dir / f"tradeInfos__replay_{stock}_{date}_{i}.zstd",
        )

    return stock, date, len(lists)


def main():
    converters_json = "converters_portable.json"
    #data_dir = Path("/scratch/project_2012747/mars_data/experiments/stylized_facts")
    data_dir = Path("some_data")
    out_dir = Path("trade_infos")

    pat = re.compile(
        r"(?P<stock>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_"
        r"(?P<type>messages|meta).parquet"
    )

    pairs = defaultdict(dict)
    for p in data_dir.glob("*.parquet"):
        m = pat.match(p.name)
        if not m:
            continue
        key = (m["stock"], m["date"])
        pairs[key][m["type"]] = p


    tasks = [
        (stock, date, files["messages"], files["meta"], str(out_dir))
        for (stock, date), files in pairs.items()
        if "messages" in files and "meta" in files
    ]

    with Pool(processes=cpu_count(), initializer=_init_worker, initargs=(converters_json,)) as pool:
        for stock, date, n_lists in pool.imap_unordered(_process_pair, tasks, chunksize=1):
            print(f"{stock} {date}: wrote {n_lists} outputs")


if __name__ == "__main__":
    main()
