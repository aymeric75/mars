import sys
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from typing import Optional, Tuple, List
from pathlib import Path

from mlib.core.trade_info import TradeInfo
from mlib.core.lob_snapshot import LobSnapshot
from market_simulation.utils import pkl_utils

from utils import Converters, make_exchange_and_orderstate, row_to_order

from report_stylized_facts import get_minute_info

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
        tqdm(messages_df.itertuples(index=False), total=n, desc="replay", unit="msg")
    ):


        # print(r)
        # print(r.Time)



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




with open("../preprocessing/converters.pkl", "rb") as f:
    converters = pickle.load(f)

messages_df = pd.read_parquet("../preprocessing/data/LOBSTER_META_2025-10-01_messages_10.parquet")
meta_df = pd.read_parquet("../preprocessing/data/LOBSTER_META_2025-10-01_meta_10.parquet")

list_of_replay_trade_infos_lists, start_lob = build_replay_trade_infos(
    messages_df,
    meta_df,
    symbol="META",
    day="2025-10-01",
    conv=converters,
    max_events=None
)


# print("JUS TO SHOW THE RESULT ")
# print(list_of_replay_trade_infos_lists)

for i, list_ in enumerate(list_of_replay_trade_infos_lists):

    pkl_utils.save_pkl_zstd(
        [(list_, start_lob), (list_, start_lob)],
        Path("folders/tradeInfos__replay_"+str(i)+".zstd")
    )









# minutes = get_minute_info(replay_trade_infos, start_lob)

# pkl_utils.save_pkl_zstd(
#     minutes,
#     Path("my_minutes_replay.zstd")
# )


# print("start_lob")
# print(start_lob)
