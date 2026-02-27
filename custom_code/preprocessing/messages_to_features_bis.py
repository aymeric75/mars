import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import random

from multiprocessing import Pool
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Optional, Tuple

from mlib.core.exchange import Exchange
from mlib.core.exchange_config import create_exchange_config_without_call_auction
from mlib.core.limit_order import LimitOrder
from market_simulation.conf import C
from market_simulation.states.order_state import OrderState
from market_simulation.utils.bin_converter import BinConverter




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




def make_exchange_and_orderstate(
    symbol: str,
    date_str: str,
    conv: Converters,
) -> Tuple[Exchange, OrderState, pd.Timestamp]:
    ex, base_time = make_exchange(symbol, date_str)

    state = OrderState(
        num_max_orders=SEQ_LEN,
        num_bins_price_level=NUM_BINS_PRICE_LEVEL,
        num_bins_pred_order_volume=NUM_BINS_ORDER_VOLUME,
        num_bins_order_interval=NUM_BINS_ORDER_INTERVAL,
        converter=conv,
    )
    ex.register_state(state)
    return ex, state, base_time



def unit_scale_to_seconds(time_unit: str) -> float:
    # messages_df["Time"] is assumed to be an integer offset; pick the right unit
    return {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}[time_unit]






def sample_parquets(input_dir, output_file, n=200):
    dfs = []
    for f in Path(input_dir).glob("*.parquet"):
        df = pd.read_parquet(f)
        dfs.append(df.sample(min(n, len(df))))
    pd.concat(dfs, ignore_index=True).to_parquet(output_file, index=False)



def row_to_order(
    r,
    *,
    symbol: str,
    time_unit: str,
    ex: Optional[Exchange] = None,   # <-- NEW: allow price lookup from current book
) -> Optional[LimitOrder]:
    """
    Message_Type mapping given by user:
      1 = new limit
      2 = cancel
      3 = delete
      4 = visible execution  (apply as cancel/reduce on the resting order_id)
      5 = hidden execution   (skip for LOB depth coherence)
      7 = tradinghalt        (skip)
      12 = ExecutionCrossTrade (apply as cancel/reduce on the resting order_id)
    """
    msg = int(r.Message_Type)

    # Convert time offset to Timestamp*

    # print("time_unit")
    # print(time_unit)
    # breakpoint()


    t = pd.to_timedelta(int(r.Time), unit=time_unit)

    # Your convention: -1 = Bid, +1 = Ask
    direction = int(r.Direction) if not pd.isna(r.Direction) else 0
    side = "B" if direction == -1 else "S"

    order_id = int(r.Order) if not pd.isna(r.Order) else -1
    price = int(r.Price) if not pd.isna(r.Price) else 0
    size = int(r.Size) if not pd.isna(r.Size) else 0

    # Skip types we cannot/should not replay into MarS' visible LOB
    if msg in (7,):
        return None

    # Hidden execution: usually doesn't change displayed depth; skip to avoid corrupting visible LOB
    if msg == 5:
        return None

    # New limit order
    if msg == 1:
        return LimitOrder(
            time=t,
            type=side,          # "B" or "S"
            price=price,
            volume=size,
            symbol=symbol,
            agent_id=-1,
            order_id=order_id,
            cancel_type="",
            cancel_id=-1,
            tag="replay",
        )

    # Cancel/Delete (2/3): MarS models both as cancel messages reducing volume on an existing order id
    if msg in (2, 3):
        # If price is missing/0, try to infer it from the current orderbook
        if (price is None or price == 0) and ex is not None and order_id >= 0:
            try:
                price = ex.get_lob(symbol).get_price_of_order_id(order_id)
            except Exception:
                return None

        return LimitOrder(
            time=t,
            type="C",
            price=price,
            volume=size,
            symbol=symbol,
            agent_id=-1,
            order_id=-1,
            cancel_type=side,      # cancel a buy if side=="B", else cancel a sell
            cancel_id=order_id,
            tag="replay",
        )


    # # Visible execution / cross trade: reduce resting visible order volume
    # if msg in (4,):
    #     # We treat it as a cancel-on-id of 'size' shares.
    #     # If price missing/0, infer from current orderbook.
    #     if (price is None or price == 0) and ex is not None and order_id >= 0:
    #         try:
    #             price = ex.get_lob(symbol).get_price_of_order_id(order_id)
    #         except Exception:
    #             # If the order is already gone (fully executed earlier), skip quietly.
    #             return None

    #     return LimitOrder(
    #         time=t,
    #         type=side,
    #         price=price,
    #         volume=size,
    #         symbol=symbol,
    #         agent_id=-1,
    #         order_id=order_id,
    #         cancel_type=side,
    #         cancel_id=order_id,
    #         tag="replay_exec_",
    #     )




    # Visible execution / cross trade: reduce resting visible order volume
    if msg in (4,):

        # We treat it as a cancel-on-id of 'size' shares.
        # If price missing/0, infer from current orderbook.
        if (price is None or price == 0) and ex is not None and order_id >= 0:
            try:
                price = ex.get_lob(symbol).get_price_of_order_id(order_id)
            except Exception:
                # If the order is already gone (fully executed earlier), skip quietly.
                return None

        return LimitOrder(
            time=t,
            type="C",
            price=price,
            volume=size,
            symbol=symbol,
            agent_id=-1,
            order_id=-1,
            cancel_type=side,
            cancel_id=order_id,
            tag="replay_exec",
        )






    # Unknown / unsupported
    return None




def return_values_for_bins(
    messages_df: pd.DataFrame,
    symbol: str,
    time_unit: str,
    max_events: Optional[int],
    *,
    sample_every_k: int = 10,          # <-- take 1 sample every K fed events
    max_samples: Optional[int] = None, # <-- stop collecting after this many samples
):
    """
    Pass 1: feed all events to Exchange for correct book evolution,
    but only COLLECT values for bin construction every K-th fed event.
    """

    # Initialize Exchange (your original logic)
    day = pd.to_datetime(messages_df["Time"].iloc[0], unit="ns").strftime("%Y-%m-%d")
    ex, base_time = make_exchange(symbol, day)
    scale = unit_scale_to_seconds(time_unit)

    price_minus_mid: List[float] = []
    lob_vols: List[float] = []
    intervals: List[float] = []
    sizes: List[float] = []

    # Track time only at sampling points (so dt reflects sampled stream)
    prev_sample_time: Optional[int] = None

    n = len(messages_df) if max_events is None else min(len(messages_df), max_events)

    fed = 0       # number of events successfully fed to exchange
    sampled = 0   # number of samples actually collected
    dt_sec = 0

    # for i, r in enumerate(messages_df.itertuples(index=False)):
    for i, r in enumerate(tqdm(messages_df.itertuples(index=False),
                              total=n,
                              desc="pass1: bins",
                              unit="msg")):



        # interval samples (based on *sampled* events)
        cur_time = int(r.Time)
        if prev_sample_time is not None:
            dt_sec = (cur_time - prev_sample_time) * scale

        prev_sample_time = cur_time


        if i >= n:
            break

        #order = row_to_order(r, symbol=symbol, base_time=base_time, time_unit=time_unit)
        order = row_to_order(r, symbol=symbol, time_unit=time_unit, ex=ex)

        if order is None:
            continue

        # Always feed to keep book consistent
        try:

            ex.submit_continuous_auction_order(order)
        except AssertionError:
            raise

        fed += 1


        # Decide whether to record this event into bin-sample arrays
        if sample_every_k > 1 and (fed % sample_every_k) != 0:
            continue

        # Optional hard cap on collected samples
        if max_samples is not None and sampled >= max_samples:
            break

        sampled += 1

        # sizes for volume bins (only new orders)
        if order.type in ("B", "S"):
            sizes.append(float(order.volume))

        if dt_sec > 0:
            intervals.append(float(dt_sec))


        # snapshot from exchange-built orderbook
        snap = ex.get_lob(symbol).snapshot(level=10)

        # mid price requires best bid/ask
        if not snap.bid_prices or not snap.ask_prices:
            continue
        best_bid = snap.bid_prices[0]
        best_ask = snap.ask_prices[0]
        if best_bid is None or best_ask is None:
            continue
        mid = (best_bid + best_ask) / 2.0

        # signed price-minus-mid sample
        if order.type in ("B", "S") and order.price:
            price_minus_mid.append(float(order.price - mid))

        # collect top-10 volumes (positive only)
        if snap.ask_volumes:
            for v in snap.ask_volumes[:10]:
                if v and v > 0:
                    lob_vols.append(float(v))
        if snap.bid_volumes:
            for v in snap.bid_volumes[:10]:
                if v and v > 0:
                    lob_vols.append(float(v))

    return price_minus_mid, sizes, intervals, lob_vols






def pass2_write_features(
    messages_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    symbol: str,
    time_unit: str,
    conv: Converters,
    out_path: str,
    max_events: Optional[int] = None,
) -> Tuple[str, int]:


    import pyarrow as pa
    import pyarrow.parquet as pq


    day = pd.to_datetime(messages_df["Time"].iloc[0], unit="ns").strftime("%Y-%m-%d")
    ex, order_state, base_time = make_exchange_and_orderstate(symbol, day, conv)

    writer = None
    buffer = []
    CHUNK_SIZE = 50_000
    written = 0

    if os.path.exists(out_path):
        os.remove(out_path)


    n = len(messages_df) if max_events is None else min(len(messages_df), max_events)


    # We use the meta info to obtain when we are in MarketHours (between code 22 and 23)
    meta = meta_df.sort_values("Time", kind="mergesort")
    mt = meta["Time"].to_numpy()
    mc = meta["System_Event_Code"].to_numpy()
    msg_t = messages_df["Time"].to_numpy()
    j = np.searchsorted(mt, msg_t)
    j = np.clip(j, 1, len(mt) - 1)
    left = j - 1
    right = j
    nearest = np.where((msg_t - mt[left]) <= (mt[right] - msg_t), left, right)
    msg_sys_code = mc[nearest]   # aligned with messages_df rows
    del meta, mt, mc, msg_t, j, left, right, nearest

    markethours = False

    for i, r in enumerate(tqdm(messages_df.itertuples(index=False),
                              total=n,
                              desc="pass2: features",
                              disable=True,
                              unit="msg")):

        if i >= n:
            break

        if i == 0:
            order_state.open_time = r.Time

        #order = row_to_order(r, symbol=symbol, base_time=base_time, time_unit=time_unit)
        order = row_to_order(r, symbol=symbol, time_unit=time_unit, ex=ex)

        if order is None:
            continue

        if order_state.open_trans_price is None and r.Price is not None:
            order_state.open_trans_price = r.Price

        try:
            ex.submit_continuous_auction_order(order)

        except:
            raise

        if len(order_state.recent_orders) == 0:
            continue

        sys_code = msg_sys_code[i]

        if sys_code == 22:
            markethours = True
        if sys_code == 23:
            markethours = False
            # snap = ex.get_lob(symbol).snapshot(level=10)
            # print(snap)
            break

        if not markethours:
            continue

        # boolean IN_BETWEN

        feat = order_state.recent_orders[-1].to_vector()
        feat = np.asarray(feat, dtype=np.int32).reshape(-1)


        # dans index tu as déjà:   order_type, price_slot, volume_slot, interval_slot (IN SECONDS !!!!!!)
        #  le prix, on le prend tel qu'il est (eg 7268100)
        # le volume aussi,
        #



        #     # index  vol_ratio_slot  trans_ratio_slot   price_change_to_open    time_to_open     lob_volumes
        #     # f0     f1              f2                 f3                       f4             f5  f6  f7  f8  f9  f10  f11  f12  f13  f14
        #     # 10624   9              0                  0                         2147          0   0   0   0   0    0    0    0    0    0



        if feat.shape[0] != TOKEN_DIM:
            continue


        buffer.append(
            {
                "i": i,
                "Time": int(r.Time),
                **{f"f{j}": int(feat[j]) for j in range(TOKEN_DIM)},
            }
        )

        if len(buffer) >= CHUNK_SIZE:
            chunk_df = pd.DataFrame(buffer)
            table = pa.Table.from_pandas(chunk_df)

            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)

            writer.write_table(table)

            written += len(buffer)
            buffer.clear()


    if buffer:
        chunk_df = pd.DataFrame(buffer)
        table = pa.Table.from_pandas(chunk_df)

        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)

        writer.write_table(table)
        written += len(buffer)

    if writer is not None:
        writer.close()

    return out_path, written





def make_exchange(symbol: str, date_str: str = "2025-10-10") -> Tuple[Exchange, pd.Timestamp]:
    market_open = pd.Timestamp(f"{date_str} 09:30:00")
    market_close = pd.Timestamp(f"{date_str} 15:00:00")
    cfg = create_exchange_config_without_call_auction(
        market_open=market_open,
        market_close=market_close,
        symbols=[symbol],
    )
    return Exchange(cfg), market_open



def process_one_file(args):

        msg_path, converters, data_folder = args

        # Build corresponding meta filename
        meta_path = msg_path.with_name(
            msg_path.name.replace("_messages.parquet", "_meta.parquet")
        )

        if meta_path.exists():
            print("Processing pair:")
            print("  Messages:", msg_path.name)
            print("  Meta:    ", meta_path.name)

            messages_df = pd.read_parquet(msg_path)
            meta_df = pd.read_parquet(meta_path)

            output_file_name = msg_path.name.replace("_messages", "_features")

            symbol = msg_path.name.split("_")[0]

            time_unit = "ns"


            # 57600003755960
            #print(pd.to_timedelta(int(57600003755960), unit=time_unit))

            out_path, n_written = pass2_write_features(
                messages_df,
                meta_df,
                symbol=symbol,
                time_unit=time_unit,
                conv=converters,
                out_path=data_folder+"final/"+output_file_name,
                max_events=None,
            )
            print(f"Wrote {n_written} feature rows -> {out_path}")


        else:
            print(f"⚠ No meta file found for {msg_path.name}")



def main():


    data_folder = "/scratch/project_2012747/mars_data/order_model/val/"
    output_folder = "/scratch/project_2012747/mars_data/order_model/val/final"

    #data_folder = "/scratch/project_2012747/mars_data/order_model/val/"
    #output_folder = "/scratch/project_2012747/mars_data/order_model/val/final"

    """
    # I. CREATING BINs VALUES

    # Randomly select 3 files of each stock
    print(Path(data_folder + "raw").glob("*_messages.parquet"))


    raw_path = Path(data_folder) / "raw"
    files_by_ticker = defaultdict(list)
    for f in raw_path.glob("*_messages.parquet"):
        files_by_ticker[f.name.split("_")[0]].append(f)
    selected_files = [
        f
        for files in files_by_ticker.values()
        for f in random.sample(files, min(1, len(files)))
    ]

    print("selected_files")
    print(selected_files)

    all_dfs = []  # collect results here
    for f in selected_files:

        print(f.name)

        df_historical_data = pd.read_parquet(f, columns=["Time", "Step", "Message_Type", "Order", "Price", "Size", "Direction"])

        price_minus_mid, sizes, intervals, lob_vols = return_values_for_bins(
            df_historical_data,
            symbol="Whatever",
            time_unit="ns",
            max_events=10000, # None,
            sample_every_k= 50 #100000,
            #=1_000_000,  # optional
        )


        df = pd.DataFrame([{
            "price_minus_mid": price_minus_mid,
            "sizes": sizes,
            "intervals": intervals,
            "lob_vols": lob_vols,
        }])

        all_dfs.append(df)



    # 🔹 Concatenate everything after the loop
    final_df = pd.concat(all_dfs, ignore_index=True)

    # 🔹 Save once
    final_df.to_parquet(data_folder + "/intermediate/bins_samples_better.parquet", index=False)


    # II. Creating Converters

    df = pd.read_parquet(data_folder + "intermediate/bins_samples.parquet")
    print("df.shape")
    print(df.shape)
    price_minus_mid = df.loc[0, "price_minus_mid"]
    sizes         = df.loc[0, "sizes"]
    intervals     = df.loc[0, "intervals"]
    lob_vols      = df.loc[0, "lob_vols"]


    print("price_minus_mid")
    print(price_minus_mid)

    converters = build_converters_from_samples(price_minus_mid, sizes, intervals, lob_vols)

    # save converters
    with open(data_folder + "intermediate/converters.pkl", "wb") as f:
        pickle.dump(converters, f)


    """


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
    #converters = Converters(**{k: bc_from_dict(v) for k, v in blob.items()})


    print("converters.price_level.bins")
    print(converters.price_level.bins)
    print("converters.order_volume.bins")
    print(converters.order_volume.bins)
    print("converters.pred_order_volume.bins")
    print(converters.pred_order_volume.bins)
    print("converters.order_interval.bins")
    print(converters.order_interval.bins)
    print("converters.lob_volume.bins")
    print(converters.lob_volume.bins)



    # III. CREATING FEATURES FILES

    # Get all message files
    message_files = sorted(Path(data_folder+"raw").glob("*_messages.parquet"))


    feature_files = sorted(
        p.name
        for p in Path(output_folder)
            .glob("*_features.parquet")
    )

    print("feature_files 1")
    print(feature_files)

    # Convert feature filenames to the corresponding "messages" filenames
    feature_as_messages = {
        f.replace("_features.parquet", "_messages.parquet")
        for f in feature_files
    }
    
    print("feature_as_messages")
    print(feature_as_messages)

    some_list = ["AAPL", "AMD", "AMZN", "ASML", "AVGO", "COST", "GOOG", "GOOGL"]


    # Filter paths
    filtered_paths = [
        p for p in message_files
        if p.name not in feature_as_messages
    ]
    print("FILTERED PATHS BEFORE")
    print(len(filtered_paths))
    print(filtered_paths[:5])
    
    filtered_paths = [
        p for p in filtered_paths
        if any(keyword in p.name for keyword in some_list)
    ]
    """
    filtered_paths = [
        p for p in filtered_paths
        if p.name.split("_")[0] not in some_list
    ]
    """
    
    print("FILTERED PATHS AFTER")
    print(len(filtered_paths))
    print(filtered_paths[:5])

    message_files = filtered_paths

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    args = [(msg_path, converters, data_folder) for msg_path in message_files]

    with Pool(processes=n_workers) as pool:

        for _ in pool.imap_unordered(process_one_file, args):
            pass




if __name__ == "__main__":
    main()
