import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle


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



def row_to_order(
    r,
    *,
    symbol: str,
    base_time: pd.Timestamp,
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


    #
    # # Visible execution / cross trade: TURN INTO AGGRESSIVE ORDER (NOT CANCEL)
    # #if msg in (4, 12):*
    # if msg in (4,):
    #     # direction encodes the RESTING side in your data (-1 bid, +1 ask)
    #     # aggressor is the opposite side
    #     if direction == -1:
    #         aggressor_side = "S"   # sell hits resting bid
    #     elif direction == +1:
    #         aggressor_side = "B"   # buy lifts resting ask
    #     else:
    #         return None

    #     # ensure we have a price (infer from book if needed)
    #     if (price is None or price == 0) and ex is not None:
    #         # fall back to best price on the resting side
    #         snap = ex.get_lob(symbol).snapshot(level=1)
    #         if aggressor_side == "S":
    #             # selling into bids => use best bid
    #             if not snap.bid_prices or snap.bid_prices[0] is None:
    #                 return None
    #             price = int(snap.bid_prices[0])
    #         else:
    #             # buying into asks => use best ask
    #             if not snap.ask_prices or snap.ask_prices[0] is None:
    #                 return None
    #             price = int(snap.ask_prices[0])

    #     # IMPORTANT:
    #     # give it a fresh order_id (<0) so Exchange assigns one, since this is "incoming aggressor"
    #     return LimitOrder(
    #         time=t,
    #         type=aggressor_side,   # "B" or "S"
    #         price=price,
    #         volume=size,
    #         symbol=symbol,
    #         agent_id=-1,
    #         order_id=-1,
    #         cancel_type="",
    #         cancel_id=-1,
    #         tag="replay_exec_as_aggressor",
    #     )



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


    # for i, r in enumerate(messages_df.itertuples(index=False)):
    for i, r in enumerate(tqdm(messages_df.itertuples(index=False),
                              total=n,
                              desc="pass1: bins",
                              unit="msg")):


        if i >= n:
            break

        #order = row_to_order(r, symbol=symbol, base_time=base_time, time_unit=time_unit)
        order = row_to_order(r, symbol=symbol, base_time=base_time, time_unit=time_unit, ex=ex)

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

        # interval samples (based on *sampled* events)
        cur_time = int(r.Time)
        if prev_sample_time is not None:
            dt_sec = (cur_time - prev_sample_time) * scale
            if dt_sec > 0:
                intervals.append(float(dt_sec))
        prev_sample_time = cur_time

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
    *,
    symbol: str,
    time_unit: str,
    conv: Converters,
    out_path: str,
    max_events: Optional[int] = None,
) -> Tuple[str, int]:
    day = pd.to_datetime(messages_df["Time"].iloc[0], unit="ns").strftime("%Y-%m-%d")
    ex, order_state, base_time = make_exchange_and_orderstate(symbol, day, conv)

    rows: List[dict] = []
    n = len(messages_df) if max_events is None else min(len(messages_df), max_events)

    for i, r in enumerate(tqdm(messages_df.itertuples(index=False),
                              total=n,
                              desc="pass2: features",
                              unit="msg")):


        if i >= n:
            break

        #order = row_to_order(r, symbol=symbol, base_time=base_time, time_unit=time_unit)
        order = row_to_order(r, symbol=symbol, base_time=base_time, time_unit=time_unit, ex=ex)

        if order is None:
            continue


        if order_state.open_trans_price is None and r.Price is not None:
            order_state.open_trans_price = r.Price

        try:

            # #if int(r.Time) > 57523138517770:
            # if i == 10000:

            #     print("r.Time is ")
            #     print(r.Time)

            #     print("Row")
            #     print(r)

            #     print("Order")
            #     print(order)

            #     # snapshot from exchange-built orderbook
            #     snap = ex.get_lob(symbol).snapshot(level=10)
            #     print("snap")
            #     print(snap)
            #     breakpoint()

            ex.submit_continuous_auction_order(order)

        except:
            raise




        # We want the vector from OrderInfo (not OrderState).
        # OrderInfo objects are stored in order_state.recent_orders.
        if len(order_state.recent_orders) == 0:
            continue

        feat = order_state.recent_orders[-1].to_vector()

        feat = np.asarray(feat, dtype=np.int32).reshape(-1)

        # # 100000
        # if i > 100000:

        #     print(r.Price)
        #     snap = ex.get_lob(symbol).snapshot(level=10)
        #     print("snapsnapsnapsnap")
        #     print(snap) # 3087800
        #     #breakpoint()
        #     # ( 3087800 / 3642300 ) - 1 =  -0.15223896988
        #     # X 10000 = -1522.3896988

        #     # print("order_state.open_trans_price")
        #     # print(order_state.open_trans_price) # 3642300
        #     # print("r.Price")
        #     # print(r.Price) # 2533300
        #     print("feat")
        #     print(feat)
        #     # if i > 100:
        #     #     breakpoint()

        #     # index  vol_ratio_slot  trans_ratio_slot   price_change_to_open    time_to_open     lob_volumes
        #     # f0     f1              f2                 f3                       f4             f5  f6  f7  f8  f9  f10  f11  f12  f13  f14
        #     # 10624   9              0                  0                         2147          0   0   0   0   0    0    0    0    0    0



        if feat.shape[0] != TOKEN_DIM:
            continue


        rows.append(
            {
                "i": i,
                "Time": int(r.Time),
                **{f"f{j}": int(feat[j]) for j in range(TOKEN_DIM)},
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(out_path, index=False)
    return out_path, len(out_df)





def make_exchange(symbol: str, date_str: str = "2025-10-10") -> Tuple[Exchange, pd.Timestamp]:
    market_open = pd.Timestamp(f"{date_str} 09:30:00")
    market_close = pd.Timestamp(f"{date_str} 15:00:00")
    cfg = create_exchange_config_without_call_auction(
        market_open=market_open,
        market_close=market_close,
        symbols=[symbol],
    )
    return Exchange(cfg), market_open



def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--messages", required=True, help="Path to *_messages_*.parquet (messages only)")
    ap.add_argument("--symbol", default="APPL", help="Exchange symbol name (any string)")
    ap.add_argument("--time-unit", default="ns", choices=["ns", "us", "ms", "s"], help="Unit of messages_df['Time']")
    ap.add_argument("--max-events", type=int, default=None, help="Process at most this many rows (debug)")
    args = ap.parse_args()

    messages_df_historical_data = pd.read_parquet("../data/2025-10-09_messages_10.parquet", columns=["Time", "Step", "Message_Type", "Order", "Price", "Size", "Direction"])


    # price_minus_mid, sizes, intervals, lob_vols = return_values_for_bins(
    #     messages_df_historical_data,
    #     symbol=args.symbol,
    #     time_unit=args.time_unit,
    #     max_events=None,
    #     sample_every_k=50,
    #     #=1_000_000,  # optional
    # )

    # df = pd.DataFrame([{
    #     "price_minus_mid": price_minus_mid,
    #     "sizes": sizes,
    #     "intervals": intervals,
    #     "lob_vols": lob_vols,
    # }])

    # df.to_parquet("../data/bins_samples.parquet", index=False)

    # print("finished")

    df = pd.read_parquet("../data/bins_samples.parquet")
    price_minus_mid = df.loc[0, "price_minus_mid"]
    sizes         = df.loc[0, "sizes"]
    intervals     = df.loc[0, "intervals"]
    lob_vols      = df.loc[0, "lob_vols"]



    converters = build_converters_from_samples(price_minus_mid, sizes, intervals, lob_vols)

    # print("converters.price_level.bins")
    # print(converters.price_level.bins)
    # print("converters.order_volume.bins")
    # print(converters.order_volume.bins)
    # print("converters.pred_order_volume.bins")
    # print(converters.pred_order_volume.bins)
    # print("converters.order_interval.bins")
    # print(converters.order_interval.bins)
    # print("converters.lob_volume.bins")
    # print(converters.lob_volume.bins)
    # breakpoint()


    # print(converters.order_interval.get_bin_index(0))


    messages_df = pd.read_parquet("../data/2025-10-10_messages_10.parquet", columns=["Time", "Step", "Message_Type", "Order", "Price", "Size", "Direction"])


    # les features ET ENSUITE ?????

    out_path, n_written = pass2_write_features(
        messages_df,
        symbol=args.symbol,
        time_unit=args.time_unit,
        conv=converters,
        out_path="../data/mymessages.parquet",
        max_events=args.max_events,
    )
    print(f"Wrote {n_written} feature rows -> {out_path}")



if __name__ == "__main__":
    main()
