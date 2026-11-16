import numpy as np
import pandas as pd
import torch
import random
import pickle
from collections import deque
from typing import Optional, Tuple, List
from pathlib import Path

from mlib.core.limit_order import LimitOrder
from mlib.core.trade_info import TradeInfo
from mlib.core.lob_snapshot import LobSnapshot
from market_simulation.utils import pkl_utils
from report_stylized_facts import get_minute_info

from utils import Converters, make_exchange_and_orderstate, row_to_order, load_ensemble_model, load_order_batch_model, load_order_model


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def fix_converter(obj):
    if not hasattr(obj, "bin_values"):
        return
    vals = obj.bin_values

    # probs attribute name (common patterns)
    probs = None
    for pname in ("bin_probs", "bin_prob", "bin_p", "probs"):
        if hasattr(obj, pname):
            probs = getattr(obj, pname)
            probs_name = pname
            break

    for i in range(1, len(vals)):
        if vals[i].size == 0:
            vals[i] = vals[i - 1].copy()

        if probs is not None:
            if probs[i].size == 0:
                probs[i] = probs[i - 1].copy()
            if probs[i].size != vals[i].size:
                probs[i] = np.ones(vals[i].size, dtype=float)
            s = probs[i].sum()
            probs[i] = probs[i] / (s if s > 0 else 1.0)

    if probs is not None:
        setattr(obj, probs_name, probs)


# NOTE: your EnsembleModel forward is: refined = ensemble(base_logits, batch_tokens)
#       your OrderBatchModel has .sample_next(prefix) which returns (B,1)


@torch.no_grad()
def _sample_next64(
    ob_model,
    past1024: torch.Tensor,  # (1, 1024) long
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = 200,
) -> torch.Tensor:
    """Autoregressively sample the next 64 batch-tokens given the past 1024 tokens."""
    prefix = past1024
    out = []
    for _ in range(64):
        tok = ob_model.sample_next(prefix, temperature=temperature, top_k=top_k)  # (1,1)
        out.append(tok)
        prefix = torch.cat([prefix, tok], dim=1)
    return torch.cat(out, dim=1)  # (1,64)


@torch.no_grad()
def build_sim_trade_infos_with_ensemble(
    messages_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    symbol: str,
    day: str,
    conv: Converters,
    # models
    order_model,  # trained OrderModel
    ob_model,  # trained OrderBatchModel
    ensemble_model,  # trained EnsembleModel
    device: str = "cuda",
    # context / rollout
    K: int = 1024,  # order-model context length (num_max_orders)
    # past1024_tokens: torch.Tensor,           # (1024,) long, last 16 minutes tokens (flattened)
    max_gen_orders: int = 50_000,
    snapshot_level: int = 10,
    time_unit: str = "ns",
) -> Tuple[List[TradeInfo], LobSnapshot]:
    """
    Minimal MarS-like simulation:
      1) warm up exchange + build initial (K,15) context by replaying real messages until MarketHours open
         and then until we have K recent order-vectors from OrderState
      2) roll forward by:
         - sampling next64 batch tokens from OrderBatchModel (once per minute-batch)
         - sampling next order index from OrderModel logits refined by EnsembleModel conditioned on next64
         - turning the sampled index into a LimitOrder via OrderState converters + current LOB
         - submitting to exchange, accumulating TradeInfos, and updating context from OrderState vectors

    Returns: (sim_trade_infos, start_lob_snapshot)
    """
    # --- exchange + state (keep same scaffold as your preprocessing) ---
    # day = pd.to_datetime(messages_df["Time"].iloc[0], unit="ns").strftime("%Y-%m-%d")

    ex, order_state, base_time = make_exchange_and_orderstate(symbol, day, conv)

    print("base_time")
    print(base_time)  # 2025-11-28 09:30:00
    print(type(base_time))  # <class 'pandas._libs.tslibs.timestamps.Timestamp'>

    order_state.num_max_orders = int(K)  # override the SEQ_LEN from messages_to_features.py

    order_model = order_model.to(device).eval()
    # ob_model = ob_model.to(device).eval()
    # ensemble_model = ensemble_model.to(device).eval()

    # --- align System_Event_Code to each message row (same logic as replay) ---
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

    # --- warmup: replay until open, then fill context deque with K vectors ---
    ctx: deque[np.ndarray] = deque(maxlen=K)  # each element is (15,)
    markethours = False
    start_lob: Optional[LobSnapshot] = None

    for i, r in enumerate(messages_df.itertuples(index=False)):
        if i == 0:
            order_state.open_time = r.Time
        o = row_to_order(r, symbol=symbol, base_time=base_time, time_unit=time_unit, ex=ex)
        if o is None:
            continue

        tis = ex.submit_continuous_auction_order(o) or []
        sys_code = int(msg_sys_code[i])

        if sys_code == 22 and not markethours:
            markethours = True
            start_lob = ex.get_lob(symbol).snapshot(level=snapshot_level)

        if sys_code == 23:
            break
        if not markethours:
            continue

        # OrderState is registered into the exchange; after submit, it gets updated.
        # We take the most recent vector when available.
        if order_state.recent_orders:
            ctx.append(order_state.recent_orders[-1].to_vector())

        if len(ctx) >= K:
            break

    if len(ctx) < K:
        raise RuntimeError(f"Warmup insufficient: got {len(ctx)} context vectors, need K={K}")

    # --- simulation loop ---
    sim_trade_infos: List[TradeInfo] = []
    # past = past1024_tokens.to(device=device, dtype=torch.long).view(1, -1)  # (1,1024)

    # cur_minute_tokens = _sample_next64(ob_model, past)                      # (1,64)
    # past = torch.cat([past[:, 64:], cur_minute_tokens], dim=1)              # roll window

    # (cheap) minute-boundary tracker: resample batch tokens when f4 changes
    last_f4: Optional[int] = None
    next_order_id = 0
    for _ in range(int(max_gen_orders)):
        # build order-model input (1, K, 15)

        X = torch.from_numpy(np.stack(ctx, axis=0)).to(device=device, dtype=torch.long).unsqueeze(0)
        """
        print("XXXXXXXXXX")
        print(X)
        print(X.shape)
        print("X min/max:", X.min().item(), X.max().item())
        """
        X[:, :, 4] = X[:, :, 4].clamp(0, 23399)

        base_logits = order_model(X)[:, -1, :]  # (1,V)
        # refined = ensemble_model(base_logits=base_logits, batch_tokens=cur_minute_tokens)  # (1,V)

        # order_idx = torch.multinomial(torch.softmax(refined, dim=-1), 1).item()
        order_idx = torch.multinomial(torch.softmax(base_logits, dim=-1), 1).item()

        # decode -> order slots using OrderState helper
        pred = order_state.get_pred_order_info(int(order_idx))  # (type, price_slot, vol_slot, interval_slot)

        lob = ex.get_lob(symbol).snapshot(level=10)
        mid = order_state.safe_mid_price(lob)
        if mid is None:
            continue

        # sample actual values from bins (same converters you built)
        px_delta = conv.price_level.sample(pred.price)
        vol = max(1, int(round(conv.pred_order_volume.sample(pred.volume))))
        dt_s = float(conv.order_interval.sample(pred.interval))
        dt_ns = int(round(dt_s * 1e9))

        # advance time from last order time (or open_time)
        prev_t = getattr(order_state.prev_order, "time", None)
        if prev_t is None:
            prev_t = order_state.open_time
        # t_ns = int(pd.Timestamp(prev_t).value) + dt_ns
        t_ns = prev_t.value + dt_ns

        # t = pd.Timestamp(t_ns)
        t = pd.Timedelta(t_ns)

        # --- Cancel (deterministic: cancel oldest order at predicted level) ---
        if pred.order_type == "C":
            lob = ex.get_lob(symbol)

            target_price = int(round(mid + px_delta))
            cancel_side = "B" if target_price <= int(mid) else "S"

            # pick the price level on that side
            levels = lob.bids if cancel_side == "B" else lob.asks
            lvl = next((L for L in levels if int(getattr(L, "price", None)) == target_price), None)
            if lvl is None or not getattr(lvl, "orders", None):
                continue  # no resting orders at that level -> can't cancel

            target = lvl.orders[0]  # FIFO: oldest resting order at that level
            cancel_vol = int(min(max(vol, 1), target.volume))

            o = LimitOrder(
                time=t,
                symbol=symbol,
                tag="sim",
                type="C",
                price=target_price,
                volume=cancel_vol,
                agent_id=-1,
                order_id=-1,
                cancel_type=cancel_side,
                cancel_id=int(target.order_id),
            )

        else:
            price = int(round(mid + px_delta))
            # o = LimitOrder(symbol=symbol, time=t, type=pred.order_type, volume=vol, price=price)

            o = LimitOrder(
                time=t,
                type=pred.order_type,  # "B" or "S"
                price=price,
                volume=vol,
                symbol=symbol,
                agent_id=-1,
                order_id=next_order_id,  # you must define/increment this counter
                cancel_type="",
                cancel_id=-1,
                tag="sim",
            )
            next_order_id += 1

        """
        base_date = pd.Timestamp(day).normalize()
        
        
        # ensure open_time is Timestamp
        if isinstance(order_state.open_time, pd.Timedelta):
            order_state.open_time = pd.Timestamp(base_date) + order_state.open_time
        
        # ensure order time is Timestamp
        if isinstance(o.time, pd.Timedelta):
            o.time = pd.Timestamp(base_date) + o.time
        
        if isinstance(order_state.prev_order.time, pd.Timedelta):
            order_state.prev_order.time = pd.Timestamp(day).normalize() + order_state.prev_order.time
        
        print("BEFORE SUBMITTINGH")
        print(type(o.time))
        print(type(order_state.prev_order.time))
        """

        tis = ex.submit_continuous_auction_order(o) or []

        if not tis:
            continue

        sim_trade_infos.extend(tis)

        #
        # update ctx from state
        if order_state.recent_orders:
            ctx.append(order_state.recent_orders[-1].to_vector())

            """
            # minute boundary heuristic: f4 is time-to-open bucket; when it changes, resample next64
            f4 = int(order_state.recent_orders[-1].time_to_open) if hasattr(order_state.recent_orders[-1], "time_to_open") else None
            if f4 is not None and last_f4 is not None and f4 != last_f4:
                cur_minute_tokens = _sample_next64(ob_model, past)
                past = torch.cat([past[:, 64:], cur_minute_tokens], dim=1)
            last_f4 = f4
            """
    return sim_trade_infos, start_lob


order_model = load_order_model(
    ckpt_path="/scratch/project_2012747/mars_runs/order_model/tb_seq/31991415/step=step=3000-val=val_loss=2.1834.ckpt", device="cuda"
)

ensemble_model = load_ensemble_model(
    ckpt_path="/scratch/project_2012747/mars_runs/ensemble_model/31731449/ckpt_step=0_val=3.307867.pt", order_vocab_size=49152, device="cuda"
)

ob_model = load_order_batch_model("/scratch/project_2012747/mars_runs/order_batch_model/31737330/val=val_loss=1.6047.ckpt", device="cuda")


with open("/scratch/project_2012747/mars_data/order_model/train/intermediate/converters.pkl", "rb") as f:
    converters = pickle.load(f)

messages_df = pd.read_parquet("/scratch/project_2012747/mars_data/order_model/val/raw/AAPL_2025-11-28_messages.parquet")
meta_df = pd.read_parquet("/scratch/project_2012747/mars_data/order_model/val/raw/AAPL_2025-11-28_meta.parquet")

# 64vectors = "/scratch/project_2012747/mars_data/order_batch_model/train/final/AAPL_2025-11-26_64vectors.zarr.zip"


# AD HOC solution to fill in empty bin values, WE SHOULD FIND A BETTER SOLUTION
# apply to all converters inside conv
for name in dir(converters):
    fix_converter(getattr(converters, name))

simu_trade_infos, start_lob = build_sim_trade_infos_with_ensemble(
    messages_df,
    meta_df,
    symbol="AAPL",
    day="2025-11-28",
    conv=converters,
    order_model=order_model,  # trained OrderModel
    ob_model=ob_model,  # trained OrderBatchModel
    ensemble_model=ensemble_model,  # trained EnsembleModel
)


minutes = get_minute_info(simu_trade_infos, start_lob)

pkl_utils.save_pkl_zstd(minutes, Path("my_minutes_simu.zstd"))
