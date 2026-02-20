import numpy as np
import pandas as pd
import torch
from collections import deque
from typing import Optional, Tuple, List

from mlib.core.limit_order import LimitOrder
from mlib.core.trade_info import TradeInfo
from mlib.core.lob_snapshot import LobSnapshot

from custom_code.preprocessing.messages_to_features import (
    Converters,
    make_exchange_and_orderstate,
    row_to_order,
)

# NOTE: your EnsembleModel forward is: refined = ensemble(base_logits, batch_tokens)
#       your OrderBatchModel has .sample_next(prefix) which returns (B,1)


@torch.no_grad()
def _sample_next64(
    ob_model,
    past1024: torch.Tensor,              # (1, 1024) long
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
    conv: Converters,
    base_time: pd.Timestamp,                 # from make_exchange()
    # models
    order_model,                             # trained OrderModel
    ob_model,                                # trained OrderBatchModel
    ensemble_model,                          # trained EnsembleModel
    device: str = "cuda",
    # context / rollout
    K: int = 1024,                           # order-model context length (num_max_orders)
    past1024_tokens: torch.Tensor,           # (1024,) long, last 16 minutes tokens (flattened)
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
    day = pd.to_datetime(messages_df["Time"].iloc[0], unit="ns").strftime("%Y-%m-%d")
    ex, order_state, _ = make_exchange_and_orderstate(symbol, day, conv)
    order_state.num_max_orders = int(K)  # override the SEQ_LEN from messages_to_features.py

    order_model = order_model.to(device).eval()
    ob_model = ob_model.to(device).eval()
    ensemble_model = ensemble_model.to(device).eval()

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
    ctx: deque[np.ndarray] = deque(maxlen=K)     # each element is (15,)
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

    if start_lob is None:
        start_lob = ex.get_lob(symbol).snapshot(level=snapshot_level)
    if len(ctx) < K:
        raise RuntimeError(f"Warmup insufficient: got {len(ctx)} context vectors, need K={K}")

    # --- simulation loop ---
    sim_trade_infos: List[TradeInfo] = []
    past = past1024_tokens.to(device=device, dtype=torch.long).view(1, -1)  # (1,1024)

    cur_minute_tokens = _sample_next64(ob_model, past)                      # (1,64)
    past = torch.cat([past[:, 64:], cur_minute_tokens], dim=1)              # roll window

    # (cheap) minute-boundary tracker: resample batch tokens when f4 changes
    last_f4: Optional[int] = None

    for _ in range(int(max_gen_orders)):
        # build order-model input (1, K, 15)
        X = torch.from_numpy(np.stack(ctx, axis=0)).to(device=device, dtype=torch.long).unsqueeze(0)

        base_logits = order_model(X)[:, -1, :]                               # (1,V)
        refined = ensemble_model(base_logits=base_logits, batch_tokens=cur_minute_tokens)  # (1,V)

        order_idx = torch.multinomial(torch.softmax(refined, dim=-1), 1).item()

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
        t_ns = int(pd.Timestamp(prev_t).value) + dt_ns
        t = pd.Timestamp(t_ns)

        # build LimitOrder (cancel orders need an id present in book; keep it minimal)
        if pred.order_type == "C":
            # pick a live order id if any, else skip
            live_ids = list(ex.get_lob(symbol).order_id_map.keys()) if hasattr(ex.get_lob(symbol), "order_id_map") else []
            if not live_ids:
                continue
            order_id = int(np.random.choice(live_ids))
            o = LimitOrder(symbol=symbol, time=t, type="C", volume=vol, price=int(mid), order_id=order_id)
        else:
            price = int(round(mid + px_delta))
            o = LimitOrder(symbol=symbol, time=t, type=pred.order_type, volume=vol, price=price)

        tis = ex.submit_continuous_auction_order(o) or []
        if not tis:
            continue
        sim_trade_infos.extend(tis)

        # update ctx from state
        if order_state.recent_orders:
            ctx.append(order_state.recent_orders[-1].to_vector())

            # minute boundary heuristic: f4 is time-to-open bucket; when it changes, resample next64
            f4 = int(order_state.recent_orders[-1].time_to_open) if hasattr(order_state.recent_orders[-1], "time_to_open") else None
            if f4 is not None and last_f4 is not None and f4 != last_f4:
                cur_minute_tokens = _sample_next64(ob_model, past)
                past = torch.cat([past[:, 64:], cur_minute_tokens], dim=1)
            last_f4 = f4

    return sim_trade_infos, start_lob
    
    
    
    
    
    

build_sim_trade_infos_with_ensemble(
    messages_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    *,
    symbol: str,
    conv: Converters,
    base_time: pd.Timestamp,                 # from make_exchange()
    # models
    order_model,                             # trained OrderModel
    ob_model,                                # trained OrderBatchModel
    ensemble_model,                          # trained EnsembleModel
    device: str = "cuda",
    # context / rollout
    K: int = 1024,                           # order-model context length (num_max_orders)
    past1024_tokens: torch.Tensor,           # (1024,) long, last 16 minutes tokens (flattened)
    max_gen_orders: int = 50_000,
    snapshot_level: int = 10,
    time_unit: str = "ns",
) -> Tuple[List[TradeInfo], LobSnapshot]:
    
    
    