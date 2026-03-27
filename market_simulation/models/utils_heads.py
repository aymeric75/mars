from __future__ import annotations

"""Small helpers for supervised heads built on top of MarS.

These functions use the convention we discussed:

- buy entry cost  = fill_buy - mid
- sell entry cost = mid - fill_sell

So:

- cost is measured against the mid price
- slippage is measured against the best quote
"""

import bisect
import re

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from custom_code.preprocessing.order_batch_model.messages_to_order_images import (
    MARKET_CLOSE_NS,
    MARKET_OPEN_NS,
    ONE_MINUTE_NS,
    chunks_to_order_images,
    compute_valid_anchor_indices as compute_valid_order_batch_anchor_indices,
    from_messages_and_snapshots_to_features as from_messages_and_snapshots_to_image_features,
    retrieve_chunk_last_16min_from_df,
)
from custom_code.preprocessing.order_model.messages_to_features_no_engine import (
    from_messages_to_features,
)
from market_simulation.models.utils import read_parquet_row_slice
from market_simulation.models.utils_order_batch_model import VQRuntimeConfig
from market_simulation.models.utils_vqgan import instantiate_vq_model


def mid_price(bid_1: float, ask_1: float) -> float:
    if ask_1 < bid_1:
        raise ValueError(f"ask_1 must be >= bid_1, got bid_1={bid_1}, ask_1={ask_1}")
    return 0.5 * (bid_1 + ask_1)


def average_fill_price(level_prices: Sequence[float], level_sizes: Sequence[float], quantity: float) -> float:
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity}")
    if len(level_prices) != len(level_sizes):
        raise ValueError("level_prices and level_sizes must have the same length")

    remaining = float(quantity)
    total_paid = 0.0

    for price, size in zip(level_prices, level_sizes):
        if size <= 0:
            continue
        traded = min(remaining, float(size))
        total_paid += traded * float(price)
        remaining -= traded
        if remaining <= 0:
            return total_paid / float(quantity)

    raise ValueError("not enough visible depth to fill the order")


def buy_fill_price(ask_prices: Sequence[float], ask_sizes: Sequence[float], quantity: float) -> float:
    return average_fill_price(level_prices=ask_prices, level_sizes=ask_sizes, quantity=quantity)


def sell_fill_price(bid_prices: Sequence[float], bid_sizes: Sequence[float], quantity: float) -> float:
    return average_fill_price(level_prices=bid_prices, level_sizes=bid_sizes, quantity=quantity)


def buy_slippage(fill_buy: float, ask_1: float) -> float:
    return fill_buy - ask_1


def sell_slippage(fill_sell: float, bid_1: float) -> float:
    return bid_1 - fill_sell


def buy_entry_cost(fill_buy: float, bid_1: float, ask_1: float) -> float:
    return fill_buy - mid_price(bid_1=bid_1, ask_1=ask_1)


def sell_entry_cost(fill_sell: float, bid_1: float, ask_1: float) -> float:
    return mid_price(bid_1=bid_1, ask_1=ask_1) - fill_sell


def long_exit_cost(fill_sell: float, bid_1: float, ask_1: float) -> float:
    return sell_entry_cost(fill_sell=fill_sell, bid_1=bid_1, ask_1=ask_1)


def short_exit_cost(fill_buy: float, bid_1: float, ask_1: float) -> float:
    return buy_entry_cost(fill_buy=fill_buy, bid_1=bid_1, ask_1=ask_1)


def buy_entry_cost_from_levels(
    bid_1: float,
    ask_1: float,
    ask_prices: Sequence[float],
    ask_sizes: Sequence[float],
    quantity: float,
) -> float:
    fill_buy = buy_fill_price(ask_prices=ask_prices, ask_sizes=ask_sizes, quantity=quantity)
    return buy_entry_cost(fill_buy=fill_buy, bid_1=bid_1, ask_1=ask_1)


def sell_entry_cost_from_levels(
    bid_1: float,
    ask_1: float,
    bid_prices: Sequence[float],
    bid_sizes: Sequence[float],
    quantity: float,
) -> float:
    fill_sell = sell_fill_price(bid_prices=bid_prices, bid_sizes=bid_sizes, quantity=quantity)
    return sell_entry_cost(fill_sell=fill_sell, bid_1=bid_1, ask_1=ask_1)


def long_exit_cost_from_levels(
    bid_1: float,
    ask_1: float,
    bid_prices: Sequence[float],
    bid_sizes: Sequence[float],
    quantity: float,
) -> float:
    return sell_entry_cost_from_levels(
        bid_1=bid_1,
        ask_1=ask_1,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        quantity=quantity,
    )


def short_exit_cost_from_levels(
    bid_1: float,
    ask_1: float,
    ask_prices: Sequence[float],
    ask_sizes: Sequence[float],
    quantity: float,
) -> float:
    return buy_entry_cost_from_levels(
        bid_1=bid_1,
        ask_1=ask_1,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        quantity=quantity,
    )


def cost_to_return(entry_cost: float, bid_1: float, ask_1: float) -> float:
    mid = mid_price(bid_1=bid_1, ask_1=ask_1)
    if mid <= 0:
        raise ValueError(f"mid price must be positive, got {mid}")
    return entry_cost / mid


def buy_entry_cost_return(fill_buy: float, bid_1: float, ask_1: float) -> float:
    return cost_to_return(
        entry_cost=buy_entry_cost(fill_buy=fill_buy, bid_1=bid_1, ask_1=ask_1),
        bid_1=bid_1,
        ask_1=ask_1,
    )


def sell_entry_cost_return(fill_sell: float, bid_1: float, ask_1: float) -> float:
    return cost_to_return(
        entry_cost=sell_entry_cost(fill_sell=fill_sell, bid_1=bid_1, ask_1=ask_1),
        bid_1=bid_1,
        ask_1=ask_1,
    )


def long_exit_cost_return(fill_sell: float, bid_1: float, ask_1: float) -> float:
    return sell_entry_cost_return(fill_sell=fill_sell, bid_1=bid_1, ask_1=ask_1)


def short_exit_cost_return(fill_buy: float, bid_1: float, ask_1: float) -> float:
    return buy_entry_cost_return(fill_buy=fill_buy, bid_1=bid_1, ask_1=ask_1)


def buy_entry_cost_return_from_levels(
    bid_1: float,
    ask_1: float,
    ask_prices: Sequence[float],
    ask_sizes: Sequence[float],
    quantity: float,
) -> float:
    return cost_to_return(
        entry_cost=buy_entry_cost_from_levels(
            bid_1=bid_1,
            ask_1=ask_1,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            quantity=quantity,
        ),
        bid_1=bid_1,
        ask_1=ask_1,
    )


def sell_entry_cost_return_from_levels(
    bid_1: float,
    ask_1: float,
    bid_prices: Sequence[float],
    bid_sizes: Sequence[float],
    quantity: float,
) -> float:
    return cost_to_return(
        entry_cost=sell_entry_cost_from_levels(
            bid_1=bid_1,
            ask_1=ask_1,
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            quantity=quantity,
        ),
        bid_1=bid_1,
        ask_1=ask_1,
    )


def long_exit_cost_return_from_levels(
    bid_1: float,
    ask_1: float,
    bid_prices: Sequence[float],
    bid_sizes: Sequence[float],
    quantity: float,
) -> float:
    return sell_entry_cost_return_from_levels(
        bid_1=bid_1,
        ask_1=ask_1,
        bid_prices=bid_prices,
        bid_sizes=bid_sizes,
        quantity=quantity,
    )


def short_exit_cost_return_from_levels(
    bid_1: float,
    ask_1: float,
    ask_prices: Sequence[float],
    ask_sizes: Sequence[float],
    quantity: float,
) -> float:
    return buy_entry_cost_return_from_levels(
        bid_1=bid_1,
        ask_1=ask_1,
        ask_prices=ask_prices,
        ask_sizes=ask_sizes,
        quantity=quantity,
    )


class OnlineReturnHeadDataset(Dataset):
    """Anchor-based dataset for supervised heads on top of MarS."""

    def __init__(
        self,
        message_files,
        seq_len: int = 1024,
        scenario: str = "order_model",
        horizon_seconds: int = 30,
        cache_size: int = 2,
        feature_chunk_size: int = 256,
        sample_chunk_size: int | None = None,
        vq_runtime: VQRuntimeConfig | None = None,
    ):
        if scenario not in {"order_model", "order_batch", "both"}:
            raise ValueError(f"Unknown scenario={scenario}")
        if scenario in {"order_batch", "both"} and vq_runtime is None:
            raise ValueError("vq_runtime is required when scenario uses order-batch tokens")

        self.seq_len = int(seq_len)
        self.scenario = scenario
        self.horizon_seconds = int(horizon_seconds)
        self.horizon_ns = self.horizon_seconds * 1_000_000_000
        self.cache_size = int(cache_size)
        self.feature_chunk_size = int(feature_chunk_size)
        self.sample_chunk_size = self.feature_chunk_size if sample_chunk_size is None else int(sample_chunk_size)
        self.vq_runtime = vq_runtime

        self.message_files = []
        self.snapshot_files = []
        for msg_path in sorted(message_files):
            snap_path = self._snapshot_path(msg_path)
            if snap_path.exists():
                self.message_files.append(Path(msg_path))
                self.snapshot_files.append(snap_path)
            else:
                print(f"Skipping {msg_path} (no snapshot file)")

        self.market_start_rows = []
        self.market_row_counts = []
        self.window_counts = []
        self.cumulative_windows = []

        self.feature_cache = OrderedDict()
        self.times_cache = OrderedDict()
        self.return_anchor_cache = OrderedDict()
        self._prefetched_chunks = []
        self._vq_model = None
        self._vq_device = None

        total_windows = 0
        print("Building return-head dataset index...")
        for msg_path in self.message_files:
            market_start, times = self._read_market_times(msg_path)
            self.market_start_rows.append(market_start)
            self.market_row_counts.append(int(times.size))
            n_windows = self._count_valid_return_windows(times)
            self.window_counts.append(n_windows)
            total_windows += n_windows
            self.cumulative_windows.append(total_windows)

        self.total_windows = total_windows
        print(f"Total return-head samples: {self.total_windows}")

    @staticmethod
    def _snapshot_path(message_path):
        return Path(str(message_path).replace("messages", "snapshots"))

    @staticmethod
    def _read_market_times(message_path: Path) -> tuple[int, np.ndarray]:
        times = pd.read_parquet(message_path, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
        matching = np.flatnonzero((times >= MARKET_OPEN_NS) & (times <= MARKET_CLOSE_NS))
        if matching.size == 0:
            return 0, np.empty(0, dtype=np.int64)

        market_start = int(matching[0])
        market_stop = int(matching[-1] + 1)
        return market_start, times[market_start:market_stop].astype(np.int64, copy=False)

    def _count_valid_return_windows(self, times: np.ndarray) -> int:
        times = np.asarray(times, dtype=np.int64)
        if times.size == 0:
            return 0

        min_anchor = self.seq_len if self.scenario in {"order_model", "both"} else 0
        if self.scenario == "order_model":
            anchors = np.arange(min_anchor, times.size, dtype=np.int64)
            future = np.searchsorted(times, times[anchors] + self.horizon_ns, side="left")
            return int(np.count_nonzero(future < times.size))

        anchors = compute_valid_order_batch_anchor_indices(times)
        anchors = anchors[anchors >= min_anchor]
        if anchors.size == 0:
            return 0
        future = np.searchsorted(times, times[anchors] + self.horizon_ns, side="left")
        return int(np.count_nonzero(future < times.size))

    @staticmethod
    def _trim_array_cache(cache: OrderedDict, max_entries: int) -> None:
        while len(cache) > max(max_entries, 1):
            cache.popitem(last=False)

    def _load_market_times(self, file_idx: int) -> np.ndarray:
        file_idx = int(file_idx)
        if file_idx in self.times_cache:
            self.times_cache.move_to_end(file_idx)
            return self.times_cache[file_idx]

        _, market_times = self._read_market_times(self.message_files[file_idx])
        self.times_cache[file_idx] = market_times
        self.times_cache.move_to_end(file_idx)
        self._trim_array_cache(self.times_cache, self.cache_size)
        return market_times

    def _load_return_anchor_indices(self, file_idx: int) -> np.ndarray:
        if self.scenario == "order_model":
            raise RuntimeError("Return-anchor cache is only needed for order-batch scenarios")

        file_idx = int(file_idx)
        if file_idx in self.return_anchor_cache:
            self.return_anchor_cache.move_to_end(file_idx)
            return self.return_anchor_cache[file_idx]

        times = self._load_market_times(file_idx)
        min_anchor = self.seq_len if self.scenario == "both" else 0
        anchors = compute_valid_order_batch_anchor_indices(times)
        anchors = anchors[anchors >= min_anchor]
        if anchors.size > 0:
            future = np.searchsorted(times, times[anchors] + self.horizon_ns, side="left")
            anchors = anchors[future < times.size]

        self.return_anchor_cache[file_idx] = anchors.astype(np.int64, copy=False)
        self.return_anchor_cache.move_to_end(file_idx)
        self._trim_array_cache(self.return_anchor_cache, self.cache_size)
        return self.return_anchor_cache[file_idx]

    def _find_best_checkpoint(self, ckpt_dir: Path) -> Path:
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

        def extract_val_loss(path: Path) -> float:
            match = re.search(r"(?:val_rec_loss|val_loss)=([0-9]+(?:\.[0-9]+)?)", path.name)
            return float(match.group(1)) if match else float("inf")

        return min(ckpts, key=extract_val_loss)

    def _ensure_vq(self):
        if self._vq_model is not None and self._vq_device is not None:
            return self._vq_model, self._vq_device
        if self.vq_runtime is None:
            raise ValueError("vq_runtime is required for order-batch tokens")

        base = Path(self.vq_runtime.latent_diffusion_root).resolve().parent
        best_ckpt = self._find_best_checkpoint(Path(self.vq_runtime.ckpt_dir))
        config_path = base / self.vq_runtime.config_relpath
        model = instantiate_vq_model(
            config_path=config_path,
            init_ckpt=str(best_ckpt),
            learning_rate=0.0,
        ).model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        self._vq_model = model
        self._vq_device = device
        return model, device

    def _locate_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self.total_windows
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_windows}")

        file_idx = bisect.bisect_right(self.cumulative_windows, idx)
        prev_total = 0 if file_idx == 0 else self.cumulative_windows[file_idx - 1]
        local_pos = idx - prev_total
        return file_idx, local_pos

    def _resolve_anchor_and_future_indices(
        self,
        file_idx: int,
        local_positions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_positions = np.asarray(local_positions, dtype=np.int64)
        times = self._load_market_times(file_idx)

        if self.scenario == "order_model":
            anchor_indices = self.seq_len + local_positions
        else:
            valid_anchors = self._load_return_anchor_indices(file_idx)
            anchor_indices = valid_anchors[local_positions]

        future_indices = np.searchsorted(times, times[anchor_indices] + self.horizon_ns, side="left")
        return anchor_indices.astype(np.int64, copy=False), future_indices.astype(np.int64, copy=False)

    def _load_order_feature_chunk(self, file_idx: int, chunk_start: int) -> np.ndarray:
        cache_key = (int(file_idx), int(chunk_start))
        if cache_key in self.feature_cache:
            self.feature_cache.move_to_end(cache_key)
            return self.feature_cache[cache_key]

        msg_path = self.message_files[file_idx]
        snap_path = self.snapshot_files[file_idx]
        market_start = self.market_start_rows[file_idx]
        market_rows = self.market_row_counts[file_idx]

        if chunk_start < 0 or chunk_start >= market_rows:
            raise IndexError(f"Chunk start {chunk_start} out of range for file {file_idx}")

        include_prev = chunk_start > 0
        raw_start = market_start + chunk_start - (1 if include_prev else 0)
        chunk_rows = min(self.feature_chunk_size + self.seq_len - 1, market_rows - chunk_start)
        raw_rows = chunk_rows + (1 if include_prev else 0)

        df = from_messages_to_features(
            msg_path,
            snap_path,
            start_row=raw_start,
            num_rows=raw_rows,
        )
        cols = df.columns.tolist()
        start = cols.index("f0")
        for i in range(start + 1, len(cols)):
            cols[i] = f"f{i - start}"
        df.columns = cols

        feats = df[[f"f{i}" for i in range(15)]].to_numpy(dtype=np.int32, copy=True)
        if include_prev:
            feats = feats[1:]

        self.feature_cache[cache_key] = feats
        self.feature_cache.move_to_end(cache_key)
        if len(self.feature_cache) > max(self.cache_size, 1):
            self.feature_cache.popitem(last=False)
        return feats

    def _build_order_context(self, file_idx: int, anchor_idx: int) -> np.ndarray:
        context_start = anchor_idx - self.seq_len
        chunk_start = (context_start // self.feature_chunk_size) * self.feature_chunk_size
        offset = context_start - chunk_start
        feats = self._load_order_feature_chunk(file_idx, chunk_start)
        return feats[offset : offset + self.seq_len]

    def _load_image_feature_slice(self, file_idx: int, start_local_row: int, num_rows: int) -> pd.DataFrame:
        if num_rows <= 0:
            return pd.DataFrame(columns=["Time", "Mars_type", "bin_price", "bin_vol"])

        market_rows = self.market_row_counts[file_idx]
        if start_local_row < 0 or start_local_row >= market_rows:
            raise IndexError(f"Slice start {start_local_row} out of range for file {file_idx}")

        num_rows = min(int(num_rows), market_rows - int(start_local_row))
        raw_start = self.market_start_rows[file_idx] + int(start_local_row)
        msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
        snap_cols = ["Ask_Price_1", "Bid_Price_1"]
        messages = read_parquet_row_slice(self.message_files[file_idx], columns=msg_cols, start_row=raw_start, num_rows=num_rows)
        snapshots = read_parquet_row_slice(self.snapshot_files[file_idx], columns=snap_cols, start_row=raw_start, num_rows=num_rows)
        return from_messages_and_snapshots_to_image_features(messages, snapshots)

    def _encode_batch_tokens(self, file_idx: int, anchor_indices: np.ndarray) -> torch.Tensor:
        market_times = self._load_market_times(file_idx)
        slice_start = int(
            np.searchsorted(
                market_times,
                market_times[int(anchor_indices.min())] - 16 * ONE_MINUTE_NS,
                side="left",
            )
        )
        slice_stop = int(anchor_indices.max()) + 1
        features = self._load_image_feature_slice(file_idx, slice_start, slice_stop - slice_start)
        relative_anchor_indices = anchor_indices - slice_start

        image_batches = []
        for anchor_idx in relative_anchor_indices:
            chunks = retrieve_chunk_last_16min_from_df(features, int(anchor_idx))
            image_batches.append(chunks_to_order_images(chunks))

        images_np = np.concatenate(image_batches, axis=0)
        images = torch.from_numpy(images_np).float().div_(100.0)
        images = images.mul_(2.0).sub_(1.0)

        model, device = self._ensure_vq()
        images = images.to(device)
        with torch.inference_mode():
            if device.type == "cuda" and self.vq_runtime is not None and self.vq_runtime.use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, info = model.encode(images)
            else:
                _, _, info = model.encode(images)

        tokens = info[2]
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        tokens = tokens.view(images.size(0), -1).to(dtype=torch.long)
        return tokens.reshape(len(local_positions), 16, -1).cpu()

    def _build_return_targets(
        self,
        file_idx: int,
        anchor_indices: np.ndarray,
        future_indices: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        slice_start = int(anchor_indices.min())
        slice_stop = int(future_indices.max()) + 1
        raw_start = self.market_start_rows[file_idx] + slice_start

        snapshots = read_parquet_row_slice(
            self.snapshot_files[file_idx],
            columns=[
                "Ask_Price_1",
                "Ask_Price_2",
                "Ask_Price_3",
                "Ask_Price_4",
                "Ask_Price_5",
                "Bid_Price_1",
                "Bid_Price_2",
                "Bid_Price_3",
                "Bid_Price_4",
                "Bid_Price_5",
                "Ask_Size_1",
                "Ask_Size_2",
                "Ask_Size_3",
                "Ask_Size_4",
                "Ask_Size_5",
                "Bid_Size_1",
                "Bid_Size_2",
                "Bid_Size_3",
                "Bid_Size_4",
                "Bid_Size_5",
            ],
            start_row=raw_start,
            num_rows=slice_stop - slice_start,
        )
        mids = ((snapshots["Ask_Price_1"] + snapshots["Bid_Price_1"]) / 2.0).to_numpy(dtype=np.float64, copy=False)

        current_mid = mids[anchor_indices - slice_start]
        future_mid = mids[future_indices - slice_start]
        target_return = future_mid / current_mid - 1.0
        ask_prices = snapshots[[f"Ask_Price_{i}" for i in range(1, 6)]].to_numpy(dtype=np.float32, copy=False)
        bid_prices = snapshots[[f"Bid_Price_{i}" for i in range(1, 6)]].to_numpy(dtype=np.float32, copy=False)
        ask_sizes = snapshots[[f"Ask_Size_{i}" for i in range(1, 6)]].to_numpy(dtype=np.float32, copy=False)
        bid_sizes = snapshots[[f"Bid_Size_{i}" for i in range(1, 6)]].to_numpy(dtype=np.float32, copy=False)
        anchor_rows = anchor_indices - slice_start
        future_rows = future_indices - slice_start
        return (
            current_mid.astype(np.float32),
            future_mid.astype(np.float32),
            target_return.astype(np.float32),
            ask_prices[anchor_rows].astype(np.float32, copy=False),
            bid_prices[anchor_rows].astype(np.float32, copy=False),
            ask_sizes[anchor_rows].astype(np.float32, copy=False),
            bid_sizes[anchor_rows].astype(np.float32, copy=False),
            ask_prices[future_rows].astype(np.float32, copy=False),
            bid_prices[future_rows].astype(np.float32, copy=False),
            ask_sizes[future_rows].astype(np.float32, copy=False),
            bid_sizes[future_rows].astype(np.float32, copy=False),
        )

    def _build_prefetched_chunk(self, file_idx: int, local_positions: np.ndarray) -> dict[str, object]:
        local_positions = np.asarray(local_positions, dtype=np.int64)
        row_by_local_idx = {int(local_pos): row for row, local_pos in enumerate(local_positions.tolist())}

        chunk: dict[str, object] = {
            "file_idx": int(file_idx),
            "local_indices": local_positions.copy(),
            "row_by_local_idx": row_by_local_idx,
        }

        anchor_indices, future_indices = self._resolve_anchor_and_future_indices(file_idx, local_positions)

        if self.scenario in {"order_model", "both"}:
            contexts = np.empty((len(local_positions), self.seq_len, 15), dtype=np.int32)
            for row, anchor_idx in enumerate(anchor_indices.tolist()):
                contexts[row] = self._build_order_context(file_idx, int(anchor_idx))
            chunk["order_contexts"] = contexts

        if self.scenario in {"order_batch", "both"}:
            chunk["batch_tokens"] = self._encode_batch_tokens(file_idx, anchor_indices)

        (
            current_mid,
            future_mid,
            target_return,
            ask_prices,
            bid_prices,
            ask_sizes,
            bid_sizes,
            future_ask_prices,
            future_bid_prices,
            future_ask_sizes,
            future_bid_sizes,
        ) = self._build_return_targets(
            file_idx,
            anchor_indices,
            future_indices,
        )
        chunk["mid_prices"] = current_mid
        chunk["future_mid_prices"] = future_mid
        chunk["target_returns"] = target_return
        chunk["ask_prices"] = ask_prices
        chunk["bid_prices"] = bid_prices
        chunk["ask_sizes"] = ask_sizes
        chunk["bid_sizes"] = bid_sizes
        chunk["future_ask_prices"] = future_ask_prices
        chunk["future_bid_prices"] = future_bid_prices
        chunk["future_ask_sizes"] = future_ask_sizes
        chunk["future_bid_sizes"] = future_bid_sizes
        return chunk

    def prefetch_chunk(self, file_idx: int, local_positions: np.ndarray) -> None:
        local_positions = np.asarray(local_positions, dtype=np.int64)
        if local_positions.size == 0:
            return

        for chunk in self._prefetched_chunks:
            if int(chunk["file_idx"]) != int(file_idx):
                continue
            cached = chunk["local_indices"]
            if len(cached) == len(local_positions) and np.array_equal(cached, local_positions):
                return

        chunk = self._build_prefetched_chunk(int(file_idx), local_positions)
        self._prefetched_chunks.append(chunk)
        if len(self._prefetched_chunks) > max(self.cache_size, 1):
            self._prefetched_chunks.pop(0)

    def _prefetch_aligned_chunk(self, file_idx: int, local_pos: int) -> None:
        block_start = (int(local_pos) // self.sample_chunk_size) * self.sample_chunk_size
        block_stop = min(block_start + self.sample_chunk_size, self.window_counts[file_idx])
        self.prefetch_chunk(file_idx, np.arange(block_start, block_stop, dtype=np.int64))

    def _sample_from_prefetched(self, file_idx: int, local_pos: int) -> dict[str, torch.Tensor] | None:
        for chunk in self._prefetched_chunks:
            if int(chunk["file_idx"]) != int(file_idx):
                continue
            row = chunk["row_by_local_idx"].get(int(local_pos))
            if row is None:
                continue

            sample = {
                "mid_price": torch.tensor(float(chunk["mid_prices"][row]), dtype=torch.float32),
                "future_mid_price": torch.tensor(float(chunk["future_mid_prices"][row]), dtype=torch.float32),
                "target_return": torch.tensor(float(chunk["target_returns"][row]), dtype=torch.float32),
                "ask_prices": torch.from_numpy(chunk["ask_prices"][row].copy()).to(dtype=torch.float32),
                "bid_prices": torch.from_numpy(chunk["bid_prices"][row].copy()).to(dtype=torch.float32),
                "ask_sizes": torch.from_numpy(chunk["ask_sizes"][row].copy()).to(dtype=torch.float32),
                "bid_sizes": torch.from_numpy(chunk["bid_sizes"][row].copy()).to(dtype=torch.float32),
                "future_ask_prices": torch.from_numpy(chunk["future_ask_prices"][row].copy()).to(dtype=torch.float32),
                "future_bid_prices": torch.from_numpy(chunk["future_bid_prices"][row].copy()).to(dtype=torch.float32),
                "future_ask_sizes": torch.from_numpy(chunk["future_ask_sizes"][row].copy()).to(dtype=torch.float32),
                "future_bid_sizes": torch.from_numpy(chunk["future_bid_sizes"][row].copy()).to(dtype=torch.float32),
            }
            if "order_contexts" in chunk:
                sample["order_context"] = torch.from_numpy(chunk["order_contexts"][row].copy()).to(dtype=torch.long)
            if "batch_tokens" in chunk:
                sample["batch_tokens"] = chunk["batch_tokens"][row].clone().to(dtype=torch.long)
            return sample
        return None

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        file_idx, local_pos = self._locate_index(idx)

        cached = self._sample_from_prefetched(file_idx, local_pos)
        if cached is not None:
            return cached

        self._prefetch_aligned_chunk(file_idx, local_pos)
        cached = self._sample_from_prefetched(file_idx, local_pos)
        if cached is not None:
            return cached

        chunk = self._build_prefetched_chunk(file_idx, np.asarray([local_pos], dtype=np.int64))
        self._prefetched_chunks.append(chunk)
        if len(self._prefetched_chunks) > max(self.cache_size, 1):
            self._prefetched_chunks.pop(0)
        cached = self._sample_from_prefetched(file_idx, local_pos)
        if cached is None:
            raise RuntimeError("Failed to build return-head sample")
        return cached


__all__ = [
    "average_fill_price",
    "buy_entry_cost",
    "buy_entry_cost_from_levels",
    "buy_entry_cost_return",
    "buy_entry_cost_return_from_levels",
    "buy_fill_price",
    "buy_slippage",
    "cost_to_return",
    "long_exit_cost",
    "long_exit_cost_from_levels",
    "long_exit_cost_return",
    "long_exit_cost_return_from_levels",
    "mid_price",
    "OnlineReturnHeadDataset",
    "sell_fill_price",
    "sell_entry_cost",
    "sell_entry_cost_from_levels",
    "sell_entry_cost_return",
    "sell_entry_cost_return_from_levels",
    "sell_slippage",
    "short_exit_cost",
    "short_exit_cost_from_levels",
    "short_exit_cost_return",
    "short_exit_cost_return_from_levels",
]
