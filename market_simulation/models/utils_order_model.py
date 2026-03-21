import bisect
import glob
import os
import random
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq

import pandas as pd

from pathlib import Path
from collections import OrderedDict
from functools import lru_cache
from torch.utils.data import DataLoader, Dataset, Subset



from custom_code.preprocessing.order_model.messages_to_features_no_engine import (
    from_messages_to_features,
)


def make_train_val_loaders(ds, val_frac=0.01, seed=0, **dl_kwargs):
    n = len(ds)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)

    n_val = int(n * val_frac)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_dl = DataLoader(Subset(ds, train_idx), shuffle=True, **dl_kwargs)
    val_dl = DataLoader(Subset(ds, val_idx), shuffle=False, **dl_kwargs)
    return train_dl, val_dl




class RawMessagesTokenDataset(Dataset):
    """
    Dataset that loads raw messages.parquet files and converts them
    on-the-fly to feature arrays using from_messages_to_features.

    A small LRU cache avoids recomputing features for the same file.
    """

    def __init__(
        self,
        message_files,
        seq_len,
        cache_size=4,
        chunk_size=2048,
    ):
        
                
        self.message_files = []
        
        for msg_path in sorted(message_files):
            snap_path = self._snapshot_path(msg_path)
        
            if snap_path.exists():
                self.message_files.append(msg_path)
            else:
                print(f"Skipping {msg_path} (no snapshot file)")
        
        
        
        self.seq_len = seq_len
        self.cache_size = cache_size
        self.chunk_size = int(chunk_size)

        # LRU cache: (file_idx, chunk_start) -> feature numpy array
        self.cache = OrderedDict()

        # Compact index: keep only window counts per file and cumulative totals.
        # This avoids materializing one Python tuple per possible training window.
        self.window_counts = []
        self.cumulative_windows = []
        self.market_start_rows = []
        self.market_row_counts = []
        total_windows = 0

        print("Building dataset index...")

        for file_idx, msg_path in enumerate(self.message_files):
            market_start, n_rows = self._market_hours_bounds(msg_path)
            print(f"feature rows {n_rows}")
            self.market_start_rows.append(market_start)
            self.market_row_counts.append(n_rows)
            n_windows = max(0, n_rows - seq_len + 1)
            self.window_counts.append(n_windows)
            total_windows += n_windows
            self.cumulative_windows.append(total_windows)

        self.total_windows = total_windows
        print(f"Total windows: {self.total_windows}")

    def _count_rows(self, parquet_path):
        """Fast row count without loading full file."""
        return pq.ParquetFile(parquet_path).metadata.num_rows

    def _market_hours_bounds(self, message_path):
        """Return the raw start row and number of rows inside market hours."""
        start = (9 * 60 * 60 + 30 * 60) * 1_000_000_000
        end = (16 * 60 * 60) * 1_000_000_000
        times = pd.read_parquet(message_path, columns=["Time"])["Time"]
        time_mask = (times >= start) & (times <= end)
        matching = np.flatnonzero(time_mask.to_numpy(copy=False))

        if matching.size == 0:
            return 0, 0

        market_start = int(matching[0])
        market_rows = int(matching[-1] - matching[0] + 1)
        return market_start, market_rows

    def _snapshot_path(self, message_path):
        """
        Infer snapshot path from message path.

        Adjust if naming differs.
        """
        return Path(str(message_path).replace("messages", "snapshots"))

    def _load_feature_chunk(self, file_idx, chunk_start):
        cache_key = (int(file_idx), int(chunk_start))

        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        msg_path = self.message_files[file_idx]
        snap_path = self._snapshot_path(msg_path)
        market_start = self.market_start_rows[file_idx]
        market_rows = self.market_row_counts[file_idx]

        if chunk_start < 0 or chunk_start >= market_rows:
            raise IndexError(f"Chunk start {chunk_start} out of range for file {file_idx}")

        include_prev = chunk_start > 0
        raw_start = market_start + chunk_start - (1 if include_prev else 0)
        chunk_rows = min(self.chunk_size + self.seq_len - 1, market_rows - chunk_start)
        raw_rows = chunk_rows + (1 if include_prev else 0)

        df = from_messages_to_features(
            msg_path,
            snap_path,
            start_row=raw_start,
            num_rows=raw_rows,
        )
    
        # rename columns to f1 -> f14
        cols = df.columns.tolist()
        start = cols.index("f0")
        for i in range(start + 1, len(cols)):
            cols[i] = f"f{i - start}"
        df.columns = cols

        feature_cols = [f"f{i}" for i in range(15)]  # f0..f14
        # convert to numpy [N,15]
        feats = df[feature_cols].to_numpy(dtype=np.int32, copy=True)
        if include_prev:
            feats = feats[1:]

        del df

        # add to cache
        self.cache[cache_key] = feats
        self.cache.move_to_end(cache_key)

        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return feats

    def prefetch_chunk(self, file_idx: int, chunk_start: int):
        if self.cache_size <= 0:
            return
        self._load_feature_chunk(file_idx, chunk_start)

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.total_windows
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_windows}")

        file_idx = bisect.bisect_right(self.cumulative_windows, idx)
        prev_total = 0 if file_idx == 0 else self.cumulative_windows[file_idx - 1]
        start = idx - prev_total
        chunk_start = (start // self.chunk_size) * self.chunk_size
        offset = start - chunk_start
        feats = self._load_feature_chunk(file_idx, chunk_start)
        seq = feats[offset : offset + self.seq_len]
        return torch.tensor(seq, dtype=torch.long)









def lm_loss_all_positions(logits: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    targets = X[:, :, 0]  # (B, K)
    logits_s = logits[:, :-1, :]  # (B, K-1, vocab)
    targ_s = targets[:, 1:]  # (B, K-1)
    return F.cross_entropy(
        logits_s.reshape(-1, logits_s.size(-1)),
        targ_s.reshape(-1),
        reduction="mean",
    )


# OrderModel import
# Keep this module free of sys.path hacks; prefer setting PYTHONPATH properly.
try:
    from market_simulation.models.order_model import OrderModel
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Could not import market_simulation. Make sure the MarS repository is on PYTHONPATH "
        "or installed in your environment."
    ) from e




def build_model_from_variant(model_variant: str, K: int):
    """
    base ~ (emb=64, layers=2, heads=4)
    small = (emb=48, layers=1, heads=4)
    """
    if model_variant == "base":
        EMB_DIM, NUM_LAYERS, NUM_HEADS = 64, 2, 4
    elif model_variant == "small":
        EMB_DIM, NUM_LAYERS, NUM_HEADS = 48, 1, 4
    else:
        raise ValueError(f"Unknown model_variant={model_variant}")

    model = OrderModel(
        emb_dim=EMB_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_max_orders=int(K),
    )

    return model, {"emb_dim": EMB_DIM, "num_layers": NUM_LAYERS, "num_heads": NUM_HEADS}
