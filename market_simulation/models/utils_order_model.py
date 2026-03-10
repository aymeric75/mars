import bisect
import glob
import os
import random
import zipfile
import numpy as np
import torch
import torch.nn.functional as F
import zarr
import pyarrow.parquet as pq

import pandas as pd

from collections import OrderedDict
from functools import lru_cache
from torch.utils.data import DataLoader, Dataset, Subset
from zarr.storage import DirectoryStore


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


def unzip_zarr_zips(train_dir="train", pattern="*.zarr.zip"):
    """Unzip *.zarr.zip -> folders (once). Returns list of extracted *.zarr dirs."""
    for zpath in sorted(glob.glob(os.path.join(train_dir, pattern))):
        out_dir = zpath[:-4]  # strip ".zip" -> "... .zarr"
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            continue
        os.makedirs(out_dir, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(out_dir)
    return sorted(glob.glob(os.path.join(train_dir, pattern.replace(".zip", ""))))  # *.zarr dirs







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
    ):
        self.message_files = sorted(message_files)
        self.seq_len = seq_len
        self.cache_size = cache_size

        # LRU cache: file -> feature numpy array
        self.cache = OrderedDict()

        # map global idx -> (file_idx, start_row)
        self.index = []

        print("Building dataset index...")

        print("message_files")
        print(self.message_files)

        for file_idx, msg_path in enumerate(self.message_files):

            n_rows = self._count_rows(msg_path)
            print(f"n_rows {n_rows}")

            n_windows = max(0, n_rows - seq_len + 1)

            for start in range(n_windows):
                self.index.append((file_idx, start))

        print(f"Total windows: {len(self.index)}")

    def _count_rows(self, parquet_path):
        """Fast row count without loading full file."""
        return pq.ParquetFile(parquet_path).metadata.num_rows

    def _snapshot_path(self, message_path):
        """
        Infer snapshot path from message path.

        Adjust if naming differs.
        """
        return message_path.replace("messages", "snapshots")

    def _load_features(self, file_idx):

        msg_path = self.message_files[file_idx]

        # cache hit
        if msg_path in self.cache:
            self.cache.move_to_end(msg_path)
            return self.cache[msg_path]

        snap_path = self._snapshot_path(msg_path)

        df = from_messages_to_features(msg_path, snap_path)

        # convert to numpy [N,15]
        feats = df[[f"f{i}" for i in range(15)]].to_numpy(dtype=np.int64)

        # add to cache
        self.cache[msg_path] = feats
        self.cache.move_to_end(msg_path)

        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return feats

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        file_idx, start = self.index[idx]

        feats = self._load_features(file_idx)

        seq = feats[start : start + self.seq_len]

        return torch.tensor(seq, dtype=torch.long)









class MultiDirZarrOrderDataset(Dataset):
    """Dataset over many Zarr DirectoryStores."""

    def __init__(self, zarr_dirs, seq_len=1024):
        self.seq_len = int(seq_len)
        self.paths = list(zarr_dirs)

        self.lens = []
        for p in self.paths:
            X = zarr.open(DirectoryStore(p), path="X", mode="r")
            self.lens.append(X.shape[0] - self.seq_len - 1)

        self.cum = []
        s = 0
        for L in self.lens:
            s += max(0, L)
            self.cum.append(s)

    def __len__(self):
        return self.cum[-1] if self.cum else 0

    @staticmethod
    @lru_cache(maxsize=16)
    def _open_X(dir_path):
        store = DirectoryStore(dir_path)
        return zarr.open(store=store, path="X", mode="r")

    def __getitem__(self, idx):
        fi = bisect.bisect_right(self.cum, idx)
        prev = 0 if fi == 0 else self.cum[fi - 1]
        j = idx - prev

        X = self._open_X(self.paths[fi])
        x = X[j : j + self.seq_len]  # (seq_len, features)
        return torch.from_numpy(x).long()


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
