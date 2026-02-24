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

from collections import OrderedDict
from functools import lru_cache
from torch.utils.data import DataLoader, Dataset, Subset
from zarr.storage import DirectoryStore


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






class ParquetFeaturesTokenDataset(Dataset):
    """
    Folder of parquet files, each contains columns f0..f14 (and maybe extra columns like Time, i).
    Returns one sample as torch.long of shape (15,).
    """

    def __init__(self, parquet_dir: str, pattern: str = "*_features.parquet", feature_cols: int = 15, seq_len=1024, rowgroup_cache_size: int = 8):
        self.seq_len = int(seq_len)
        self.parquet_dir = str(parquet_dir)
        self.pattern = str(pattern)
        self.feature_cols = int(feature_cols)
        
        self.rowgroup_cache_size = int(rowgroup_cache_size)
        self._rg_cache: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()


        self.paths = sorted(glob.glob(os.path.join(self.parquet_dir, self.pattern)))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"No parquet files found in {self.parquet_dir} with pattern {self.pattern}")

        # Determine lengths per file (requires a parquet engine: pyarrow or fastparquet)
        self.lens = []
        self._rg_cum: list[list[int]] = []
        
        for p in self.paths:
            meta = self._read_parquet_metadata(p)
            self.lens.append(int(meta.num_rows))
        
            rg_sizes = [int(meta.row_group(i).num_rows) for i in range(meta.num_row_groups)]
            cum = []
            s = 0
            for n in rg_sizes:
                s += n
                cum.append(s)
            self._rg_cum.append(cum)

        
        self.win_lens = [max(0, L - self.seq_len + 1) for L in self.lens]
        
        self.win_cum = []
        s = 0
        for W in self.win_lens:
            s += W
            self.win_cum.append(s)

    def __len__(self) -> int:
        return self.win_cum[-1] if self.win_cum else 0


    @staticmethod
    @lru_cache(maxsize=512)
    def _read_parquet_metadata(path: str):
        return pq.read_metadata(path)
        
    
    def _load_row_group(self, path: str, rg: int) -> np.ndarray:
        """
        Load a single row-group from a parquet file.
    
        Reads only the specified row-group `rg` from `path`, restricted to
        the feature columns (f0..f{feature_cols-1}), and returns it as a
        NumPy array of shape (num_rows_in_rowgroup, feature_cols).
    
        Results are cached in a small LRU cache to avoid repeatedly
        decompressing the same row-group during sequential access.
        """
        
        key = (path, int(rg))
        if key in self._rg_cache:
            self._rg_cache.move_to_end(key)
            return self._rg_cache[key]
    
        
        cols = [f"f{k}" for k in range(self.feature_cols)]
    
        pf = pq.ParquetFile(path)
        table = pf.read_row_group(int(rg), columns=cols)
        arr = table.to_pandas()[cols].to_numpy(dtype=np.int64, copy=False)
    
        self._rg_cache[key] = arr
        self._rg_cache.move_to_end(key)
        while len(self._rg_cache) > self.rowgroup_cache_size:
            self._rg_cache.popitem(last=False)
        return arr
    
    def _load_rows_range(self, fi: int, start: int, length: int) -> np.ndarray:
        """Return array of shape (length, feature_cols) from file fi, starting at row 'start'."""
        path = self.paths[fi]
        end = start + length
    
        rg_cum = self._rg_cum[fi]
        rg_start = bisect.bisect_right(rg_cum, start)
        rg_end = bisect.bisect_right(rg_cum, end - 1)
    
        chunks = []
        for rg in range(rg_start, rg_end + 1):
            arr = self._load_row_group(path, rg)  # shape (rg_rows, feature_cols)
    
            rg_prev = 0 if rg == 0 else rg_cum[rg - 1]
            lo = max(0, start - rg_prev)
            hi = min(arr.shape[0], end - rg_prev)
            if lo < hi:
                chunks.append(arr[lo:hi])
    
        if not chunks:
            # should not happen unless length==0 or bounds are wrong
            return np.empty((0, self.feature_cols), dtype=np.int64)
    
        return np.concatenate(chunks, axis=0)


    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # map global window idx -> (file fi, start row j)
        fi = bisect.bisect_right(self.win_cum, idx)
        prev = 0 if fi == 0 else self.win_cum[fi - 1]
        j = idx - prev  # start row in that file
    
        # load window rows: shape (seq_len, feature_cols)
        window = self._load_rows_range(fi, j, self.seq_len)
    
        # safety (should always be exact length if win_lens computed correctly)
        if window.shape[0] != self.seq_len:
            # if you prefer strictness: raise instead
            # raise IndexError(...)
            pad = self.seq_len - window.shape[0]
            if pad > 0:
                window = np.pad(window, ((0, pad), (0, 0)), mode="edge")
    
        # transpose to (feature_cols, seq_len)
        #window = window.T  # (15, 1024)
    
        return torch.from_numpy(window).long()









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
