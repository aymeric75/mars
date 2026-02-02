import bisect
import glob
import os
import random
import zipfile
from functools import lru_cache

import torch
import torch.nn.functional as F
import zarr
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
