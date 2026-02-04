from __future__ import annotations

import bisect
from functools import lru_cache
from typing import Sequence

import torch
import zarr
from numcodecs import blosc  # noqa: F401 (often needed in some envs)
from torch.utils.data import Dataset
from zarr.storage import DirectoryStore


class MultiDirZarrTokenDataset(Dataset):
    """Dataset over many Zarr DirectoryStores containing 1D token arrays.

    Assumes each zarr dir contains an array at `array_path` with shape (T,)
    (or (T, 1) which will be squeezed).
    Returns windows of length `seq_len` as torch.long of shape (seq_len,).
    """

    def __init__(self, zarr_dirs: Sequence[str], seq_len: int = 1024, array_path: str = "tokens"):
        self.seq_len = int(seq_len)
        self.array_path = str(array_path)
        self.paths = list(zarr_dirs)

        self.lens = []
        for p in self.paths:
            A = zarr.open(DirectoryStore(p), path=self.array_path, mode="r")
            T = A.shape[0]
            self.lens.append(T - self.seq_len - 1)

        self.cum = []
        s = 0
        for L in self.lens:
            s += max(0, L)
            self.cum.append(s)

    def __len__(self) -> int:
        return self.cum[-1] if self.cum else 0

    @staticmethod
    @lru_cache(maxsize=16)
    def _open_array(dir_path: str, array_path: str):
        return zarr.open(DirectoryStore(dir_path), path=array_path, mode="r")

    def __getitem__(self, idx: int) -> torch.Tensor:
        fi = bisect.bisect_right(self.cum, idx)
        prev = 0 if fi == 0 else self.cum[fi - 1]
        j = idx - prev

        A = self._open_array(self.paths[fi], self.array_path)
        x = A[j : j + self.seq_len]  # (seq_len,) or (seq_len,1)
        x = x.squeeze(-1) if getattr(x, "ndim", 1) == 2 and x.shape[1] == 1 else x
        return torch.from_numpy(x).long()


def lm_loss_next_token(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Next-token CE loss. logits: (B,T,V), input_ids: (B,T)."""
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
