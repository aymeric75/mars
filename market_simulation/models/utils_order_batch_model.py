from __future__ import annotations

import bisect
from functools import lru_cache
from typing import Sequence

import torch
import zarr
import numpy as np

from numcodecs import blosc  # noqa: F401
from torch.utils.data import Dataset
from zarr.storage import DirectoryStore




class TokenDataset(Dataset):
    def __init__(self, files):
        self.arrays = [np.load(f, mmap_mode="r") for f in files]
        self.idx = np.cumsum([0] + [a.shape[0] for a in self.arrays])

    def __len__(self):
        return self.idx[-1]

    def __getitem__(self, i):
        j = np.searchsorted(self.idx, i, side="right") - 1
        k = i - self.idx[j]
        x = self.arrays[j][k]          # (16,64)
        return torch.tensor(x.reshape(-1), dtype=torch.long)  # 1024 tokens


"""
class MultiDirZarrTokenDataset(Dataset):
    
    '''
    Many extracted Zarr dirs, each contains an array (N, 16, 64).
    Returns one sample flattened to (1024,) as torch.long.
    '''
    
    def __init__(self, zarr_dirs: Sequence[str], array_path: str = "arr_0"):
        
        self.paths = list(zarr_dirs)
        self.array_path = str(array_path)

        self.lens: list[int] = []
        for p in self.paths:
            A = zarr.open(DirectoryStore(p), path=self.array_path, mode="r")
            self.lens.append(int(A.shape[0]))

        self.cum: list[int] = []
        s = 0
        for L in self.lens:
            s += L
            self.cum.append(s)

    def __len__(self) -> int:
        return self.cum[-1] if self.cum else 0

    @staticmethod
    @lru_cache(maxsize=2)
    def _open_array(dir_path: str, array_path: str):
        return zarr.open(DirectoryStore(dir_path), path=array_path, mode="r")

    def __getitem__(self, idx: int) -> torch.Tensor:
        fi = bisect.bisect_right(self.cum, idx)
        prev = 0 if fi == 0 else self.cum[fi - 1]
        j = idx - prev

        A = self._open_array(self.paths[fi], self.array_path)
        x = A[j].reshape(-1)  # (16,64) -> (1024,)
        return torch.from_numpy(x).long()
"""


def lm_loss_next_token(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
