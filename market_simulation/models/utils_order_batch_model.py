from __future__ import annotations

import bisect
from functools import lru_cache
from typing import Sequence

import torch
import zarr
import numpy as np

from pathlib import Path
from numcodecs import blosc  # noqa: F401
from torch.utils.data import Dataset
from zarr.storage import DirectoryStore




class MultiDirZarrTokenDataset(Dataset):
    """
    Many Zarr directories, each contains a root array shaped (N, 16, 64).
    Returns one sample flattened to (1024,) as torch.long.
    """
    def __init__(self, zarr_dirs: Sequence[str]):
        self.paths = list(map(str, zarr_dirs))

        self.lens: list[int] = []
        for p in self.paths:
            A = zarr.open(DirectoryStore(p), mode="r")  # root array
            self.lens.append(int(A.shape[0]))

        self.cum: list[int] = []
        s = 0
        for L in self.lens:
            s += L
            self.cum.append(s)

    def __len__(self) -> int:
        return self.cum[-1] if self.cum else 0

    @staticmethod
    @lru_cache(maxsize=32)  # usually better than 2 when many files
    def _open_array(dir_path: str):
        return zarr.open(DirectoryStore(dir_path), mode="r")

    def __getitem__(self, idx: int) -> torch.Tensor:
        fi = bisect.bisect_right(self.cum, idx)
        prev = 0 if fi == 0 else self.cum[fi - 1]
        j = idx - prev

        A = self._open_array(self.paths[fi])
        x = A[j].reshape(-1)  # (16,64) -> (1024,)
        return torch.from_numpy(np.asarray(x, dtype=np.int64))  # safe dtype



def lm_loss_next_token(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
