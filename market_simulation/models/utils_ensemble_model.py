from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import zarr


# -------------------------
# Dataset + DataLoader
# -------------------------

class EnsembleTrainDataset(Dataset):
    """
    Returns dict with:
      next64      : (64,) int64
      topk_idx    : (64,) int64
      topk_logit  : (64,) float32
      target      : ()   int64
    """

    def __init__(
        self,
        *,
        next64_path: str,
        topk_idx_path: str,
        topk_logit_path: str,
        targets_path: str,
    ) -> None:
        self.next64 = zarr.open(next64_path, mode="r")         # (N, 64) int32
        self.topk_idx = zarr.open(topk_idx_path, mode="r")     # (N, 64) int32
        self.topk_logit = zarr.open(topk_logit_path, mode="r") # (N, 64) float16/float32
        self.targets = zarr.open(targets_path, mode="r")       # (N,) int32

        n = int(self.next64.shape[0])
        if int(self.topk_idx.shape[0]) != n or int(self.topk_logit.shape[0]) != n or int(self.targets.shape[0]) != n:
            raise ValueError(
                f"Length mismatch: next64={self.next64.shape[0]}, topk_idx={self.topk_idx.shape[0]}, "
                f"topk_logit={self.topk_logit.shape[0]}, targets={self.targets.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.next64.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        next64 = np.asarray(self.next64[i], dtype=np.int64)                 # (64,)
        topk_idx = np.asarray(self.topk_idx[i], dtype=np.int64)             # (64,)
        topk_logit = np.asarray(self.topk_logit[i], dtype=np.float32)       # (64,)
        target = np.int64(self.targets[i])                                  # ()

        return {
            "next64": torch.from_numpy(next64),
            "topk_idx": torch.from_numpy(topk_idx),
            "topk_logit": torch.from_numpy(topk_logit),
            "target": torch.tensor(target, dtype=torch.long),
        }


def make_ensemble_train_loader(
    *,
    next64_path: str,
    topk_idx_path: str,
    topk_logit_path: str,
    targets_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    ds = EnsembleTrainDataset(
        next64_path=next64_path,
        topk_idx_path=topk_idx_path,
        topk_logit_path=topk_logit_path,
        targets_path=targets_path,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# -------------------------
# Top-k -> dense logits
# -------------------------

def topk_to_dense_logits(
    topk_idx: Tensor,        # (B, K) long
    topk_logit: Tensor,      # (B, K) float
    vocab_size: int,         # e.g. 49152
    *,
    fill_value: float = -1e9
) -> Tensor:
    """
    Builds dense base_logits (B, V) from top-k representation.
    Missing logits are set to fill_value (so they are effectively impossible).
    """
    if topk_idx.dim() != 2 or topk_logit.dim() != 2:
        raise ValueError("topk_idx and topk_logit must be (B, K)")
    if topk_idx.shape != topk_logit.shape:
        raise ValueError(f"shape mismatch: {tuple(topk_idx.shape)} vs {tuple(topk_logit.shape)}")

    B, _K = topk_idx.shape
    base = torch.full((B, vocab_size), fill_value, device=topk_idx.device, dtype=topk_logit.dtype)
    base.scatter_(dim=1, index=topk_idx, src=topk_logit)
    return base


# -------------------------
# One training step
# -------------------------

def ensemble_training_step(
    *,
    model,                    # EnsembleModel
    batch: dict[str, Tensor],
    vocab_size: int,           # 49152
) -> Tensor:
    """
    Minimal single training step:
      loss = CE(refined_logits, target)
    """
    next64 = batch["next64"]                  # (B, 64) long
    topk_idx = batch["topk_idx"]              # (B, 64) long
    topk_logit = batch["topk_logit"]          # (B, 64) float
    target = batch["target"]                  # (B,) long

    base_logits = topk_to_dense_logits(topk_idx, topk_logit, vocab_size=vocab_size)  # (B, V)
    refined_logits = model(base_logits=base_logits, batch_tokens=next64)             # (B, V)

    return torch.nn.functional.cross_entropy(refined_logits, target)
