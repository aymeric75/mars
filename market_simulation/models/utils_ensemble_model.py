from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
import zarr


# -------------------------
# Dataset
# -------------------------

class EnsembleTrainDataset(Dataset):
    """
    Returns dict with:
      next64 : (64,) int64
      logits : (V,)  float16 / float32
      target : ()    int64
    """

    def __init__(self, *, next64_path: str, logits_path: str, targets_path: str) -> None:
        self.next64 = zarr.open(next64_path, mode="r")  # likely Array (N,64)

        lg = zarr.open_group(logits_path, mode="r")
        self.logits = lg["logits"]                      # Array (N,V)

        tg = zarr.open_group(targets_path, mode="r")
        self.targets = tg["target_f0"]                  # Array (N,)

        n = int(self.next64.shape[0])
        if int(self.logits.shape[0]) != n or int(self.targets.shape[0]) != n:
            raise ValueError(
                f"Length mismatch: next64={self.next64.shape[0]}, logits={self.logits.shape[0]}, targets={self.targets.shape[0]}"
            )

    def __len__(self) -> int:
        return int(self.next64.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "next64": torch.from_numpy(np.asarray(self.next64[i], dtype=np.int64)),
            "logits": torch.from_numpy(np.asarray(self.logits[i])),   # keep dtype
            "target": torch.tensor(int(self.targets[i]), dtype=torch.long),
        }


# -------------------------
# One training step
# -------------------------

def ensemble_training_step(
    *,
    model,                    # EnsembleModel
    batch: dict[str, Tensor],
) -> Tensor:
    """
    loss = CE(refined_logits, target)
    """
    next64 = batch["next64"]            # (B, 64)
    base_logits = batch["logits"]       # (B, V)
    target = batch["target"]            # (B,)

    refined_logits = model(
        base_logits=base_logits,
        batch_tokens=next64,
    )                                   # (B, V)

    return torch.nn.functional.cross_entropy(refined_logits, target)
