from __future__ import annotations

import numpy as np
import torch
import zarr

from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset



def _open_array_or_group(path: str):
    obj = zarr.open(path, mode="r")
    if hasattr(obj, "shape"):   # zarr Array
        return obj
    if "logits" in obj:
        return obj["logits"]
    if "target_f0" in obj:
        return obj["target_f0"]
    if "next1_tokens" in obj:
        return obj["next1_tokens"]
    raise ValueError(f"Could not find expected array in {path}")


def _collect_trios(data_dir: str):
    data_dir = Path(data_dir)

    tokens_files = sorted(data_dir.glob("*_tokens.zarr"))
    trios = []

    for tok_path in tokens_files:
        stem = tok_path.name[:-len("_tokens.zarr")]   # e.g. AMD_2025-11-03
        logits_path = data_dir / f"{stem}_dense-f16.zarr"
        targets_path = data_dir / f"{stem}_targets.zarr"

        if not logits_path.exists() or not targets_path.exists():
            continue

        next64 = _open_array_or_group(str(tok_path))
        logits = _open_array_or_group(str(logits_path))
        targets = _open_array_or_group(str(targets_path))

        n = int(next64.shape[0])
        if int(logits.shape[0]) != n or int(targets.shape[0]) != n:
            raise ValueError(
                f"Length mismatch for {stem}: "
                f"tokens={next64.shape[0]}, logits={logits.shape[0]}, targets={targets.shape[0]}"
            )

        trios.append({
            "stem": stem,
            "tokens_path": str(tok_path),
            "logits_path": str(logits_path),
            "targets_path": str(targets_path),
            "n": n,
        })

    if not trios:
        raise ValueError(f"No valid trio found in {data_dir}")

    return trios

# -------------------------
# Dataset
# -------------------------

class EnsembleTrainDataset(Dataset):
    """
    Reads all *_tokens.zarr / *_dense-f16.zarr / *_targets.zarr from one folder.
    Returns:
      next64 : (64,) int64
      logits : (V,)  float16/float32
      target : ()    int64
    """

    def __init__(self, *, data_dir: str) -> None:
        self.files = _collect_trios(data_dir)

        self.starts = []
        total = 0
        for f in self.files:
            self.starts.append(total)
            total += f["n"]
        self.total_len = total

        self.tokens_arrs = [_open_array_or_group(f["tokens_path"]) for f in self.files]
        self.logits_arrs = [_open_array_or_group(f["logits_path"]) for f in self.files]
        self.targets_arrs = [_open_array_or_group(f["targets_path"]) for f in self.files]

    def __len__(self) -> int:
        return self.total_len

    def _locate(self, i: int):
        if i < 0 or i >= self.total_len:
            raise IndexError(i)

        file_idx = 0
        while file_idx + 1 < len(self.starts) and self.starts[file_idx + 1] <= i:
            file_idx += 1
        local_idx = i - self.starts[file_idx]
        return file_idx, local_idx

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        file_idx, local_idx = self._locate(i)

        next64 = self.tokens_arrs[file_idx][local_idx]
        logits = self.logits_arrs[file_idx][local_idx]
        target = self.targets_arrs[file_idx][local_idx]

        return {
            "next64": torch.from_numpy(np.asarray(next64, dtype=np.int64)),
            "logits": torch.from_numpy(np.asarray(logits)),
            "target": torch.tensor(int(target), dtype=torch.long),
        }


def ensemble_training_step(*, model, batch: dict[str, Tensor]) -> Tensor:
    next64 = batch["next64"]      # (B, 64)
    base_logits = batch["logits"] # (B, V)
    target = batch["target"]      # (B,)

    refined_logits = model(
        base_logits=base_logits,
        batch_tokens=next64,
    )

    return torch.nn.functional.cross_entropy(refined_logits, target)
