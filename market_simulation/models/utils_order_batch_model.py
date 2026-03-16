from __future__ import annotations

import lightning.pytorch as pl
import bisect
import torch
import zarr
import numpy as np
import pandas as pd
import sys
import re

from omegaconf import OmegaConf
from functools import lru_cache
from typing import Sequence
from dataclasses import dataclass
from collections import OrderedDict

from pathlib import Path
from numcodecs import blosc  # noqa: F401
from torch.utils.data import Dataset
from zarr.storage import DirectoryStore

from custom_code.preprocessing.order_batch_model.messages_to_order_images import (
    from_messages_and_snapshots_to_features,
    retrieve_chunk_last_16min_from_df,
    chunks_to_order_images,
    compute_valid_anchor_indices,
)



repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "third_party" / "latent_diffusion"))
sys.path.insert(0, str(repo_root / "third_party" / "taming-transformers"))

from ldm.util import instantiate_from_config





@dataclass
class VQRuntimeConfig:
    ckpt_dir: str
    latent_diffusion_root: str
    taming_root: str
    config_relpath: str = "latent_diffusion/models/first_stage_models/vq-f4/config.yaml"
    use_autocast: bool = True


class _VQForOrders(torch.nn.Module):
    def __init__(self, vqmodel: torch.nn.Module):
        super().__init__()
        self.m = vqmodel


class OnlineMessageTokenDataset(Dataset):
    def __init__(
        self,
        message_files: Sequence[str | Path],
        cache_size: int = 2,
        vq_runtime: VQRuntimeConfig | None = None,
    ):
        self.message_files: list[Path] = []
        self.snapshot_files: list[Path] = []
        self.cache_size = int(cache_size)
        self.cache: OrderedDict[Path, pd.DataFrame] = OrderedDict()
        self.index: list[tuple[int, int]] = []
        self.vq_runtime = vq_runtime
        self._vq_model: torch.nn.Module | None = None
        self._vq_device: torch.device | None = None

        for msg_path in sorted(Path(p) for p in message_files):
            snap_path = self._snapshot_path(msg_path)
            if snap_path.exists():
                self.message_files.append(msg_path)
                self.snapshot_files.append(snap_path)
            else:
                print(f"Skipping {msg_path} (no snapshot file)")

        print("Building order-batch dataset index...")
        for file_idx, msg_path in enumerate(self.message_files):
            times = self._read_times(msg_path)
            valid_anchors = compute_valid_anchor_indices(times)
            for anchor_idx in valid_anchors:
                self.index.append((file_idx, int(anchor_idx)))
        print(f"Total windows: {len(self.index)}")

    def _snapshot_path(self, message_path: str | Path) -> Path:
        return Path(str(message_path).replace("messages", "snapshots"))

    def _read_times(self, msg_path: Path) -> np.ndarray:
        times = pd.read_parquet(msg_path, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
        return times

    def _load_features(self, file_idx: int) -> pd.DataFrame:
        msg_path = self.message_files[file_idx]
        if msg_path in self.cache:
            self.cache.move_to_end(msg_path)
            return self.cache[msg_path]

        msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
        snap_cols = ["Ask_Price_1", "Bid_Price_1"]
        messages = pd.read_parquet(msg_path, columns=msg_cols)
        snapshots = pd.read_parquet(self.snapshot_files[file_idx], columns=snap_cols)
        features = from_messages_and_snapshots_to_features(messages, snapshots)

        self.cache[msg_path] = features
        self.cache.move_to_end(msg_path)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return features

    def _ensure_vq(self) -> tuple[torch.nn.Module, torch.device]:
          
        if self._vq_model is not None and self._vq_device is not None:
            return self._vq_model, self._vq_device
        if self.vq_runtime is None:
            raise ValueError("vq_runtime must be provided for OnlineMessageTokenDataset")

        base = Path(self.vq_runtime.latent_diffusion_root).resolve().parent
        for extra in [
            str(Path(self.vq_runtime.latent_diffusion_root).resolve()),
            str(Path(self.vq_runtime.taming_root).resolve()),
        ]:
            if extra not in sys.path:
                sys.path.insert(0, extra)


        cfg = OmegaConf.load(base / self.vq_runtime.config_relpath)
        vq = instantiate_from_config(cfg.model)
        best_ckpt = self._find_best_checkpoint(Path(self.vq_runtime.ckpt_dir))
        model = _VQForOrders(vq)
        ckpt = torch.load(best_ckpt, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._vq_model = model
        self._vq_device = device
        return model, device

    def _find_best_checkpoint(self, ckpt_dir: Path) -> Path:
        ckpts = list(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

        def extract_val_loss(p: Path) -> float:
            m = re.search(r"val_loss=([0-9]+(?:\.[0-9]+)?)", p.name)
            return float(m.group(1)) if m else float("inf")

        return min(ckpts, key=extract_val_loss)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, anchor_idx = self.index[idx]
        features = self._load_features(file_idx)
        chunks = retrieve_chunk_last_16min_from_df(features, anchor_idx)
        images_np = chunks_to_order_images(chunks)
        images = torch.from_numpy(images_np).float().div_(100.0)
        images = images.mul_(2.0).sub_(1.0)

        model, device = self._ensure_vq()
        images = images.to(device)
        with torch.inference_mode():
            if device.type == "cuda" and self.vq_runtime is not None and self.vq_runtime.use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, info = model.m.encode(images)
            else:
                _, _, info = model.m.encode(images)

        tokens = info[2]
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        tokens = tokens.view(images.size(0), -1).reshape(-1).to(dtype=torch.long)
        return tokens.cpu()



def lm_loss_next_token(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )

