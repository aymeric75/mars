from __future__ import annotations

import lightning.pytorch as pl
import bisect
import torch

import numpy as np
import pandas as pd
import re

from typing import Sequence
from dataclasses import dataclass

from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from custom_code.preprocessing.order_batch_model.messages_to_order_images import (
    MARKET_CLOSE_NS,
    MARKET_OPEN_NS,
    ONE_MINUTE_NS,
    from_messages_and_snapshots_to_features,
    retrieve_chunk_last_16min_from_df,
    chunks_to_order_images,
    compute_valid_anchor_indices,
)
from market_simulation.models.utils import read_parquet_row_slice
from market_simulation.models.utils_vqgan import instantiate_vq_model


@dataclass
class VQRuntimeConfig:
    ckpt_dir: str
    latent_diffusion_root: str
    taming_root: str
    config_relpath: str = "latent_diffusion/models/first_stage_models/vq-f4/config.yaml"
    use_autocast: bool = True


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
        self.market_start_rows: list[int] = []
        self.market_row_counts: list[int] = []
        self.market_times: list[np.ndarray] = []
        self.valid_anchor_indices: list[np.ndarray] = []
        self.valid_anchor_times: list[np.ndarray] = []
        self.window_counts: list[int] = []
        self.cumulative_windows: list[int] = []
        self._prefetched_chunks: list[dict[str, object]] = []
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
        total_windows = 0
        for file_idx, msg_path in enumerate(tqdm(self.message_files, desc="Indexing order-batch files")):
            market_start, times = self._read_market_times(msg_path)
            self.market_start_rows.append(market_start)
            self.market_row_counts.append(int(times.size))
            self.market_times.append(times)
            valid_anchors = compute_valid_anchor_indices(times)
            self.valid_anchor_indices.append(valid_anchors.astype(np.int64, copy=False))
            self.valid_anchor_times.append(times[valid_anchors].astype(np.int64, copy=False))
            n_windows = int(valid_anchors.size)
            self.window_counts.append(n_windows)
            total_windows += n_windows
            self.cumulative_windows.append(total_windows)
        self.total_windows = total_windows
        print(f"Total windows: {self.total_windows}")

    def _snapshot_path(self, message_path: str | Path) -> Path:
        return Path(str(message_path).replace("messages", "snapshots"))

    def _read_market_times(self, msg_path: Path) -> tuple[int, np.ndarray]:
        times = pd.read_parquet(msg_path, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
        matching = np.flatnonzero((times >= MARKET_OPEN_NS) & (times <= MARKET_CLOSE_NS))
        if matching.size == 0:
            return 0, np.empty(0, dtype=np.int64)

        market_start = int(matching[0])
        market_stop = int(matching[-1] + 1)
        return market_start, times[market_start:market_stop].astype(np.int64, copy=False)

    def _load_feature_slice(self, file_idx: int, start_local_row: int, num_rows: int) -> pd.DataFrame:
        if num_rows <= 0:
            return pd.DataFrame(columns=["Time", "Mars_type", "bin_price", "bin_vol"])

        market_rows = self.market_row_counts[file_idx]
        if start_local_row < 0 or start_local_row >= market_rows:
            raise IndexError(f"Slice start {start_local_row} out of range for file {file_idx}")

        num_rows = min(int(num_rows), market_rows - int(start_local_row))
        raw_start = self.market_start_rows[file_idx] + int(start_local_row)
        msg_path = self.message_files[file_idx]
        msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
        snap_cols = ["Ask_Price_1", "Bid_Price_1"]
        messages = read_parquet_row_slice(msg_path, columns=msg_cols, start_row=raw_start, num_rows=num_rows)
        snapshots = read_parquet_row_slice(
            self.snapshot_files[file_idx],
            columns=snap_cols,
            start_row=raw_start,
            num_rows=num_rows,
        )
        return from_messages_and_snapshots_to_features(messages, snapshots)

    def _ensure_vq(self) -> tuple[torch.nn.Module, torch.device]:
        if self._vq_model is not None and self._vq_device is not None:
            return self._vq_model, self._vq_device
        if self.vq_runtime is None:
            raise ValueError("vq_runtime must be provided for OnlineMessageTokenDataset")

        base = Path(self.vq_runtime.latent_diffusion_root).resolve().parent
        best_ckpt = self._find_best_checkpoint(Path(self.vq_runtime.ckpt_dir))
        config_path = base / self.vq_runtime.config_relpath
        model = instantiate_vq_model(
            config_path=config_path,
            init_ckpt=str(best_ckpt),
            learning_rate=0.0,
        ).model
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

    def _locate_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self.total_windows
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_windows}")

        file_idx = bisect.bisect_right(self.cumulative_windows, idx)
        prev_total = 0 if file_idx == 0 else self.cumulative_windows[file_idx - 1]
        local_idx = idx - prev_total
        return file_idx, local_idx

    def _encode_anchor_batch(self, file_idx: int, local_indices: Sequence[int]) -> torch.Tensor:
        local_indices_np = np.asarray(local_indices, dtype=np.int64)
        anchor_indices = self.valid_anchor_indices[file_idx][local_indices_np]
        market_times = self.market_times[file_idx]
        slice_start = int(
            np.searchsorted(
                market_times,
                market_times[int(anchor_indices.min())] - 16 * ONE_MINUTE_NS,
                side="left",
            )
        )
        slice_stop = int(anchor_indices.max()) + 1
        features = self._load_feature_slice(file_idx, slice_start, slice_stop - slice_start)
        relative_anchor_indices = anchor_indices - slice_start

        image_batches = []
        for anchor_idx in relative_anchor_indices:
            chunks = retrieve_chunk_last_16min_from_df(features, int(anchor_idx))
            image_batches.append(chunks_to_order_images(chunks))

        images_np = np.concatenate(image_batches, axis=0)
        images = torch.from_numpy(images_np).float().div_(100.0)
        images = images.mul_(2.0).sub_(1.0)

        model, device = self._ensure_vq()
        images = images.to(device)
        with torch.inference_mode():
            if device.type == "cuda" and self.vq_runtime is not None and self.vq_runtime.use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, info = model.encode(images)
            else:
                _, _, info = model.encode(images)

        tokens = info[2]
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        tokens = tokens.view(images.size(0), -1).to(dtype=torch.long)
        return tokens.reshape(len(local_indices), -1).cpu()

    def prefetch_chunk(self, file_idx: int, local_indices: Sequence[int]) -> None:
        local_indices_np = np.asarray(local_indices, dtype=np.int64)
        if local_indices_np.size == 0:
            return

        for chunk in self._prefetched_chunks:
            if int(chunk["file_idx"]) != file_idx:
                continue
            cached = chunk["local_indices"]
            if len(cached) == len(local_indices_np) and np.array_equal(cached, local_indices_np):
                return

        tokens = self._encode_anchor_batch(file_idx, local_indices_np)
        row_by_local_idx = {int(local_idx): row for row, local_idx in enumerate(local_indices_np.tolist())}
        self._prefetched_chunks.append(
            {
                "file_idx": int(file_idx),
                "local_indices": local_indices_np.copy(),
                "row_by_local_idx": row_by_local_idx,
                "tokens": tokens,
            }
        )
        if len(self._prefetched_chunks) > 2:
            self._prefetched_chunks.pop(0)

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, local_idx = self._locate_index(idx)
        for chunk in self._prefetched_chunks:
            if int(chunk["file_idx"]) != file_idx:
                continue
            row = chunk["row_by_local_idx"].get(int(local_idx))
            if row is not None:
                return chunk["tokens"][row]

        return self._encode_anchor_batch(file_idx, [local_idx])[0]


def lm_loss_next_token(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    logits = logits[:, :-1, :].contiguous()
    targets = input_ids[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
    )
