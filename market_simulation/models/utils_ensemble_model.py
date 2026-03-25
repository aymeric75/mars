from __future__ import annotations

import bisect
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from torch.utils.data import BatchSampler, Dataset

from custom_code.preprocessing.order_batch_model.messages_to_order_images import (
    MARKET_CLOSE_NS,
    MARKET_OPEN_NS,
    ONE_MINUTE_NS,
    ORDER_IMAGE_MAX_VALUE,
    chunk_to_order_image,
    from_messages_and_snapshots_to_features,
)
from custom_code.preprocessing.order_model.messages_to_features_no_engine import (
    from_messages_to_features,
)
from market_simulation.models.utils import read_parquet_row_slice


class EnsembleChunkBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: "OnlineEnsembleDataset",
        batch_size: int,
        num_samples: int | None = None,
        chunk_size: int = 256,
        seed: int = 7,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.num_samples = len(dataset) if num_samples is None else int(num_samples)
        self.chunk_size = int(chunk_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)

        self.chunk_groups: list[tuple[int, np.ndarray]] = []
        for file_idx, file_windows in enumerate(self.dataset.window_counts):
            for start in range(0, file_windows, self.chunk_size):
                stop = min(start + self.chunk_size, file_windows)
                local_positions = np.arange(start, stop, dtype=np.int64)
                if local_positions.size > 0:
                    self.chunk_groups.append((file_idx, local_positions))

    def __iter__(self):
        if not self.chunk_groups or self.num_samples <= 0:
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        yielded = 0
        batch: list[int] = []
        chunk_order = torch.randperm(len(self.chunk_groups), generator=generator).tolist()

        for pos, chunk_id in enumerate(chunk_order):
            if yielded >= self.num_samples:
                break

            file_idx, local_positions = self.chunk_groups[chunk_id]
            self.dataset.prefetch_chunk(file_idx, local_positions)

            if pos + 1 < len(chunk_order):
                next_file_idx, next_local_positions = self.chunk_groups[chunk_order[pos + 1]]
                self.dataset.prefetch_chunk(next_file_idx, next_local_positions)

            file_base = 0 if file_idx == 0 else self.dataset.cumulative_windows[file_idx - 1]
            local_order = torch.randperm(len(local_positions), generator=generator).tolist()

            for offset in local_order:
                batch.append(file_base + int(local_positions[offset]))
                yielded += 1

                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

                if yielded >= self.num_samples:
                    break

        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size


class OnlineEnsembleDataset(Dataset):
    """Read raw parquet files and build ensemble samples on the fly.

    Each sample is anchored on one market-hours order index `i` and returns:
    - `context`: rows `[i-seq_len : i]` excluded, shaped `(seq_len, 15)`
    - `target`: `f0[i]`
    - `next_image`: replay next-minute order image starting at the anchor time
    """

    def __init__(
        self,
        message_files: Sequence[str | Path],
        seq_len: int = 1024,
        cache_size: int = 2,
        feature_chunk_size: int = 256,
        sample_chunk_size: int | None = None,
    ):
        self.seq_len = int(seq_len)
        self.cache_size = int(cache_size)
        self.feature_chunk_size = int(feature_chunk_size)
        self.sample_chunk_size = self.feature_chunk_size if sample_chunk_size is None else int(sample_chunk_size)

        self.message_files: list[Path] = []
        self.snapshot_files: list[Path] = []
        self.market_start_rows: list[int] = []
        self.market_row_counts: list[int] = []
        self.window_counts: list[int] = []
        self.cumulative_windows: list[int] = []
        self.feature_cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()
        self.times_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.valid_anchor_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._prefetched_chunks: list[dict[str, object]] = []

        for msg_path in sorted(Path(p) for p in message_files):
            snap_path = self._snapshot_path(msg_path)
            if snap_path.exists():
                self.message_files.append(msg_path)
                self.snapshot_files.append(snap_path)
            else:
                print(f"Skipping {msg_path} (no snapshot file)")

        print("Building ensemble dataset index...")
        total_windows = 0
        for msg_path in self.message_files:
            market_start, market_times = self._read_market_times(msg_path)
            self.market_start_rows.append(market_start)
            self.market_row_counts.append(int(market_times.size))
            n_windows = int(self._compute_valid_anchor_indices(market_times).size)
            self.window_counts.append(n_windows)
            total_windows += n_windows
            self.cumulative_windows.append(total_windows)

        self.total_windows = total_windows
        print(f"Total ensemble samples: {self.total_windows}")

    @staticmethod
    def _snapshot_path(message_path: str | Path) -> Path:
        return Path(str(message_path).replace("messages", "snapshots"))

    @staticmethod
    def _read_market_times(msg_path: Path) -> tuple[int, np.ndarray]:
        times = pd.read_parquet(msg_path, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
        matching = np.flatnonzero((times >= MARKET_OPEN_NS) & (times <= MARKET_CLOSE_NS))
        if matching.size == 0:
            return 0, np.empty(0, dtype=np.int64)

        market_start = int(matching[0])
        market_stop = int(matching[-1] + 1)
        return market_start, times[market_start:market_stop].astype(np.int64, copy=False)

    def _compute_valid_anchor_indices(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=np.int64)
        if times.size <= self.seq_len:
            return np.empty(0, dtype=np.int64)

        anchors = np.arange(self.seq_len, times.size, dtype=np.int64)
        starts = np.searchsorted(times, times[anchors], side="left")
        stops = np.searchsorted(times, times[anchors] + ONE_MINUTE_NS, side="left")
        valid_mask = stops > starts
        return anchors[valid_mask].astype(np.int64, copy=False)

    def _trim_array_cache(self, cache: OrderedDict, max_entries: int) -> None:
        while len(cache) > max(max_entries, 1):
            cache.popitem(last=False)

    def _load_market_times(self, file_idx: int) -> np.ndarray:
        file_idx = int(file_idx)
        if file_idx in self.times_cache:
            self.times_cache.move_to_end(file_idx)
            return self.times_cache[file_idx]

        _, market_times = self._read_market_times(self.message_files[file_idx])
        self.times_cache[file_idx] = market_times
        self.times_cache.move_to_end(file_idx)
        self._trim_array_cache(self.times_cache, self.cache_size)
        return market_times

    def _load_valid_anchor_indices(self, file_idx: int) -> np.ndarray:
        file_idx = int(file_idx)
        if file_idx in self.valid_anchor_cache:
            self.valid_anchor_cache.move_to_end(file_idx)
            return self.valid_anchor_cache[file_idx]

        valid_anchor_indices = self._compute_valid_anchor_indices(self._load_market_times(file_idx))
        self.valid_anchor_cache[file_idx] = valid_anchor_indices
        self.valid_anchor_cache.move_to_end(file_idx)
        self._trim_array_cache(self.valid_anchor_cache, self.cache_size)
        return valid_anchor_indices

    def _locate_index(self, idx: int) -> tuple[int, int]:
        if idx < 0:
            idx += self.total_windows
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_windows}")

        file_idx = bisect.bisect_right(self.cumulative_windows, idx)
        prev_total = 0 if file_idx == 0 else self.cumulative_windows[file_idx - 1]
        local_pos = idx - prev_total
        return file_idx, local_pos

    def _load_feature_chunk(self, file_idx: int, chunk_start: int) -> np.ndarray:
        cache_key = (int(file_idx), int(chunk_start))
        if cache_key in self.feature_cache:
            self.feature_cache.move_to_end(cache_key)
            return self.feature_cache[cache_key]

        market_rows = self.market_row_counts[file_idx]
        if chunk_start < 0 or chunk_start >= market_rows:
            raise IndexError(f"Chunk start {chunk_start} out of range for file {file_idx}")

        include_prev = chunk_start > 0
        raw_start = self.market_start_rows[file_idx] + chunk_start - (1 if include_prev else 0)
        chunk_rows = min(self.feature_chunk_size + self.seq_len, market_rows - chunk_start)
        raw_rows = chunk_rows + (1 if include_prev else 0)

        df = from_messages_to_features(
            self.message_files[file_idx],
            self.snapshot_files[file_idx],
            start_row=raw_start,
            num_rows=raw_rows,
        )

        cols = df.columns.tolist()
        start = cols.index("f0")
        for i in range(start + 1, len(cols)):
            cols[i] = f"f{i - start}"
        df.columns = cols

        feats = df[[f"f{i}" for i in range(15)]].to_numpy(dtype=np.int32, copy=True)
        if include_prev:
            feats = feats[1:]

        self.feature_cache[cache_key] = feats
        self.feature_cache.move_to_end(cache_key)
        if len(self.feature_cache) > max(self.cache_size, 1):
            self.feature_cache.popitem(last=False)

        return feats

    def _context_and_target(self, file_idx: int, local_pos: int) -> tuple[np.ndarray, np.int64]:
        anchor_idx = int(self._load_valid_anchor_indices(file_idx)[local_pos])
        context_start = anchor_idx - self.seq_len
        chunk_start = (context_start // self.feature_chunk_size) * self.feature_chunk_size
        offset = context_start - chunk_start
        feats = self._load_feature_chunk(file_idx, chunk_start)

        context = feats[offset : offset + self.seq_len]
        target = np.int64(feats[offset + self.seq_len, 0])
        return context, target

    def _load_image_feature_slice(self, file_idx: int, start_local_row: int, num_rows: int) -> pd.DataFrame:
        if num_rows <= 0:
            return pd.DataFrame(columns=["Time", "Mars_type", "bin_price", "bin_vol"])

        market_rows = self.market_row_counts[file_idx]
        if start_local_row < 0 or start_local_row >= market_rows:
            raise IndexError(f"Slice start {start_local_row} out of range for file {file_idx}")

        num_rows = min(int(num_rows), market_rows - int(start_local_row))
        raw_start = self.market_start_rows[file_idx] + int(start_local_row)

        msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
        snap_cols = ["Ask_Price_1", "Bid_Price_1"]
        messages = read_parquet_row_slice(
            self.message_files[file_idx],
            columns=msg_cols,
            start_row=raw_start,
            num_rows=num_rows,
        )
        snapshots = read_parquet_row_slice(
            self.snapshot_files[file_idx],
            columns=snap_cols,
            start_row=raw_start,
            num_rows=num_rows,
        )
        return from_messages_and_snapshots_to_features(messages, snapshots)

    def _build_next_images(self, file_idx: int, local_positions: np.ndarray) -> np.ndarray:
        valid_anchor_indices = self._load_valid_anchor_indices(file_idx)
        anchor_indices = valid_anchor_indices[local_positions]
        market_times = self._load_market_times(file_idx)

        starts = np.searchsorted(market_times, market_times[anchor_indices], side="left")
        stops = np.searchsorted(market_times, market_times[anchor_indices] + ONE_MINUTE_NS, side="left")
        slice_start = int(starts.min())
        slice_stop = int(stops.max())

        features = self._load_image_feature_slice(file_idx, slice_start, slice_stop - slice_start)
        images = []
        for start, stop in zip(starts - slice_start, stops - slice_start):
            image = chunk_to_order_image(features.iloc[int(start) : int(stop)])
            images.append(image)

        images_np = np.stack(images, axis=0).astype(np.float32, copy=False)
        images_np /= float(ORDER_IMAGE_MAX_VALUE)
        images_np = images_np * 2.0 - 1.0
        return images_np

    def _build_prefetched_chunk(self, file_idx: int, local_positions: np.ndarray) -> dict[str, object]:
        local_positions = np.asarray(local_positions, dtype=np.int64)
        contexts = np.empty((len(local_positions), self.seq_len, 15), dtype=np.int32)
        targets = np.empty(len(local_positions), dtype=np.int64)

        for row, local_pos in enumerate(local_positions.tolist()):
            context, target = self._context_and_target(file_idx, int(local_pos))
            contexts[row] = context
            targets[row] = target

        next_images = self._build_next_images(file_idx, local_positions)
        row_by_local_idx = {int(local_pos): row for row, local_pos in enumerate(local_positions.tolist())}
        return {
            "file_idx": int(file_idx),
            "local_indices": local_positions.copy(),
            "row_by_local_idx": row_by_local_idx,
            "contexts": contexts,
            "targets": targets,
            "next_images": next_images,
        }

    def prefetch_chunk(self, file_idx: int, local_positions: Sequence[int]) -> None:
        local_positions_np = np.asarray(local_positions, dtype=np.int64)
        if local_positions_np.size == 0:
            return

        for chunk in self._prefetched_chunks:
            if int(chunk["file_idx"]) != int(file_idx):
                continue
            cached = chunk["local_indices"]
            if len(cached) == len(local_positions_np) and np.array_equal(cached, local_positions_np):
                return

        chunk = self._build_prefetched_chunk(int(file_idx), local_positions_np)
        self._prefetched_chunks.append(chunk)
        if len(self._prefetched_chunks) > max(self.cache_size, 1):
            self._prefetched_chunks.pop(0)

    def _prefetch_aligned_chunk(self, file_idx: int, local_pos: int) -> None:
        block_start = (int(local_pos) // self.sample_chunk_size) * self.sample_chunk_size
        block_stop = min(block_start + self.sample_chunk_size, self.window_counts[file_idx])
        self.prefetch_chunk(file_idx, np.arange(block_start, block_stop, dtype=np.int64))

    def _sample_from_prefetched(self, file_idx: int, local_pos: int) -> dict[str, torch.Tensor] | None:
        for chunk in self._prefetched_chunks:
            if int(chunk["file_idx"]) != int(file_idx):
                continue
            row = chunk["row_by_local_idx"].get(int(local_pos))
            if row is None:
                continue

            contexts = chunk["contexts"]
            targets = chunk["targets"]
            next_images = chunk["next_images"]
            return {
                "context": torch.from_numpy(contexts[row].copy()).to(dtype=torch.long),
                "target": torch.tensor(int(targets[row]), dtype=torch.long),
                "next_image": torch.from_numpy(next_images[row].copy()).to(dtype=torch.float32),
            }
        return None

    def __len__(self) -> int:
        return self.total_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, local_pos = self._locate_index(idx)

        cached = self._sample_from_prefetched(file_idx, local_pos)
        if cached is not None:
            return cached

        self._prefetch_aligned_chunk(file_idx, local_pos)
        cached = self._sample_from_prefetched(file_idx, local_pos)
        if cached is not None:
            return cached

        context, target = self._context_and_target(file_idx, local_pos)
        next_image = self._build_next_images(file_idx, np.asarray([local_pos], dtype=np.int64))[0]
        return {
            "context": torch.from_numpy(context.copy()).to(dtype=torch.long),
            "target": torch.tensor(int(target), dtype=torch.long),
            "next_image": torch.from_numpy(next_image.copy()).to(dtype=torch.float32),
        }
