from __future__ import annotations

import argparse
import sys

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_code.preprocessing.order_batch_model.messages_to_order_images import (
    ONE_MINUTE_NS,
    ORDER_IMAGE_MAX_VALUE,
    chunk_to_order_image,
    compute_valid_anchor_indices,
    from_messages_and_snapshots_to_features as from_messages_and_snapshots_to_image_features,
)
from custom_code.preprocessing.order_model.messages_to_features_no_engine import (
    from_messages_to_features,
)
from custom_code.testing.utils import load_order_batch_model, load_order_model
from custom_code.training.ensemble_model.train_Ensemble_Model_hypersearch import (
    EnsembleLightningModule,
)
from market_simulation.models.utils import read_parquet_row_slice
from market_simulation.models.utils_order_model import lm_loss_all_positions
from market_simulation.models.utils_vqgan import instantiate_vq_model


MARKET_OPEN_NS = (9 * 60 * 60 + 30 * 60) * 1_000_000_000
MARKET_CLOSE_NS = (16 * 60 * 60) * 1_000_000_000


@dataclass
class EnsembleRuntime:
    ensemble_model: torch.nn.Module
    vq_model: torch.nn.Module
    order_batch_model: torch.nn.Module | None
    tokens_per_image: int
    token_source: str
    device: str


@dataclass
class FileMetrics:
    message_file: Path
    stock: str
    day: str
    total_possible_windows: int
    evaluated_windows: int
    base_teacher_forced_loss_sum: float
    base_teacher_forced_token_count: int
    base_next_token_loss_sum: float
    base_next_token_correct: int
    ensemble_next_token_loss_sum: float = 0.0
    ensemble_next_token_correct: int = 0
    ensemble_evaluated_windows: int = 0

    @property
    def base_teacher_forced_loss(self) -> float:
        if self.base_teacher_forced_token_count == 0:
            return float("nan")
        return self.base_teacher_forced_loss_sum / self.base_teacher_forced_token_count

    @property
    def base_next_token_loss(self) -> float:
        if self.evaluated_windows == 0:
            return float("nan")
        return self.base_next_token_loss_sum / self.evaluated_windows

    @property
    def base_next_token_accuracy(self) -> float:
        if self.evaluated_windows == 0:
            return float("nan")
        return self.base_next_token_correct / self.evaluated_windows

    @property
    def has_ensemble(self) -> bool:
        return self.ensemble_evaluated_windows > 0

    @property
    def ensemble_next_token_loss(self) -> float:
        if self.ensemble_evaluated_windows == 0:
            return float("nan")
        return self.ensemble_next_token_loss_sum / self.ensemble_evaluated_windows

    @property
    def ensemble_next_token_accuracy(self) -> float:
        if self.ensemble_evaluated_windows == 0:
            return float("nan")
        return self.ensemble_next_token_correct / self.ensemble_evaluated_windows

    @property
    def ensemble_loss_advantage(self) -> float:
        if not self.has_ensemble:
            return float("nan")
        return self.base_next_token_loss - self.ensemble_next_token_loss

    @property
    def ensemble_accuracy_advantage(self) -> float:
        if not self.has_ensemble:
            return float("nan")
        return self.ensemble_next_token_accuracy - self.base_next_token_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Order Model and, optionally, the Ensemble Model on the exact same sampled windows from data/test with bounded RAM usage."
        ),
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "order_model" / "tensorboard" / "bs=8_lr=1e-4" / "step=step=13920-val=val_loss=3.2903.ckpt",
    )
    parser.add_argument(
        "--ensemble_ckpt",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "ensemble_model" / "tensorboard" / "bs=8_lr=5e-5" / "step=step=1925-val=val_loss=4.6967.ckpt",
        help="Optional Ensemble Lightning .ckpt checkpoint. If omitted, only the Order Model is evaluated.",
    )
    parser.add_argument(
        "--vq_ckpt",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "vqgan" / "2500steps" / "tensorboard" / "bs=8_lr=1e-5" / "step=4606-val_rec_loss=0.038047.ckpt",
    )
    parser.add_argument(
        "--vq_config",
        type=Path,
        default=REPO_ROOT / "third_party" / "latent_diffusion" / "models" / "first_stage_models" / "vq-f4" / "config.yaml",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=REPO_ROOT / "data" / "test",
    )
    parser.add_argument("--stock", type=str, default=None)
    parser.add_argument("--day", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_windows_per_file", type=int, default=256)
    parser.add_argument(
        "--ensemble_token_source",
        choices=["ground_truth", "predicted"],
        default="ground_truth",
        help="Use true next-image VQ tokens or Order-Batch-predicted next-image VQ tokens for the Ensemble.",
    )
    parser.add_argument(
        "--order_batch_ckpt",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "order_batch_model" / "tensorboard" / "bs=2_lr=1e-4" / "step=step=0-val=val_loss=1.8096.ckpt",
        help="Order-Batch Lightning .ckpt used only when --ensemble_token_source predicted.",
    )
    parser.add_argument("--ensemble_tokens_per_image", type=int, default=64)
    parser.add_argument(
        "--all_windows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate every possible next-token window in each selected file.",
    )
    parser.add_argument(
        "--max_rows_per_slice",
        type=int,
        default=4096,
        help="Upper bound on market-hours rows materialized at once per file slice.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to cpu.", flush=True)
        return "cpu"
    return requested


def iter_file_pairs(
    data_dir: Path,
    stock: str | None,
    day: str | None,
    max_files: int | None,
) -> list[tuple[Path, Path, str, str]]:
    pairs: list[tuple[Path, Path, str, str]] = []

    for message_file in sorted(data_dir.glob("*_messages.parquet")):
        snapshot_file = Path(str(message_file).replace("_messages", "_snapshots"))
        if not snapshot_file.exists():
            continue

        parts = message_file.stem.split("_")
        if len(parts) < 3:
            continue
        file_stock = parts[0]
        file_day = parts[1]

        if stock is not None and file_stock != stock:
            continue
        if day is not None and file_day != day:
            continue

        pairs.append((message_file, snapshot_file, file_stock, file_day))

    if max_files is not None:
        pairs = pairs[:max_files]

    return pairs


def read_market_index(message_file: Path) -> tuple[int, np.ndarray]:
    times = pd.read_parquet(message_file, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
    market_rows = np.flatnonzero((times >= MARKET_OPEN_NS) & (times <= MARKET_CLOSE_NS))
    if market_rows.size == 0:
        return 0, np.empty(0, dtype=np.int64)

    market_start = int(market_rows[0])
    market_stop = int(market_rows[-1] + 1)
    market_times = times[market_start:market_stop].astype(np.int64, copy=False)
    return market_start, market_times


def choose_window_starts(
    candidate_starts: np.ndarray,
    all_windows: bool,
    max_windows_per_file: int,
) -> np.ndarray:
    candidate_starts = np.asarray(candidate_starts, dtype=np.int64)
    total_possible_windows = int(candidate_starts.size)
    if total_possible_windows <= 0:
        return np.empty(0, dtype=np.int64)

    if all_windows or total_possible_windows <= max_windows_per_file:
        return candidate_starts.copy()

    select_positions = np.linspace(
        0,
        total_possible_windows - 1,
        num=max_windows_per_file,
        dtype=np.int64,
    )
    return np.unique(candidate_starts[select_positions])


def compute_candidate_starts(
    market_times: np.ndarray,
    seq_len: int,
    require_predicted_history: bool,
) -> np.ndarray:
    market_row_count = int(market_times.size)
    if market_row_count <= seq_len:
        return np.empty(0, dtype=np.int64)

    if not require_predicted_history:
        return np.arange(market_row_count - seq_len, dtype=np.int64)

    valid_anchors = compute_valid_anchor_indices(market_times)
    if valid_anchors.size == 0:
        return np.empty(0, dtype=np.int64)

    valid_anchors = valid_anchors[valid_anchors >= seq_len]
    if valid_anchors.size == 0:
        return np.empty(0, dtype=np.int64)

    return (valid_anchors - seq_len).astype(np.int64, copy=False)


def group_window_starts(
    starts: np.ndarray,
    seq_len: int,
    max_rows_per_slice: int,
    market_times: np.ndarray,
    image_mode: str | None,
) -> list[np.ndarray]:
    if starts.size == 0:
        return []

    anchor_indices = starts + seq_len
    if image_mode == "ground_truth":
        image_starts = anchor_indices
        image_stops = np.searchsorted(
            market_times,
            market_times[anchor_indices] + ONE_MINUTE_NS,
            side="left",
        )
    elif image_mode == "predicted":
        image_starts = np.searchsorted(
            market_times,
            market_times[anchor_indices] - 16 * ONE_MINUTE_NS,
            side="left",
        )
        image_stops = anchor_indices
    else:
        image_starts = anchor_indices
        image_stops = anchor_indices

    groups: list[list[int]] = [[int(starts[0])]]
    current_context_min = int(starts[0])
    current_image_start_min = int(image_starts[0])
    current_image_stop_max = int(image_stops[0])

    for start, image_start, image_stop in zip(starts[1:], image_starts[1:], image_stops[1:]):
        start_int = int(start)
        image_start_int = int(image_start)
        image_stop_int = int(image_stop)

        needed_context_rows = start_int + seq_len + 1 - current_context_min
        needed_image_rows = max(current_image_stop_max, image_stop_int) - min(current_image_start_min, image_start_int)
        needed_rows = max(needed_context_rows, needed_image_rows)

        if needed_rows > max_rows_per_slice:
            groups.append([start_int])
            current_context_min = start_int
            current_image_start_min = image_start_int
            current_image_stop_max = image_stop_int
            continue

        groups[-1].append(start_int)
        current_image_start_min = min(current_image_start_min, image_start_int)
        current_image_stop_max = max(current_image_stop_max, image_stop_int)

    return [np.asarray(group, dtype=np.int64) for group in groups]


def load_feature_slice(
    message_file: Path,
    snapshot_file: Path,
    market_start: int,
    market_slice_start: int,
    market_slice_stop: int,
) -> np.ndarray:
    include_prev = market_slice_start > 0
    raw_start = market_start + market_slice_start - (1 if include_prev else 0)
    raw_rows = (market_slice_stop - market_slice_start) + (1 if include_prev else 0)

    feature_df = from_messages_to_features(
        message_file,
        snapshot_file,
        start_row=raw_start,
        num_rows=raw_rows,
    )

    cols = feature_df.columns.tolist()
    f0_idx = cols.index("f0")
    for idx in range(f0_idx + 1, len(cols)):
        cols[idx] = f"f{idx - f0_idx}"
    feature_df.columns = cols

    feature_array = np.ascontiguousarray(feature_df[[f"f{i}" for i in range(15)]].to_numpy(dtype=np.int64, copy=True))
    if include_prev:
        feature_array = feature_array[1:]
    return feature_array


def load_image_feature_slice(
    message_file: Path,
    snapshot_file: Path,
    market_start: int,
    market_slice_start: int,
    market_slice_stop: int,
) -> pd.DataFrame:
    raw_start = market_start + market_slice_start
    raw_rows = market_slice_stop - market_slice_start

    msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
    snap_cols = ["Ask_Price_1", "Bid_Price_1"]
    messages = read_parquet_row_slice(
        message_file,
        columns=msg_cols,
        start_row=raw_start,
        num_rows=raw_rows,
    )
    snapshots = read_parquet_row_slice(
        snapshot_file,
        columns=snap_cols,
        start_row=raw_start,
        num_rows=raw_rows,
    )
    return from_messages_and_snapshots_to_image_features(messages, snapshots)


def load_ground_truth_next_images(
    image_features: pd.DataFrame,
    image_slice_start: int,
    market_times: np.ndarray,
    anchor_indices: np.ndarray,
) -> np.ndarray:
    next_stops = np.searchsorted(
        market_times,
        market_times[anchor_indices] + ONE_MINUTE_NS,
        side="left",
    )

    images = []
    for anchor_idx, next_stop in zip(anchor_indices.tolist(), next_stops.tolist()):
        local_start = int(anchor_idx) - image_slice_start
        local_stop = int(next_stop) - image_slice_start
        image = chunk_to_order_image(image_features.iloc[local_start:local_stop])
        images.append(image)

    images_np = np.stack(images, axis=0).astype(np.float32, copy=False)
    images_np /= float(ORDER_IMAGE_MAX_VALUE)
    images_np = images_np * 2.0 - 1.0
    return images_np


def build_predicted_prefix_tokens(
    image_features: pd.DataFrame,
    market_times: np.ndarray,
    anchor_indices: np.ndarray,
    vq_model: torch.nn.Module,
    tokens_per_image: int,
    device: str,
) -> torch.Tensor:
    times = image_features["Time"].to_numpy(dtype=np.int64, copy=False)
    context_images = []

    for anchor_idx in anchor_indices.tolist():
        anchor_time = int(market_times[int(anchor_idx)])
        cut_points = np.searchsorted(
            times,
            anchor_time - np.arange(16, -1, -1, dtype=np.int64) * ONE_MINUTE_NS,
            side="left",
        )

        for chunk_id in range(16):
            chunk = image_features.iloc[int(cut_points[chunk_id]) : int(cut_points[chunk_id + 1])]
            context_images.append(chunk_to_order_image(chunk))

    context_images_np = np.stack(context_images, axis=0).astype(np.float32, copy=False)
    context_images_np /= float(ORDER_IMAGE_MAX_VALUE)
    context_images_np = context_images_np * 2.0 - 1.0

    context_tokens = encode_images_to_tokens(
        vq_model=vq_model,
        images_chw=context_images_np,
        device=device,
    )
    if context_tokens.size(1) != tokens_per_image:
        raise RuntimeError(f"Expected {tokens_per_image} VQ tokens per context image, got {context_tokens.size(1)}.")

    batch_size = int(anchor_indices.size)
    return context_tokens.view(batch_size, 16 * tokens_per_image)


def predict_next_image_tokens(
    order_batch_model: torch.nn.Module,
    prefix_tokens: torch.Tensor,
    tokens_per_image: int,
    device: str,
) -> torch.Tensor:
    generated = prefix_tokens.to(device=device, dtype=torch.long, non_blocking=True)
    if generated.ndim == 1:
        generated = generated.unsqueeze(0)

    with torch.inference_mode():
        for _ in range(tokens_per_image):
            next_token = order_batch_model.top_next(generated)
            generated = torch.cat([generated, next_token], dim=1)

    return generated[:, -tokens_per_image:]


def load_next_image_slice(
    message_file: Path,
    snapshot_file: Path,
    market_start: int,
    market_times: np.ndarray,
    anchor_indices: np.ndarray,
) -> np.ndarray:
    next_stops = np.searchsorted(
        market_times,
        market_times[anchor_indices] + ONE_MINUTE_NS,
        side="left",
    )
    slice_start = int(anchor_indices.min())
    slice_stop = int(next_stops.max())
    raw_start = market_start + slice_start
    raw_rows = slice_stop - slice_start

    image_features = load_image_feature_slice(
        message_file=message_file,
        snapshot_file=snapshot_file,
        market_start=market_start,
        market_slice_start=slice_start,
        market_slice_stop=slice_stop,
    )
    return load_ground_truth_next_images(
        image_features=image_features,
        image_slice_start=slice_start,
        market_times=market_times,
        anchor_indices=anchor_indices,
    )


def load_vq_model(ckpt_path: Path, config_path: Path, device: str) -> torch.nn.Module:
    wrapper = instantiate_vq_model(
        config_path=config_path,
        init_ckpt=str(ckpt_path),
        learning_rate=0.0,
    )
    model = wrapper.model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def encode_images_to_tokens(
    vq_model: torch.nn.Module,
    images_chw: np.ndarray,
    device: str,
) -> torch.Tensor:
    inputs = torch.from_numpy(images_chw).to(device=device, dtype=torch.float32, non_blocking=True)
    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, _, info = vq_model.encode(inputs)
        else:
            _, _, info = vq_model.encode(inputs)

    tokens = info[2]
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]
    return tokens.view(inputs.size(0), -1).to(dtype=torch.long)


def maybe_load_ensemble_runtime(args: argparse.Namespace, device: str) -> EnsembleRuntime | None:
    if args.ensemble_ckpt is None:
        return None

    print(f"Loading Ensemble checkpoint from {args.ensemble_ckpt}", flush=True)
    ensemble_lm = EnsembleLightningModule.load_from_checkpoint(
        str(args.ensemble_ckpt),
        map_location="cpu",
        strict=True,
    )
    ensemble_model = ensemble_lm.model.to(device)
    ensemble_model.eval()
    for param in ensemble_model.parameters():
        param.requires_grad_(False)

    order_batch_model = None
    if args.ensemble_token_source == "predicted":
        print(f"Loading Order-Batch checkpoint from {args.order_batch_ckpt}", flush=True)
        order_batch_model = load_order_batch_model(
            ckpt_path=str(args.order_batch_ckpt),
            device=device,
        ).to(device)
        order_batch_model.eval()
        for param in order_batch_model.parameters():
            param.requires_grad_(False)

    print(f"Loading VQ checkpoint from {args.vq_ckpt}", flush=True)
    vq_model = load_vq_model(
        ckpt_path=args.vq_ckpt,
        config_path=args.vq_config,
        device=device,
    )

    return EnsembleRuntime(
        ensemble_model=ensemble_model,
        vq_model=vq_model,
        order_batch_model=order_batch_model,
        tokens_per_image=args.ensemble_tokens_per_image,
        token_source=args.ensemble_token_source,
        device=device,
    )


def evaluate_file(
    message_file: Path,
    snapshot_file: Path,
    stock: str,
    day: str,
    order_model: torch.nn.Module,
    ensemble_runtime: EnsembleRuntime | None,
    device: str,
    seq_len: int,
    batch_size: int,
    max_windows_per_file: int,
    all_windows: bool,
    max_rows_per_slice: int,
) -> FileMetrics | None:
    market_start, market_times = read_market_index(message_file)
    market_row_count = int(market_times.size)
    require_predicted_history = ensemble_runtime is not None and ensemble_runtime.token_source == "predicted"
    candidate_starts = compute_candidate_starts(
        market_times=market_times,
        seq_len=seq_len,
        require_predicted_history=require_predicted_history,
    )
    total_possible_windows = int(candidate_starts.size)
    if total_possible_windows <= 0:
        print(
            f"Skipping {message_file.name}: no valid windows for seq_len={seq_len} and token_source="
            f"{ensemble_runtime.token_source if ensemble_runtime is not None else 'none'}.",
            flush=True,
        )
        return None

    starts = choose_window_starts(
        candidate_starts=candidate_starts,
        all_windows=all_windows,
        max_windows_per_file=max_windows_per_file,
    )
    if starts.size == 0:
        return None

    groups = group_window_starts(
        starts=starts,
        seq_len=seq_len,
        max_rows_per_slice=max_rows_per_slice,
        market_times=market_times,
        image_mode=ensemble_runtime.token_source if ensemble_runtime is not None else None,
    )

    print(
        f"Evaluating {message_file.name}: {len(starts)}/{total_possible_windows} windows across {len(groups)} slice(s).",
        flush=True,
    )

    metrics = FileMetrics(
        message_file=message_file,
        stock=stock,
        day=day,
        total_possible_windows=total_possible_windows,
        evaluated_windows=0,
        base_teacher_forced_loss_sum=0.0,
        base_teacher_forced_token_count=0,
        base_next_token_loss_sum=0.0,
        base_next_token_correct=0,
    )

    with torch.inference_mode():
        for group in groups:
            slice_start = int(group[0])
            slice_stop = int(group[-1] + seq_len + 1)
            feature_array = load_feature_slice(
                message_file=message_file,
                snapshot_file=snapshot_file,
                market_start=market_start,
                market_slice_start=slice_start,
                market_slice_stop=slice_stop,
            )

            expected_rows = slice_stop - slice_start
            if feature_array.shape[0] != expected_rows:
                raise RuntimeError(f"Slice length mismatch for {message_file.name}: expected {expected_rows}, got {feature_array.shape[0]}.")

            local_starts = group - slice_start
            anchor_indices = group + seq_len
            ensemble_batch_tokens = None
            if ensemble_runtime is not None:
                if ensemble_runtime.token_source == "ground_truth":
                    next_images = load_next_image_slice(
                        message_file=message_file,
                        snapshot_file=snapshot_file,
                        market_start=market_start,
                        market_times=market_times,
                        anchor_indices=anchor_indices,
                    )
                    if next_images.shape[0] != len(group):
                        raise RuntimeError(f"Next-image batch mismatch for {message_file.name}: expected {len(group)}, got {next_images.shape[0]}.")
                    ensemble_batch_tokens = encode_images_to_tokens(
                        vq_model=ensemble_runtime.vq_model,
                        images_chw=next_images,
                        device=ensemble_runtime.device,
                    )
                elif ensemble_runtime.token_source == "predicted":
                    if ensemble_runtime.order_batch_model is None:
                        raise RuntimeError("Predicted token mode requires a loaded Order-Batch model.")
                    history_starts = np.searchsorted(
                        market_times,
                        market_times[anchor_indices] - 16 * ONE_MINUTE_NS,
                        side="left",
                    )
                    image_slice_start = int(history_starts.min())
                    image_slice_stop = int(anchor_indices.max())
                    image_features = load_image_feature_slice(
                        message_file=message_file,
                        snapshot_file=snapshot_file,
                        market_start=market_start,
                        market_slice_start=image_slice_start,
                        market_slice_stop=image_slice_stop,
                    )
                    prefix_tokens = build_predicted_prefix_tokens(
                        image_features=image_features,
                        market_times=market_times,
                        anchor_indices=anchor_indices,
                        vq_model=ensemble_runtime.vq_model,
                        tokens_per_image=ensemble_runtime.tokens_per_image,
                        device=ensemble_runtime.device,
                    )
                    ensemble_batch_tokens = predict_next_image_tokens(
                        order_batch_model=ensemble_runtime.order_batch_model,
                        prefix_tokens=prefix_tokens,
                        tokens_per_image=ensemble_runtime.tokens_per_image,
                        device=ensemble_runtime.device,
                    )
                else:
                    raise ValueError(f"Unknown ensemble token source: {ensemble_runtime.token_source}")

                if ensemble_batch_tokens.size(0) != len(group):
                    raise RuntimeError(
                        f"Ensemble token batch mismatch for {message_file.name}: expected {len(group)}, got {ensemble_batch_tokens.size(0)}."
                    )
                if ensemble_batch_tokens.size(1) != ensemble_runtime.tokens_per_image:
                    raise RuntimeError(
                        f"Expected {ensemble_runtime.tokens_per_image} ensemble tokens per image, got {ensemble_batch_tokens.size(1)}."
                    )

            for batch_start in range(0, len(local_starts), batch_size):
                batch_local_starts = local_starts[batch_start : batch_start + batch_size]
                windows = np.stack(
                    [feature_array[int(start) : int(start) + seq_len] for start in batch_local_starts],
                    axis=0,
                )
                next_targets = np.asarray(
                    [feature_array[int(start) + seq_len, 0] for start in batch_local_starts],
                    dtype=np.int64,
                )

                x = torch.from_numpy(windows).to(device=device, dtype=torch.long, non_blocking=True)
                y_next = torch.from_numpy(next_targets).to(device=device, dtype=torch.long, non_blocking=True)

                logits = order_model(x)
                teacher_forced_loss = lm_loss_all_positions(logits, x)
                base_next_logits = logits[:, -1, :]
                base_next_token_loss = F.cross_entropy(base_next_logits, y_next, reduction="sum")
                base_next_pred = torch.argmax(base_next_logits, dim=1)

                batch_count = int(x.size(0))
                metrics.evaluated_windows += batch_count
                metrics.base_teacher_forced_loss_sum += float(teacher_forced_loss.item() * batch_count * (seq_len - 1))
                metrics.base_teacher_forced_token_count += batch_count * (seq_len - 1)
                metrics.base_next_token_loss_sum += float(base_next_token_loss.item())
                metrics.base_next_token_correct += int((base_next_pred == y_next).sum().item())

                if ensemble_runtime is None or ensemble_batch_tokens is None:
                    continue

                next_tokens = ensemble_batch_tokens[batch_start : batch_start + batch_count].to(
                    device=ensemble_runtime.device,
                    dtype=torch.long,
                    non_blocking=True,
                )

                refined_logits = ensemble_runtime.ensemble_model(
                    base_logits=base_next_logits,
                    batch_tokens=next_tokens,
                )
                ensemble_next_token_loss = F.cross_entropy(refined_logits, y_next, reduction="sum")
                ensemble_next_pred = torch.argmax(refined_logits, dim=1)

                metrics.ensemble_evaluated_windows += batch_count
                metrics.ensemble_next_token_loss_sum += float(ensemble_next_token_loss.item())
                metrics.ensemble_next_token_correct += int((ensemble_next_pred == y_next).sum().item())

    return metrics


def print_summary(per_file_metrics: list[FileMetrics]) -> None:
    use_ensemble = any(item.has_ensemble for item in per_file_metrics)

    print("\nPer-file summary:", flush=True)
    for metrics in per_file_metrics:
        line = (
            f"  {metrics.stock} {metrics.day}: "
            f"evaluated={metrics.evaluated_windows}/{metrics.total_possible_windows}, "
            f"base_teacher_forced_loss={metrics.base_teacher_forced_loss:.4f}, "
            f"base_next_loss={metrics.base_next_token_loss:.4f}, "
            f"base_next_acc={100.0 * metrics.base_next_token_accuracy:.2f}%"
        )
        if use_ensemble and metrics.has_ensemble:
            line += (
                f", ensemble_next_loss={metrics.ensemble_next_token_loss:.4f}, "
                f"ensemble_next_acc={100.0 * metrics.ensemble_next_token_accuracy:.2f}%, "
                f"loss_advantage={metrics.ensemble_loss_advantage:.4f}, "
                f"acc_advantage={100.0 * metrics.ensemble_accuracy_advantage:.2f}%"
            )
        print(line, flush=True)

    total_windows = sum(item.evaluated_windows for item in per_file_metrics)
    total_teacher_tokens = sum(item.base_teacher_forced_token_count for item in per_file_metrics)
    total_teacher_loss = sum(item.base_teacher_forced_loss_sum for item in per_file_metrics)
    total_base_next_loss = sum(item.base_next_token_loss_sum for item in per_file_metrics)
    total_base_next_correct = sum(item.base_next_token_correct for item in per_file_metrics)

    overall_teacher = total_teacher_loss / total_teacher_tokens if total_teacher_tokens else float("nan")
    overall_base_next_loss = total_base_next_loss / total_windows if total_windows else float("nan")
    overall_base_next_acc = total_base_next_correct / total_windows if total_windows else float("nan")

    print("\nOverall summary:", flush=True)
    print(f"  files={len(per_file_metrics)}", flush=True)
    print(f"  evaluated_windows={total_windows}", flush=True)
    print(f"  base_teacher_forced_loss={overall_teacher:.4f}", flush=True)
    print(f"  base_next_loss={overall_base_next_loss:.4f}", flush=True)
    print(f"  base_next_acc={100.0 * overall_base_next_acc:.2f}%", flush=True)

    if use_ensemble:
        total_ensemble_windows = sum(item.ensemble_evaluated_windows for item in per_file_metrics)
        total_ensemble_loss = sum(item.ensemble_next_token_loss_sum for item in per_file_metrics)
        total_ensemble_correct = sum(item.ensemble_next_token_correct for item in per_file_metrics)
        overall_ensemble_loss = total_ensemble_loss / total_ensemble_windows if total_ensemble_windows else float("nan")
        overall_ensemble_acc = total_ensemble_correct / total_ensemble_windows if total_ensemble_windows else float("nan")

        print(f"  ensemble_next_loss={overall_ensemble_loss:.4f}", flush=True)
        print(f"  ensemble_next_acc={100.0 * overall_ensemble_acc:.2f}%", flush=True)
        print(f"  ensemble_loss_advantage={overall_base_next_loss - overall_ensemble_loss:.4f}", flush=True)
        print(f"  ensemble_acc_advantage={100.0 * (overall_ensemble_acc - overall_base_next_acc):.2f}%", flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.max_windows_per_file <= 0:
        raise ValueError("--max_windows_per_file must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.seq_len <= 1:
        raise ValueError("--seq_len must be greater than 1")
    if args.ensemble_tokens_per_image <= 0:
        raise ValueError("--ensemble_tokens_per_image must be positive")

    min_rows_per_slice = args.seq_len + 1
    if args.max_rows_per_slice < min_rows_per_slice:
        raise ValueError(f"--max_rows_per_slice must be at least {min_rows_per_slice}")

    file_pairs = iter_file_pairs(
        data_dir=args.data_dir,
        stock=args.stock,
        day=args.day,
        max_files=args.max_files,
    )
    if not file_pairs:
        raise FileNotFoundError(f"No matching message/snapshot pairs found in {args.data_dir}")

    print(f"Loading Order Model checkpoint from {args.ckpt}", flush=True)
    order_model = load_order_model(ckpt_path=str(args.ckpt), device=device, K=args.seq_len).eval()
    ensemble_runtime = maybe_load_ensemble_runtime(args, device)

    per_file_metrics: list[FileMetrics] = []
    for message_file, snapshot_file, stock, day in file_pairs:
        metrics = evaluate_file(
            message_file=message_file,
            snapshot_file=snapshot_file,
            stock=stock,
            day=day,
            order_model=order_model,
            ensemble_runtime=ensemble_runtime,
            device=device,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            max_windows_per_file=args.max_windows_per_file,
            all_windows=args.all_windows,
            max_rows_per_slice=args.max_rows_per_slice,
        )
        if metrics is not None:
            per_file_metrics.append(metrics)

    if not per_file_metrics:
        raise RuntimeError("No files produced evaluable windows.")

    print_summary(per_file_metrics)


if __name__ == "__main__":
    main()
