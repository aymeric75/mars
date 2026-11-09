from __future__ import annotations

import argparse
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from omegaconf import OmegaConf
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_code.preprocessing.order_batch_model.messages_to_order_images import (
    MARKET_CLOSE_NS,
    MARKET_OPEN_NS,
    ONE_MINUTE_NS,
    ORDER_IMAGE_MAX_VALUE,
    chunk_to_order_image,
    chunks_to_order_images,
    compute_valid_anchor_indices,
    from_messages_and_snapshots_to_features,
    retrieve_chunk_last_16min_from_df,
)
from custom_code.testing.utils import load_order_batch_model
from market_simulation.models.utils import read_parquet_row_slice
from market_simulation.models.utils_vqgan import instantiate_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize 16 context order-batch images, the predicted next image, and the ground-truth next image.",
    )
    parser.add_argument(
        "--order_batch_ckpt",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "order_batch_model" / "tensorboard" / "bs=2_lr=1e-4" / "step=step=0-val=val_loss=1.8096.ckpt",
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
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument("--file_index", type=int, default=-1)
    parser.add_argument("--anchor_choice", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_vq_model(ckpt_path: Path, config_path: Path, device: str):
    print(f"Loading VQ model config from {config_path}", flush=True)
    cfg = OmegaConf.load(str(config_path))
    cfg.model.params.ckpt_path = None
    cfg.model.params.lossconfig = {"target": "torch.nn.Identity"}

    print("Instantiating VQ model...", flush=True)
    model = instantiate_from_config(cfg.model)
    print(f"Loading VQ checkpoint from {ckpt_path}", flush=True)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items() if key.startswith("model.")}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("loss.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load was not clean. Missing keys: {missing}. Unexpected keys: {unexpected}.",
        )
    model.eval()
    model.to(device)
    print(f"VQ model ready on {device}", flush=True)
    return model


def to_vq_input(images_chw_uint8: np.ndarray, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(images_chw_uint8).float().div_(float(ORDER_IMAGE_MAX_VALUE))
    tensor = tensor.mul_(2.0).sub_(1.0)
    return tensor.to(device)


def image_uint8_from_order_image(image_chw: np.ndarray) -> np.ndarray:
    image = np.asarray(image_chw, dtype=np.float32)
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image / float(ORDER_IMAGE_MAX_VALUE), 0.0, 1.0)
    return np.rint(image * 255.0).astype(np.uint8)


def image_uint8_from_vq_output(image_chw: np.ndarray) -> np.ndarray:
    image = np.asarray(image_chw, dtype=np.float32)
    image = np.transpose(image, (1, 2, 0))
    image = np.clip((image + 1.0) * 0.5, 0.0, 1.0)
    return np.rint(image * 255.0).astype(np.uint8)


def save_rgb_image(image_hwc: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_hwc, mode="RGB").save(out_path)


def save_context_grid(images_hwc: list[np.ndarray], out_path: Path, cols: int = 4) -> None:
    if not images_hwc:
        raise ValueError("images_hwc must not be empty")

    tile_h, tile_w, channels = images_hwc[0].shape
    if channels != 3:
        raise ValueError(f"Expected RGB images, got shape {images_hwc[0].shape}")

    rows = (len(images_hwc) + cols - 1) // cols
    margin = 8
    label_h = 20
    canvas = np.full(
        (
            rows * (tile_h + label_h) + (rows + 1) * margin,
            cols * tile_w + (cols + 1) * margin,
            3,
        ),
        255,
        dtype=np.uint8,
    )
    canvas_image = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(canvas_image)

    for idx, image in enumerate(images_hwc):
        row = idx // cols
        col = idx % cols
        y = margin + row * (tile_h + label_h + margin)
        x = margin + col * (tile_w + margin)
        canvas_image.paste(Image.fromarray(image, mode="RGB"), (x, y))
        draw.text((x, y + tile_h + 2), f"ctx_{idx:02d}", fill=(0, 0, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas_image.save(out_path)


def load_test_sample(
    data_dir: Path,
    file_index: int,
    anchor_choice: int,
    rng: np.random.Generator,
) -> tuple[Path, pd.DataFrame, int]:
    message_files = sorted(data_dir.glob("*messages*.parquet"))
    if not message_files:
        raise FileNotFoundError(f"No *messages*.parquet files found in {data_dir}")
    if file_index < 0:
        file_index = int(rng.integers(0, len(message_files)))
    if file_index >= len(message_files):
        raise IndexError(f"file_index={file_index} out of range for {len(message_files)} files")

    msg_path = message_files[file_index]
    snap_path = Path(str(msg_path).replace("messages", "snapshots"))
    if not snap_path.exists():
        raise FileNotFoundError(f"Missing snapshot file for {msg_path}")

    print(f"Indexing test file {msg_path.name} from Time column only...", flush=True)
    all_times = pd.read_parquet(msg_path, columns=["Time"])["Time"].to_numpy(dtype=np.int64, copy=False)
    market_rows = np.flatnonzero((all_times >= MARKET_OPEN_NS) & (all_times <= MARKET_CLOSE_NS))
    if market_rows.size == 0:
        raise RuntimeError(f"No market-hours rows found in {msg_path.name}")

    market_start = int(market_rows[0])
    market_stop = int(market_rows[-1] + 1)
    market_times = all_times[market_start:market_stop].astype(np.int64, copy=False)
    valid = compute_valid_anchor_indices(market_times)
    if valid.size == 0:
        raise RuntimeError("No valid anchors found in the selected test file.")

    usable: list[int] = []
    for anchor_idx in valid.tolist():
        next_stop = int(
            np.searchsorted(
                market_times,
                market_times[int(anchor_idx)] + ONE_MINUTE_NS,
                side="left",
            )
        )
        if next_stop > int(anchor_idx):
            usable.append(int(anchor_idx))

    if not usable:
        raise RuntimeError("No valid anchors with a non-empty next minute chunk were found.")
    if anchor_choice < 0:
        anchor_choice = int(rng.integers(0, len(usable)))
    if anchor_choice >= len(usable):
        raise IndexError(f"anchor_choice={anchor_choice} out of range for {len(usable)} usable anchors")

    anchor_idx = usable[anchor_choice]
    slice_start = int(
        np.searchsorted(
            market_times,
            market_times[anchor_idx] - 16 * ONE_MINUTE_NS,
            side="left",
        )
    )
    slice_stop = int(
        np.searchsorted(
            market_times,
            market_times[anchor_idx] + ONE_MINUTE_NS,
            side="left",
        )
    )
    slice_stop = max(slice_stop, anchor_idx + 1)

    raw_start = market_start + slice_start
    raw_rows = slice_stop - slice_start
    print(
        f"Loading one sliced parquet window: market_rows[{slice_start}:{slice_stop}] -> raw_rows[{raw_start}:{raw_start + raw_rows}]",
        flush=True,
    )

    msg_cols = ["Time", "Message_Type", "Direction", "Price", "Size"]
    snap_cols = ["Ask_Price_1", "Bid_Price_1"]
    messages = read_parquet_row_slice(
        msg_path,
        columns=msg_cols,
        start_row=raw_start,
        num_rows=raw_rows,
    )
    snapshots = read_parquet_row_slice(
        snap_path,
        columns=snap_cols,
        start_row=raw_start,
        num_rows=raw_rows,
    )
    features = from_messages_and_snapshots_to_features(messages, snapshots)
    return msg_path, features, anchor_idx - slice_start


def next_minute_chunk(features: pd.DataFrame, anchor_idx: int) -> pd.DataFrame:
    times = features["Time"].to_numpy(dtype=np.int64, copy=False)
    anchor_time = int(times[anchor_idx])
    start = int(np.searchsorted(times, anchor_time, side="left"))
    stop = int(np.searchsorted(times, anchor_time + ONE_MINUTE_NS, side="left"))
    return features.iloc[start:stop]


def encode_images_to_tokens(vq_model, images_chw_uint8: np.ndarray, device: str) -> tuple[torch.Tensor, tuple[int, int]]:
    print(f"Encoding {images_chw_uint8.shape[0]} context images with VQ...", flush=True)
    inputs = to_vq_input(images_chw_uint8, device)
    with torch.inference_mode():
        _, _, info = vq_model.encode(inputs)

    tokens = info[2]
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]

    if tokens.ndim == 3:
        grid_hw = (int(tokens.shape[1]), int(tokens.shape[2]))
        flat_tokens = tokens.reshape(tokens.shape[0], -1)
    elif tokens.ndim == 2:
        per_image = int(tokens.shape[1])
        side = int(round(per_image**0.5))
        if side * side != per_image:
            raise ValueError(f"Cannot infer square token grid from {per_image} tokens")
        grid_hw = (side, side)
        flat_tokens = tokens
    elif tokens.ndim == 1:
        batch_size = int(images_chw_uint8.shape[0])
        if tokens.numel() % batch_size != 0:
            raise ValueError(
                f"Flat token vector of length {tokens.numel()} is not divisible by batch size {batch_size}",
            )
        per_image = int(tokens.numel() // batch_size)
        side = int(round(per_image**0.5))
        if side * side != per_image:
            raise ValueError(f"Cannot infer square token grid from {per_image} flat tokens per image")
        grid_hw = (side, side)
        flat_tokens = tokens.view(batch_size, per_image)
    else:
        raise ValueError(f"Unexpected token shape from VQ encoder: {tuple(tokens.shape)}")

    return flat_tokens.to(dtype=torch.long), grid_hw


def decode_tokens_to_image(vq_model, tokens_flat: torch.Tensor, grid_hw: tuple[int, int]) -> np.ndarray:
    h, w = grid_hw
    print(f"Decoding predicted tokens back to image with grid {h}x{w}...", flush=True)
    with torch.inference_mode():
        quant = vq_model.quantize.get_codebook_entry(
            tokens_flat.reshape(-1),
            shape=(1, h, w, int(vq_model.embed_dim)),
        )
        decoded = vq_model.decode(quant)
    return decoded.detach().cpu().squeeze(0).numpy()


def predict_next_image_tokens(order_batch_model, prefix_tokens: torch.Tensor, tokens_per_image: int, device: str) -> torch.Tensor:
    print(
        f"Autoregressively predicting {tokens_per_image} next-image tokens from prefix length {prefix_tokens.numel()}...",
        flush=True,
    )
    generated = prefix_tokens.unsqueeze(0).to(device=device, dtype=torch.long)
    with torch.inference_mode():
        for step in range(tokens_per_image):
            next_token = order_batch_model.top_next(generated)
            generated = torch.cat([generated, next_token], dim=1)
            if (step + 1) % 8 == 0 or step + 1 == tokens_per_image:
                print(f"  predicted {step + 1}/{tokens_per_image} tokens", flush=True)
    return generated[0, -tokens_per_image:].detach().cpu()


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    device = args.device
    print(f"Using device: {device}", flush=True)
    print(f"Loading order-batch model from {args.order_batch_ckpt.resolve()}", flush=True)
    order_batch_model = load_order_batch_model(str(args.order_batch_ckpt.resolve()), device=device).to(device)
    print("Order-batch model ready", flush=True)
    vq_model = load_vq_model(args.vq_ckpt.resolve(), args.vq_config.resolve(), device=device)

    print(f"Loading sliced test sample from {args.data_dir.resolve()}", flush=True)
    msg_path, features, anchor_idx = load_test_sample(
        args.data_dir.resolve(),
        int(args.file_index),
        int(args.anchor_choice),
        rng,
    )
    print(f"Loaded sliced feature frame from {msg_path.name} with {len(features)} rows", flush=True)
    print(f"Selected relative anchor index {anchor_idx}", flush=True)

    print("Building 16 context images and next-minute ground truth image...", flush=True)
    context_chunks = retrieve_chunk_last_16min_from_df(features, anchor_idx)
    context_images = chunks_to_order_images(context_chunks)
    gt_next_chunk = next_minute_chunk(features, anchor_idx)
    gt_next_image = chunk_to_order_image(gt_next_chunk)

    context_tokens, grid_hw = encode_images_to_tokens(vq_model, context_images, device=device)
    tokens_per_image = int(context_tokens.shape[1])
    print(f"Inferred token grid {grid_hw} and {tokens_per_image} tokens per image", flush=True)
    prefix_tokens = context_tokens.reshape(-1)
    predicted_tokens = predict_next_image_tokens(
        order_batch_model=order_batch_model,
        prefix_tokens=prefix_tokens,
        tokens_per_image=tokens_per_image,
        device=device,
    )
    predicted_image = decode_tokens_to_image(vq_model, predicted_tokens.to(device), grid_hw)

    context_dir = output_dir / "context_images"
    context_uint8_images: list[np.ndarray] = []
    print(f"Saving outputs under {output_dir}", flush=True)
    for idx, image in enumerate(context_images):
        image_uint8 = image_uint8_from_order_image(image)
        context_uint8_images.append(image_uint8)
        save_rgb_image(image_uint8, context_dir / f"context_{idx:02d}.png")

    save_context_grid(context_uint8_images, output_dir / "context_grid.png")
    save_rgb_image(image_uint8_from_vq_output(predicted_image), output_dir / "next_predicted.png")
    save_rgb_image(image_uint8_from_order_image(gt_next_image), output_dir / "next_ground_truth.png")

    print(f"Test file: {msg_path.name}", flush=True)
    print(f"Anchor index: {anchor_idx}", flush=True)
    print(f"Tokens per image: {tokens_per_image}", flush=True)
    print(f"Saved outputs under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
