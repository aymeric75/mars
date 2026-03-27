from __future__ import annotations

import argparse
import sys

from pathlib import Path

import numpy as np
import torch

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from market_simulation.models.utils_vqgan import (
    ORDER_IMAGE_MAX_VALUE,
    RawMinuteOrderImageDataset,
    instantiate_vq_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained VQGAN checkpoint, reconstruct 5 order images from data/test, and save them.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "vqgan" / "tensorboard" / "bs=8_lr=1.5e-5" / "step=14584-val_rec_loss=0.031307.ckpt",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=REPO_ROOT / "data" / "test",
    )
    parser.add_argument(
        "--converter_json_path",
        type=Path,
        default=REPO_ROOT / "custom_code" / "training" / "converters_portable.json",
    )
    parser.add_argument(
        "--vq_config",
        type=Path,
        default=REPO_ROOT / "third_party" / "latent_diffusion" / "models" / "first_stage_models" / "vq-f4" / "config.yaml",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible test-sample selection.",
    )
    parser.add_argument("--learning_rate", type=float, default=4.5e-6)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def load_vqgan_checkpoint(
    ckpt_path: Path,
    vq_config: Path,
    learning_rate: float,
    device: str,
):
    model = instantiate_vq_model(
        config_path=vq_config,
        init_ckpt=None,
        learning_rate=learning_rate,
    )

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped_state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
        missing, unexpected = model.model.load_state_dict(stripped_state_dict, strict=True)

    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load was not clean. Missing keys: {missing}. Unexpected keys: {unexpected}.",
        )

    model.eval()
    model.to(device)
    return model


def model_space_to_order_image_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.clip((image + 1.0) * 0.5 * float(ORDER_IMAGE_MAX_VALUE), 0.0, float(ORDER_IMAGE_MAX_VALUE))
    return np.rint(image).astype(np.uint8)


def order_image_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.clip(image / float(ORDER_IMAGE_MAX_VALUE), 0.0, 1.0)
    return np.rint(image * 255.0).astype(np.uint8)


def save_order_image_matrices(image: np.ndarray, out_path: Path, scale: int = 16, gap: int = 24) -> None:
    image = order_image_to_uint8_hwc(image)
    matrix_h, matrix_w, num_channels = image.shape
    labels = ["S", "B", "C"]
    top_pad = 28
    border = 2

    channels: list[Image.Image] = []
    for idx in range(num_channels):
        channel = Image.fromarray(image[:, :, idx], mode="L").resize(
            (matrix_w * scale, matrix_h * scale),
            resample=Image.Resampling.NEAREST,
        ).convert("RGB")
        channel = Image.new("RGB", (channel.width + 2 * border, channel.height + 2 * border), color=(0, 0, 0))
        inner = Image.fromarray(image[:, :, idx], mode="L").resize(
            (matrix_w * scale, matrix_h * scale),
            resample=Image.Resampling.NEAREST,
        ).convert("RGB")
        channel.paste(inner, (border, border))
        channels.append(channel)

    canvas_w = sum(channel.width for channel in channels) + gap * (len(channels) - 1)
    canvas_h = top_pad + max(channel.height for channel in channels)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for idx, channel in enumerate(channels):
        label = labels[idx] if idx < len(labels) else f"C{idx}"
        text_x = x + channel.width // 2 - 4
        draw.text((text_x, 6), label, fill=(0, 0, 0))
        canvas.paste(channel, (x, top_pad))
        x += channel.width + gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_dataset(data_dir: Path, converter_json_path: Path) -> RawMinuteOrderImageDataset:
    message_files = sorted(data_dir.glob("*messages.parquet"))
    if not message_files:
        raise FileNotFoundError(f"No *messages.parquet files found in {data_dir}")

    return RawMinuteOrderImageDataset(
        message_files=message_files,
        converter_json_path=converter_json_path,
        include_empty_minutes=False,
        max_minutes_per_file=None,
        minute_stride=1,
    )


def reconstruct_samples(
    model,
    dataset: RawMinuteOrderImageDataset,
    num_samples: int,
    output_dir: Path,
    device: str,
    seed: int | None = None,
) -> None:
    originals_dir = output_dir / "originals"
    reconstructions_dir = output_dir / "reconstructions"

    total = min(num_samples, len(dataset))
    if total == 0:
        raise RuntimeError("The dataset is empty; nothing to reconstruct.")

    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(dataset), size=total, replace=False)

    with torch.no_grad():
        for sample_num, dataset_idx in enumerate(sampled_indices):
            sample = dataset[int(dataset_idx)]
            image_hwc = sample["image"].astype(np.float32, copy=False)
            batch = {"image": torch.from_numpy(image_hwc).unsqueeze(0).to(device)}
            x = model.model.get_input(batch, model.model.image_key)
            xrec, _, _ = model.model(x, return_pred_indices=True)

            original_order_image = model_space_to_order_image_hwc(image_hwc)
            reconstruction_order_image = model_space_to_order_image_hwc(
                xrec.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
            )
            file_idx = int(sample["file_idx"])
            minute_id = int(sample["minute_id"])
            stem = f"sample_{sample_num:02d}_file{file_idx:02d}_minute{minute_id:03d}"

            save_order_image_matrices(original_order_image, originals_dir / f"{stem}_original.png")
            save_order_image_matrices(reconstruction_order_image, reconstructions_dir / f"{stem}_reconstruction.png")

            print(
                f"Saved sample {sample_num + 1}/{total}: dataset_idx={int(dataset_idx)}, file_idx={file_idx}, minute_id={minute_id}",
                flush=True,
            )


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(
        data_dir=args.data_dir.resolve(),
        converter_json_path=args.converter_json_path.resolve(),
    )
    model = load_vqgan_checkpoint(
        ckpt_path=args.ckpt_path.resolve(),
        vq_config=args.vq_config.resolve(),
        learning_rate=float(args.learning_rate),
        device=args.device,
    )

    reconstruct_samples(
        model=model,
        dataset=dataset,
        num_samples=int(args.num_samples),
        output_dir=output_dir,
        device=args.device,
        seed=args.seed,
    )

    print(f"Finished. Images saved under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
