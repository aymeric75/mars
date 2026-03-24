from __future__ import annotations

import argparse
import sys

from pathlib import Path

import numpy as np
import torch

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from market_simulation.models.utils_vqgan import RawMinuteOrderImageDataset, instantiate_vq_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a trained VQGAN checkpoint, reconstruct 5 order images from data/test, and save them.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        default=REPO_ROOT / "mars_runs" / "vqgan" / "tensorboard" / "bs=8_lr=4.5e-6" / "step=4846-val_rec_loss=0.041324.ckpt",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=REPO_ROOT / "data" / "test",
    )
    parser.add_argument(
        "--converter_json_path",
        type=Path,
        default=REPO_ROOT / "custom_code" / "training" / "vqgan" / "converters_portable.json",
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


def image_to_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    image = np.clip((image + 1.0) * 0.5, 0.0, 1.0)
    image = np.rint(image * 255.0).astype(np.uint8)
    return image


def save_rgb_image(image: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(out_path)


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
) -> None:
    originals_dir = output_dir / "originals"
    reconstructions_dir = output_dir / "reconstructions"

    total = min(num_samples, len(dataset))
    if total == 0:
        raise RuntimeError("The dataset is empty; nothing to reconstruct.")

    with torch.no_grad():
        for idx in range(total):
            sample = dataset[idx]
            image_hwc = sample["image"].astype(np.float32, copy=False)
            batch = {"image": torch.from_numpy(image_hwc).unsqueeze(0).to(device)}
            x = model.model.get_input(batch, model.model.image_key)
            xrec, _, _ = model.model(x, return_pred_indices=True)

            original_uint8 = image_to_uint8_hwc(image_hwc)
            reconstruction_uint8 = image_to_uint8_hwc(
                xrec.detach().cpu().squeeze(0).permute(1, 2, 0).numpy(),
            )

            file_idx = int(sample["file_idx"])
            minute_id = int(sample["minute_id"])
            stem = f"sample_{idx:02d}_file{file_idx:02d}_minute{minute_id:03d}"

            save_rgb_image(original_uint8, originals_dir / f"{stem}_original.png")
            save_rgb_image(reconstruction_uint8, reconstructions_dir / f"{stem}_reconstruction.png")

            print(
                f"Saved sample {idx + 1}/{total}: file_idx={file_idx}, minute_id={minute_id}",
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
    )

    print(f"Finished. Images saved under {output_dir}", flush=True)


if __name__ == "__main__":
    main()
