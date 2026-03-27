from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from custom_code.training.heads.train_Heads_hypersearch import (
    HeadsChunkBatchSampler,
    HeadsLightningModule,
)
from market_simulation.models.utils_heads import OnlineReturnHeadDataset, VQRuntimeConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS = ["unprofitable", "unclear", "profitable"]


def collate_batch(batch):
    keys = batch[0].keys()
    return {key: torch.stack([sample[key].contiguous().clone() for sample in batch], dim=0) for key in keys}


def save_confusion_matrix(confusion: torch.Tensor, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = confusion.cpu().numpy()
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=20, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color="black")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads_ckpt", required=True)
    parser.add_argument("--order_model_ckpt", default=None)
    parser.add_argument("--order_batch_ckpt", default=None)
    parser.add_argument("--vq_ckpt_dir", default=None)
    parser.add_argument("--latent_diffusion_root", default=str(REPO_ROOT / "third_party" / "latent_diffusion"))
    parser.add_argument("--taming_root", default=str(REPO_ROOT / "third_party" / "taming-transformers"))
    parser.add_argument("--vq_config_relpath", default="latent_diffusion/models/first_stage_models/vq-f4/config.yaml")
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "test"))
    parser.add_argument("--pattern", default="*messages*.parquet")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", default=str(Path(__file__).resolve().parent / "confusion_matrix.png"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_kwargs = {"map_location": device, "strict": True}
    if args.order_model_ckpt is not None:
        load_kwargs["order_model_ckpt"] = args.order_model_ckpt
    if args.order_batch_ckpt is not None:
        load_kwargs["order_batch_ckpt"] = args.order_batch_ckpt
    model = HeadsLightningModule.load_from_checkpoint(args.heads_ckpt, **load_kwargs)
    model.to(device)
    model.eval()

    vq_runtime = None
    if model.hparams.scenario in {"order_batch", "both"}:
        if args.vq_ckpt_dir is None:
            raise ValueError("--vq_ckpt_dir is required for scenario using order-batch tokens")
        vq_runtime = VQRuntimeConfig(
            ckpt_dir=args.vq_ckpt_dir,
            latent_diffusion_root=args.latent_diffusion_root,
            taming_root=args.taming_root,
            config_relpath=args.vq_config_relpath,
            use_autocast=True,
        )

    dataset = OnlineReturnHeadDataset(
        message_files=[str(path) for path in sorted(Path(args.data_dir).glob(args.pattern))],
        seq_len=int(model.hparams.get("seq_len", 1024)),
        scenario=str(model.hparams.scenario),
        horizon_seconds=int(model.hparams.get("horizon_seconds", 30)),
        cache_size=2,
        feature_chunk_size=128,
        sample_chunk_size=128,
        vq_runtime=vq_runtime,
    )

    sampler = HeadsChunkBatchSampler(
        dataset,
        batch_size=int(args.batch_size),
        num_samples=min(int(args.num_samples), len(dataset)),
        chunk_size=128,
        seed=int(args.seed),
        drop_last=False,
        resample_each_iter=False,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_batch, num_workers=0, pin_memory=True)

    total = 0
    correct = 0
    confusion = torch.zeros((3, 3), dtype=torch.long)

    with torch.inference_mode():
        for batch in loader:
            order_features, batch_features = model._encode_features(batch)
            outputs = model.model(order_features=order_features, batch_features=batch_features)
            probs = torch.softmax(outputs["profit_logits"], dim=-1)
            pred = torch.argmax(probs, dim=-1)
            true = model._profit_targets(batch).cpu()
            pnl = model._pnl_targets(batch).cpu()

            pred_cpu = pred.cpu()
            probs_cpu = probs.cpu()

            for i in range(pred_cpu.size(0)):
                t = int(true[i])
                p = int(pred_cpu[i])
                confusion[t, p] += 1
                total += 1
                correct += int(t == p)
                print(
                    f"sample={total - 1} "
                    f"true={LABELS[t]} "
                    f"pred={LABELS[p]} "
                    f"p_unprof={float(probs_cpu[i, 0]):.4f} "
                    f"p_unclear={float(probs_cpu[i, 1]):.4f} "
                    f"p_prof={float(probs_cpu[i, 2]):.4f} "
                    f"pnl={float(pnl[i]):.1f}"
                )

    print(f"accuracy={(correct / total):.6f}" if total else "accuracy=nan")
    output_path = Path(args.output_path)
    save_confusion_matrix(confusion, output_path)
    print(f"confusion_matrix_path={output_path}")


if __name__ == "__main__":
    main()


# Example:
# python /home/random/projects/MarS/custom_code/testing/heads/profitable_trade_probability.py \
#   --heads_ckpt /home/random/projects/MarS/mars_runs/heads/tensorboard/head=probability_scenario=order_model_side=long_bs=8_lr=1e-4_hd=128/step=2880-val_loss=6.40123367e-01.ckpt \
#   --order_model_ckpt /home/random/projects/MarS/mars_runs/order_model/tensorboard/bs=8_lr=1e-4/step=step=13920-val=val_loss=3.2903.ckpt \
#   --data_dir /home/random/projects/MarS/data/test \
#   --num_samples 64 \
#   --output_path /home/random/projects/MarS/custom_code/testing/heads/confusion_matrix.png
