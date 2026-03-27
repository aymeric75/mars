from __future__ import annotations

import argparse
from pathlib import Path

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
    print("confusion_matrix=true_rows_pred_cols")
    for row in confusion.tolist():
        print(row)


if __name__ == "__main__":
    main()
