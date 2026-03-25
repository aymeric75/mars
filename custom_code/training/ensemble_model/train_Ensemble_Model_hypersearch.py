from __future__ import annotations

import argparse
import os
import re

from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from custom_code.training.order_model.train_Order_Model_hypersearch import OrderLightningModule
from custom_code.training.utils import TextProgressCallback
from market_simulation.models.ensemble_model import EnsembleModel
from market_simulation.models.utils_ensemble_model import (
    EnsembleChunkBatchSampler,
    OnlineEnsembleDataset,
)
from market_simulation.models.utils_vqgan import instantiate_vq_model


REPO_ROOT = Path(__file__).resolve().parents[3]


def find_best_checkpoint(ckpt_dir: str | Path, metric_name: str = "val_loss") -> Path:
    ckpt_dir = Path(ckpt_dir)
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    pattern = re.compile(rf"{re.escape(metric_name)}=([0-9]+(?:\.[0-9]+)?)")

    def extract_metric(path: Path) -> float:
        match = pattern.search(path.name)
        return float(match.group(1)) if match else float("inf")

    return min(ckpts, key=extract_metric)


class EnsembleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        train_dir: str,
        val_dir: str,
        pattern: str,
        seq_len: int,
        batch_size: int,
        num_workers: int,
        cache_size: int,
        train_num_samples: int | None,
        train_chunk_size: int,
        val_num_samples: int | None,
        val_chunk_size: int | None,
        seed: int,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.pattern = pattern
        self.seq_len = int(seq_len)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.cache_size = int(cache_size)
        self.train_num_samples = None if train_num_samples is None else int(train_num_samples)
        self.train_chunk_size = int(train_chunk_size)
        self.val_num_samples = None if val_num_samples is None else int(val_num_samples)
        self.val_chunk_size = self.train_chunk_size if val_chunk_size is None else int(val_chunk_size)
        self.seed = int(seed)
        self.train_ds: OnlineEnsembleDataset | None = None
        self.val_ds: OnlineEnsembleDataset | None = None

    def setup(self, stage: str | None = None):
        train_files = sorted(Path(self.train_dir).glob(self.pattern))
        val_files = sorted(Path(self.val_dir).glob(self.pattern))
        self.train_ds = OnlineEnsembleDataset(
            message_files=[str(path) for path in train_files],
            seq_len=self.seq_len,
            cache_size=self.cache_size,
            feature_chunk_size=self.train_chunk_size,
            sample_chunk_size=self.train_chunk_size,
        )
        self.val_ds = OnlineEnsembleDataset(
            message_files=[str(path) for path in val_files],
            seq_len=self.seq_len,
            cache_size=self.cache_size,
            feature_chunk_size=self.val_chunk_size,
            sample_chunk_size=self.val_chunk_size,
        )

    @staticmethod
    def collate_batch(batch):
        contexts = torch.stack([sample["context"].contiguous().clone() for sample in batch], dim=0)
        targets = torch.stack([sample["target"].contiguous().clone() for sample in batch], dim=0)
        next_images = torch.stack([sample["next_image"].contiguous().clone() for sample in batch], dim=0)
        return {
            "context": contexts,
            "target": targets,
            "next_image": next_images,
        }

    def train_dataloader(self):
        batch_sampler = EnsembleChunkBatchSampler(
            self.train_ds,
            batch_size=self.batch_size,
            num_samples=self.train_num_samples,
            chunk_size=self.train_chunk_size,
            seed=self.seed,
            drop_last=True,
        )
        return DataLoader(
            self.train_ds,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        batch_sampler = EnsembleChunkBatchSampler(
            self.val_ds,
            batch_size=self.batch_size,
            num_samples=self.val_num_samples,
            chunk_size=self.val_chunk_size,
            seed=self.seed,
            drop_last=False,
        )
        return DataLoader(
            self.val_ds,
            batch_sampler=batch_sampler,
            collate_fn=self.collate_batch,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


class EnsembleLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        order_vocab_size: int,
        order_model_ckpt: str,
        vq_ckpt_dir: str,
        latent_diffusion_root: str,
        lr: float,
        batch_vocab_size: int = 8192,
        batch_tokens_len: int = 64,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        vq_config_relpath: str = "latent_diffusion/models/first_stage_models/vq-f4/config.yaml",
        vq_metric_name: str = "val_rec_loss",
        vq_use_autocast: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = EnsembleModel(
            order_vocab_size=int(order_vocab_size),
            batch_vocab_size=int(batch_vocab_size),
            batch_tokens_len=int(batch_tokens_len),
            d_model=int(d_model),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=float(dropout),
        )
        self.lr = float(lr)

        self.__dict__["_order_model_runtime"] = None
        self.__dict__["_vq_model_runtime"] = None
        self.__dict__["_vq_device"] = None

    def forward(self, base_logits: torch.Tensor, batch_tokens: torch.Tensor) -> torch.Tensor:
        return self.model(base_logits=base_logits, batch_tokens=batch_tokens)

    def _ensure_runtime_models(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        order_model = self.__dict__.get("_order_model_runtime")
        vq_model = self.__dict__.get("_vq_model_runtime")
        if order_model is not None and vq_model is not None:
            return order_model, vq_model

        device = self.trainer.strategy.root_device if self.trainer is not None else self.device

        order_lm = OrderLightningModule.load_from_checkpoint(
            self.hparams.order_model_ckpt,
            map_location="cpu",
            strict=True,
        )
        order_model = order_lm.model.to(device)
        order_model.eval()
        for param in order_model.parameters():
            param.requires_grad_(False)

        vq_ckpt = find_best_checkpoint(self.hparams.vq_ckpt_dir, metric_name=self.hparams.vq_metric_name)
        latent_root = Path(self.hparams.latent_diffusion_root).resolve()
        config_path = latent_root.parent / self.hparams.vq_config_relpath
        vq_model = instantiate_vq_model(
            config_path=config_path,
            init_ckpt=str(vq_ckpt),
            learning_rate=0.0,
        ).model
        vq_model.to(device)
        vq_model.eval()
        for param in vq_model.parameters():
            param.requires_grad_(False)

        self.__dict__["_order_model_runtime"] = order_model
        self.__dict__["_vq_model_runtime"] = vq_model
        self.__dict__["_vq_device"] = device
        return order_model, vq_model

    def _encode_next_images(self, next_images: torch.Tensor) -> torch.Tensor:
        _, vq_model = self._ensure_runtime_models()
        device = self.__dict__["_vq_device"]
        next_images = next_images.to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.no_grad():
            if device.type == "cuda" and self.hparams.vq_use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, _, info = vq_model.encode(next_images)
            else:
                _, _, info = vq_model.encode(next_images)

        tokens = info[2]
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        return tokens.view(next_images.size(0), -1).to(dtype=torch.long)

    def _shared_step(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        order_model, _ = self._ensure_runtime_models()

        context = batch["context"].to(device=self.device, dtype=torch.long, non_blocking=True)
        target = batch["target"].to(device=self.device, dtype=torch.long, non_blocking=True)
        next_images = batch["next_image"]

        with torch.no_grad():
            base_logits = order_model(context)[:, -1, :]
            next_tokens = self._encode_next_images(next_images)

        refined_logits = self(base_logits=base_logits, batch_tokens=next_tokens)
        loss = F.cross_entropy(refined_logits, target)
        base_loss = F.cross_entropy(base_logits, target)
        advantage = base_loss - loss

        on_step = stage == "train"
        self.log(f"{stage}_loss", loss, on_step=on_step, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log(f"{stage}_base_loss", base_loss, on_step=on_step, on_epoch=True, sync_dist=True, prog_bar=False)
        self.log(
            f"{stage}_loss_advantage",
            advantage,
            on_step=on_step,
            on_epoch=True,
            sync_dist=True,
            prog_bar=False,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--pattern", default="*messages*.parquet")
    parser.add_argument("--run_root", default="")
    parser.add_argument("--run_name", default=None)

    parser.add_argument("--order_model_ckpt", required=True)
    parser.add_argument("--vq_ckpt_dir", required=True)
    parser.add_argument("--latent_diffusion_root", default=str(REPO_ROOT / "third_party" / "latent_diffusion"))
    parser.add_argument(
        "--vq_config_relpath",
        default="latent_diffusion/models/first_stage_models/vq-f4/config.yaml",
    )
    parser.add_argument("--vq_metric_name", default="val_rec_loss")
    parser.add_argument("--vq_use_autocast", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--order_vocab_size", type=int, default=49152)
    parser.add_argument("--batch_vocab_size", type=int, default=8192)
    parser.add_argument("--batch_tokens_len", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cache_size", type=int, default=2)
    parser.add_argument("--train_num_samples", type=int, default=None)
    parser.add_argument("--train_chunk_size", type=int, default=128)
    parser.add_argument("--val_num_samples", type=int, default=256)
    parser.add_argument("--val_chunk_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    parser.add_argument("--matmul_precision", default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--gradient_clip_algorithm", default="norm", choices=["norm", "value"])
    parser.add_argument("--val_check_interval", type=int, default=200)
    parser.add_argument("--limit_val_batches", type=int, default=64)
    parser.add_argument("--print_every_n_steps", type=int, default=10)
    args = parser.parse_args()

    run_root = args.run_root or str(REPO_ROOT / "mars_runs" / "ensemble_model")
    os.makedirs(run_root, exist_ok=True)
    run_name = args.run_name or f"bs={args.batch_size}_lr={args.lr:g}"
    run_dir = os.path.join(run_root, "tensorboard", run_name)
    os.makedirs(run_dir, exist_ok=True)

    for filename in os.listdir(run_dir):
        if filename.endswith(".ckpt"):
            os.remove(os.path.join(run_dir, filename))

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not args.deterministic

    dm = EnsembleDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        pattern=args.pattern,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cache_size=args.cache_size,
        train_num_samples=args.train_num_samples,
        train_chunk_size=args.train_chunk_size,
        val_num_samples=args.val_num_samples,
        val_chunk_size=args.val_chunk_size,
        seed=args.seed,
    )

    model = EnsembleLightningModule(
        order_vocab_size=args.order_vocab_size,
        order_model_ckpt=args.order_model_ckpt,
        vq_ckpt_dir=args.vq_ckpt_dir,
        latent_diffusion_root=args.latent_diffusion_root,
        vq_config_relpath=args.vq_config_relpath,
        vq_metric_name=args.vq_metric_name,
        vq_use_autocast=args.vq_use_autocast,
        batch_vocab_size=args.batch_vocab_size,
        batch_tokens_len=args.batch_tokens_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        lr=args.lr,
    )

    logger = TensorBoardLogger(save_dir=run_root, name="tensorboard", version=run_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="step={step}-val={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )
    progress_cb = TextProgressCallback(print_every_n_steps=args.print_every_n_steps)

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=logger,
        callbacks=[ckpt_cb, progress_cb],
        enable_checkpointing=True,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy="auto",
        max_steps=args.max_steps,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        log_every_n_steps=4,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        deterministic=args.deterministic,
        enable_progress_bar=False,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
