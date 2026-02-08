from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only

from market_simulation.models.ensemble_model import EnsembleModel
from market_simulation.models.utils_ensemble_model import (
    MultiFileEnsembleDataset,
    unzip_zarr_zips_inplace,
)


class EnsembleDataModule(L.LightningDataModule):
    def __init__(
        self,
        parquets_dir: str | Path,
        next1s_dir: str | Path,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.01,
        seed: int = 42,
    ):
        super().__init__()
        self.parquets_dir = Path(parquets_dir)
        self.next1s_dir = Path(next1s_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.ds_train = None
        self.ds_val = None

    @rank_zero_only
    def prepare_data(self):
        unzip_zarr_zips_inplace(self.next1s_dir)

    def setup(self, stage: Optional[str] = None):
        ds = MultiFileEnsembleDataset(
            parquets_dir=self.parquets_dir,
            next1s_dir=self.next1s_dir,
            parquet_pattern="features_*_cut.parquet",
            next1_pattern="next1_tokens_*.zarr",
        )

        n_total = len(ds)
        n_val = max(1, int(n_total * self.val_split))
        n_train = n_total - n_val

        g = torch.Generator().manual_seed(self.seed)
        self.ds_train, self.ds_val = random_split(ds, [n_train, n_val], generator=g)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class EnsembleLightningModule(L.LightningModule):
    # wrapper like OrderBatchLightningModule
    def __init__(self, order_vocab_size: int, lr: float = 3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = EnsembleModel(order_vocab_size=order_vocab_size)

    def forward(self, features, batch_tokens):
        return self.model(features, batch_tokens)
        
        
    @staticmethod
    def _loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # expects logits (B, 64, V) and targets (B, 64)
        if logits.ndim != 3:
            raise RuntimeError(f"Expected logits (B, 64, V), got {tuple(logits.shape)}")
        if targets.ndim != 2:
            raise RuntimeError(f"Expected targets (B, 64), got {tuple(targets.shape)}")
        B, T, V = logits.shape
        return F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))

    def training_step(self, batch, batch_idx):
        features, batch_tokens = batch
        features = torch.as_tensor(features, device=self.device)
        batch_tokens = torch.as_tensor(batch_tokens, device=self.device, dtype=torch.long)
    
        logits = self(features, batch_tokens)
        loss = self._loss_from_logits(logits, batch_tokens)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        features, batch_tokens = batch
        features = torch.as_tensor(features, device=self.device)
        batch_tokens = torch.as_tensor(batch_tokens, device=self.device, dtype=torch.long)
    
        logits = self(features, batch_tokens)
        loss = self._loss_from_logits(logits, batch_tokens)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--parquets_dir", type=str, required=True)
    p.add_argument("--next1s_dir", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_split", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--precision", type=str, default="bf16-mixed")

    # REQUIRED by your EnsembleModel ctor
    p.add_argument("--order_vocab_size", type=int, required=True)

    # Batch-order style: everything under this
    p.add_argument("--default_root_dir", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    L.seed_everything(args.seed, workers=True)

    run_root = Path(args.default_root_dir)
    logger = TensorBoardLogger(save_dir=str(run_root), name="lightning_logs")

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_root / "checkpoints"),
        filename="epoch{epoch:03d}-val{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    dm = EnsembleDataModule(
        parquets_dir=args.parquets_dir,
        next1s_dir=args.next1s_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
    )

    model = EnsembleLightningModule(order_vocab_size=args.order_vocab_size, lr=args.lr)

    trainer = L.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="ddp",
        max_epochs=args.max_epochs,
        precision=args.precision,
        default_root_dir=str(run_root),
        logger=logger,
        callbacks=[ckpt_cb, LearningRateMonitor(logging_interval="step")],
        log_every_n_steps=50,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
