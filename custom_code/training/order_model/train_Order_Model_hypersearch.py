import argparse
import os
import random
import glob
import lightning.pytorch as pl
import torch
import numpy as np

from torch.utils.data import DataLoader, Subset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from market_simulation.models.utils_order_model import (
    RawMessagesTokenDataset,
    build_model_from_variant,
    lm_loss_all_positions,
    unzip_zarr_zips,
)

class OrderBatchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str,
        batch_size: int,
        num_workers: int,
        seed: int = 7,
        pattern: str = "*_messages.parquet",
        feature_cols: int = 15,
        seq_len: int = 1024,
        cache_size: int = 4,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pattern = pattern
        self.feature_cols = int(feature_cols)
        self.seed = seed
        self.seq_len = int(seq_len)
        self.cache_size = int(cache_size)
        self._train = None
        self._val = None

    def prepare_data(self):
        # nothing to unzip anymore
        pass

    def setup(self, stage: str | None = None):
        self._train = RawMessagesTokenDataset(
            parquet_dir=self.train_dir,
            pattern=self.pattern,
            seq_len=self.seq_len,
            cache_size=self.cache_size,
        )
        self._val = RawMessagesTokenDataset(
            parquet_dir=self.val_dir,
            pattern=self.pattern,
            seq_len=self.seq_len,
            cache_size=self.cache_size,
        )


    @staticmethod
    def seed_worker(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )


    def val_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=self.seed_worker,
            generator=g,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=2 if self.num_workers > 0 else None,
        )



class OrderLightningModule(pl.LightningModule):
    def __init__(self, model_variant: str, K: int, lr: float):
        super().__init__()
        self.save_hyperparameters()

        self.model, _ = build_model_from_variant(model_variant, K=int(K))
        self.lr = float(lr)

    def forward(self, X):
        return self.model(X)

    def training_step(self, batch, batch_idx):
        X = batch
        logits = self(X)
        loss = lm_loss_all_positions(logits, X)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        X = batch
        logits = self(X)
        loss = lm_loss_all_positions(logits, X)
        self.log("val_loss", loss, on_step=False, sync_dist=False, prog_bar=False)
        return loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True)
    p.add_argument("--val_dir", required=True)
    p.add_argument("--pattern", default="*_messages.parquet")
    p.add_argument("--feature_cols", type=int, default=15)
    p.add_argument("--cache_size", type=int, default=4)
    p.add_argument("--model_variant", default="base", choices=["base", "small"])
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--run_root", default="checkpoints_order_model")
    p.add_argument("--run_name", default=None)

    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)

    dm = OrderBatchDataModule(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        pattern=args.pattern,
        feature_cols=args.feature_cols,
        seq_len=args.K,
        cache_size=args.cache_size,
    )

    model = OrderLightningModule(model_variant=args.model_variant, K=args.K, lr=args.lr)


    run_dir = args.run_root
    os.makedirs(run_dir, exist_ok=True)

    run_name = args.run_name or f"bs={args.batch_size}_lr={args.lr:g}"

    logger = TensorBoardLogger(
        save_dir=run_dir,
        name="tensorboard",
        version=run_name,   # <--- makes each run distinct in one TB logdir
    )



    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,                          # <--- checkpoints saved HERE
        filename="step={step}-val={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=logger,
        callbacks=[ckpt_cb],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.is_available() else "auto",
        max_steps=args.max_steps,
        precision=args.precision,
        log_every_n_steps=4,
        val_check_interval=120,
        limit_val_batches=10,
        deterministic=True,
        enable_checkpointing=True,
        enable_progress_bar=False,
    )


    trainer.fit(model, dm)


if __name__ == "__main__":
    # Lightning uses this env var in some cluster setups; harmless otherwise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
