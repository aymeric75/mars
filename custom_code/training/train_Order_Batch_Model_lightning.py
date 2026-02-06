# train_Order_Batch_Model_lightning.py
import argparse
import glob
import os

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset

from market_simulation.models.order_batch_model import OrderBatchModel
from market_simulation.models.utils_order_batch_model import MultiDirZarrTokenDataset, lm_loss_next_token
from market_simulation.models.utils_order_model import unzip_zarr_zips


class OrderBatchDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        pattern: str,
        array_path: str,
        batch_size: int,
        val_frac: float,
        seed: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.pattern = pattern
        self.array_path = array_path
        self.batch_size = int(batch_size)
        self.val_frac = float(val_frac)
        self.seed = int(seed)
        self.num_workers = int(num_workers)

        self._train = None
        self._val = None

    def prepare_data(self):
        # unzip *.zarr.zip -> *.zarr (safe if already extracted)
        unzip_zarr_zips(self.train_dir, self.pattern)

    def setup(self, stage: str | None = None):
        # use extracted dirs only
        zarr_dirs = [p[:-4] for p in glob.glob(os.path.join(self.train_dir, self.pattern))]
        zarr_dirs = [d for d in zarr_dirs if os.path.isdir(d)]

        ds = MultiDirZarrTokenDataset(zarr_dirs=zarr_dirs, array_path=self.array_path)

        n = len(ds)
        n_val = max(1, int(n * self.val_frac))
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()

        self._val = Subset(ds, perm[:n_val])
        self._train = Subset(ds, perm[n_val:])

    def train_dataloader(self):
        return DataLoader(
            self._train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        return DataLoader(
            self._val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )


class OrderBatchLightningModule(pl.LightningModule):
    def __init__(self, emb_dim: int, num_layers: int, num_heads: int, vocab_size: int, lr: float):
        super().__init__()
        self.save_hyperparameters()

        self.model = OrderBatchModel(
            emb_dim=int(emb_dim),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            vocab_size=int(vocab_size),
        )
        self.lr = float(lr)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)  # (B,T,V)

    def training_step(self, batch, batch_idx):
        input_ids = batch  # (B,1024)
        logits = self(input_ids)
        loss = lm_loss_next_token(logits, input_ids)
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch
        logits = self(input_ids)
        loss = lm_loss_next_token(logits, input_ids)
        self.log("val_loss", loss, on_step=False, sync_dist=False, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", default="../../data/order_batch_model/past16s")
    p.add_argument("--pattern", default="past16_tokens_*.zarr.zip")
    p.add_argument("--array_path", default="arr_0")
    p.add_argument("--batch_size", type=int, default=8)

    p.add_argument("--emb_dim", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--vocab_size", type=int, default=8192)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)

    dm = OrderBatchDataModule(
        train_dir=args.train_dir,
        pattern=args.pattern,
        array_path=args.array_path,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = OrderBatchLightningModule(
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        lr=args.lr,
    )

    run_dir = "checkpoints_batch_order_model"
    os.makedirs(run_dir, exist_ok=True)

    logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard")

    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
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
        log_every_n_steps=10,
        val_check_interval=200,
        limit_val_batches=10,
        enable_checkpointing=True,
        enable_progress_bar=False,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
