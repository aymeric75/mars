import argparse
import os
import glob
import lightning.pytorch as pl
import torch

from torch.utils.data import DataLoader, Subset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from market_simulation.models.utils_order_model import (
    MultiDirZarrOrderDataset,
    build_model_from_variant,
    lm_loss_all_positions,
    unzip_zarr_zips,
)


class OrderDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        pattern: str,
        K: int,
        batch_size: int,
        val_frac: float,
        seed: int,
        num_workers: int,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.pattern = pattern
        self.K = int(K)
        self.batch_size = int(batch_size)
        self.val_frac = float(val_frac)
        self.seed = int(seed)
        self.num_workers = int(num_workers)

        self._train = None
        self._val = None

    def prepare_data(self):
        # Extract once (safe to run on every rank)
        unzip_zarr_zips(self.train_dir, self.pattern)

    def setup(self, stage: str | None = None):

        
    
        # list already-extracted dirs (no unzip here)
        zarr_dirs = [p[:-4] for p in glob.glob(os.path.join(self.train_dir, self.pattern))]
        zarr_dirs = [d for d in zarr_dirs if os.path.isdir(d)]

        ds = MultiDirZarrOrderDataset(zarr_dirs, seq_len=self.K)

        n = len(ds)
        n_val = max(1, int(n * self.val_frac))
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()

        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        self._train = Subset(ds, train_idx)
        self._val = Subset(ds, val_idx)

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
    p.add_argument("--train_dir", default="../../data/order_model/train")
    p.add_argument("--pattern", default="*_features.zarr.zip")
    p.add_argument("--model_variant", default="base", choices=["base", "small"])
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])
    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)

    dm = OrderDataModule(
        train_dir=args.train_dir,
        pattern=args.pattern,
        K=args.K,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = OrderLightningModule(model_variant=args.model_variant, K=args.K, lr=args.lr)
    
        
    run_dir = "checkpoints_order_model"
    os.makedirs(run_dir, exist_ok=True)
    
    logger = TensorBoardLogger(
        save_dir=run_dir,          # <--- logs go under checkpoints_order_model/...
        name="tensorboard"         # .../tensorboard/version_0/
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
        enable_checkpointing=True,
        enable_progress_bar=False,
    )
    

    trainer.fit(model, dm)


if __name__ == "__main__":
    # Lightning uses this env var in some cluster setups; harmless otherwise.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
