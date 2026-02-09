import argparse
import os
import shutil
import io
import lobdatamanager as ldm
import lightning.pytorch as pl
import torch
#from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from market_simulation.models.ensemble_model import EnsembleModel
from market_simulation.models.utils_ensemble_model import (
    EnsembleTrainDataset,
    ensemble_training_step,
)




def save_checkpoint(model, optimizer, step: int, run_dir: str):
    """
    Manual checkpoint save (non-atomic, local).
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": step,
        "model_class": model.__class__.__name__,
    }

    ckpt_path = Path(run_dir) / f"ckpt_step={step}.pt"
    torch.save(ckpt, ckpt_path)






class EnsembleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        next64_path: str,
        logits_path: str,
        targets_path: str,
        batch_size: int,
        val_frac: float,
        seed: int,
        num_workers: int,
    ):
        super().__init__()
        self.next64_path = next64_path
        self.logits_path = logits_path
        self.targets_path = targets_path

        self.batch_size = int(batch_size)
        self.val_frac = float(val_frac)
        self.seed = int(seed)
        self.num_workers = int(num_workers)

    def setup(self, stage: str | None = None):
        ds = EnsembleTrainDataset(
            next64_path=self.next64_path,
            logits_path=self.logits_path,
            targets_path=self.targets_path,
        )

        n = len(ds)
        n_val = max(1, int(n * self.val_frac))
        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=g).tolist()

        self.train_ds = Subset(ds, perm[n_val:])
        self.val_ds = Subset(ds, perm[:n_val])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )






class EnsembleLightningModule(pl.LightningModule):
    def __init__(self, order_vocab_size: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = EnsembleModel(order_vocab_size=int(order_vocab_size))
        self.lr = float(lr)

    def training_step(self, batch, batch_idx):
        loss = ensemble_training_step(model=self.model, batch=batch)
        self.log("train_loss", loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = ensemble_training_step(model=self.model, batch=batch)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_validation_end(self):
        if not self.trainer.is_global_zero:
            return

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizers(),
            step=self.trainer.global_step,
            run_dir=self.trainer.default_root_dir,
        )



def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--next64_path", type=str, required=True)
    p.add_argument("--logits_path", type=str, required=True)
    p.add_argument("--targets_path", type=str, required=True)

    # hparams
    p.add_argument("--order_vocab_size", type=int, default=49152)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=10_000)
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--precision", default="bf16-mixed")
    p.add_argument("--default_root_dir", default="")
    args = p.parse_args()

    run_dir = args.default_root_dir or "checkpoints_ensemble_model"
    os.makedirs(run_dir, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    dm = EnsembleDataModule(
        next64_path=args.next64_path,
        logits_path=args.logits_path,
        targets_path=args.targets_path,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = EnsembleLightningModule(
        order_vocab_size=args.order_vocab_size,
        lr=args.lr,
    )

    logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard")

    """
    ckpt_cb = ModelCheckpoint(
        dirpath=run_dir,
        filename="val={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    """
    
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=logger,
        callbacks=[], #[ckpt_cb]
        enable_checkpointing=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto" if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.is_available() else "auto",
        max_steps=args.max_steps,
        precision=args.precision,
        log_every_n_steps=10,
        val_check_interval=10, #50,
        limit_val_batches=10,
        enable_progress_bar=False,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
