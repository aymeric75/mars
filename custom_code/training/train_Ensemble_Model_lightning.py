# custom_code/training/train_Ensemble_Model_lightning.py
from __future__ import annotations

import argparse
import os
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import zipfile
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset, random_split
from market_simulation.models.ensemble_model import EnsembleModel
from market_simulation.models.utils_ensemble_model import MultiFileEnsembleDataset
from pathlib import Path
from typing import Optional


from lightning.pytorch.utilities import rank_zero_only



def _ensure_unzipped_zarrs(next1s_dir: Path, unzip_dirname: str = "_unzipped") -> Path:
    """
    Unzips all `next1_tokens_*.zarr.zip` into `next1s_dir/_unzipped/` (idempotent),
    and returns the directory that contains extracted `.zarr/` folders.
    """
    next1s_dir = Path(next1s_dir)
    out_dir = next1s_dir / unzip_dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    for z in sorted(next1s_dir.glob("next1_tokens_*.zarr.zip")):
        with zipfile.ZipFile(z, "r") as zf:
            # assume zip contains one top-level ".zarr/" folder
            top = zf.namelist()[0].split("/")[0]
            target = out_dir / top
            if target.exists():
                continue
            zf.extractall(out_dir)

    return out_dir


class EnsembleDataModule(L.LightningDataModule):
    def __init__(
        self,
        parquets_dir: str | Path,
        next1s_dir: str | Path,
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.01,
        seed: int = 42,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.parquets_dir = Path(parquets_dir)
        self.next1s_dir = Path(next1s_dir)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        self._effective_next1s_dir: Optional[Path] = None
        self.ds_train = None
        self.ds_val = None

    @rank_zero_only
    def prepare_data(self):
        # unzip once (only on rank 0), other ranks will see extracted folders in setup()
        self._effective_next1s_dir = _ensure_unzipped_zarrs(self.next1s_dir)

    def setup(self, stage: Optional[str] = None):
        # In case prepare_data wasn't called (or for safety), ensure path is set
        if self._effective_next1s_dir is None:
            # If already unzipped, this is cheap (just checks existence)
            self._effective_next1s_dir = _ensure_unzipped_zarrs(self.next1s_dir)

        ds = MultiFileEnsembleDataset(
            parquets_dir=self.parquets_dir,
            next1s_dir=self._effective_next1s_dir,          # <- use extracted .zarr folders
            parquet_pattern="features_*_cut.parquet",
            next1_pattern="next1_tokens_*.zarr",
        )

        n_total = len(ds)
        n_val = max(1, int(n_total * self.val_split))
        n_train = n_total - n_val
        self.ds_train, self.ds_val = random_split(
            ds, [n_train, n_val], generator=None
        )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )






def _ce_64tokens(logits_bt_v: torch.Tensor, target_bt: torch.Tensor) -> torch.Tensor:
    """
    logits_bt_v: (B, T=64, V)
    target_bt:   (B, T=64) long
    """
    B, T, V = logits_bt_v.shape
    return F.cross_entropy(logits_bt_v.reshape(B * T, V), target_bt.reshape(B * T), reduction="mean")


class EnsembleLightningModule(pl.LightningModule):
    """
    Trains an EnsembleModel to refine a base distribution (base_logits) using next1 batch tokens.

    With your current dataset (features15, next1_tokens64) we need base_logits.
    Minimal bridge (default): learn base_logits from features via Linear(15 -> V).

    If later you want paper-faithful training:
      - replace base_logits = base_head(features) with frozen OrderModel logits
        (either computed online with another dataset containing order contexts,
         or precomputed and loaded from disk).
    """

    def __init__(
        self,
        order_vocab_size: int,
        batch_vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.ensemble = EnsembleModel(
            order_vocab_size=int(order_vocab_size),
            batch_vocab_size=int(batch_vocab_size),
            batch_tokens_len=64,
            d_model=int(d_model),
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            dropout=0.1,
        )

        # Minimal bridge: learnable base logits from 15 integer features
        self.base_head = torch.nn.Linear(15, int(order_vocab_size), bias=True)

        self.lr = float(lr)

    def training_step(self, batch, batch_idx):
        feats15, next1_tokens = batch  # (B,15) long, (B,64) long

        # base logits (B,V) from features
        base_logits = self.base_head(feats15.float())

        # refine logits (B,V)
        refined_logits = self.ensemble(base_logits=base_logits, batch_tokens=next1_tokens)

        # We need a target for CE. With your current data, the natural target is:
        # "predict the next1 image tokens". But refined_logits is for order vocab (V), not batch vocab.
        #
        # Therefore: by default, we train the ensemble to improve the *base logits* objective
        # only if you supply targets in that vocab. Since you don't (yet), we provide a clean, usable
        # default: train a decoder from refined_logits to 64-token batch vocab.
        #
        # This keeps the pipeline runnable now, and you can later replace this head with the real
        # paper objective (next order token likelihood).
        #
        # If you *don't* want this, delete batch_head and change loss accordingly.

        # Map refined order-side state -> batch token logits (B,64,batch_vocab)
        # simplest: broadcast a per-sample vector to 64 positions then linear.
        x = refined_logits.unsqueeze(1).expand(-1, 64, -1)  # (B,64,V)
        batch_head = getattr(self, "batch_head", None)
        if batch_head is None:
            self.batch_head = torch.nn.Linear(x.size(-1), self.hparams.batch_vocab_size, bias=True).to(x.device)
            batch_head = self.batch_head
        logits_bt_v = batch_head(x)  # (B,64,BV)

        loss = _ce_64tokens(logits_bt_v, next1_tokens)
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feats15, next1_tokens = batch
        base_logits = self.base_head(feats15.float())
        refined_logits = self.ensemble(base_logits=base_logits, batch_tokens=next1_tokens)
        x = refined_logits.unsqueeze(1).expand(-1, 64, -1)
        logits_bt_v = self.batch_head(x)  # created in training_step
        loss = _ce_64tokens(logits_bt_v, next1_tokens)
        self.log("val_loss", loss, on_step=False, sync_dist=False, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--parquets_dir", default="../../data/ensemble/parquets")
    p.add_argument("--next1s_dir", default="../../data/ensemble/next1s")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--val_frac", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)

    # model
    p.add_argument("--order_vocab_size", type=int, required=True)   # V
    p.add_argument("--batch_vocab_size", type=int, default=8192)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)

    # train
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=20000)
    p.add_argument("--precision", default="bf16-mixed", choices=["32-true", "16-mixed", "bf16-mixed"])

    args = p.parse_args()

    pl.seed_everything(args.seed, workers=True)

    dm = EnsembleDataModule(
        parquets_dir=args.parquets_dir,
        next1s_dir=args.next1s_dir,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = EnsembleLightningModule(
        order_vocab_size=args.order_vocab_size,
        batch_vocab_size=args.batch_vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        lr=args.lr,
    )

    run_dir = "checkpoints_ensemble_model"
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
